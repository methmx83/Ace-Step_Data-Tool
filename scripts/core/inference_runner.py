#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/core/inference_runner.py — Inferenz-Runner mit Retry und Content-Retry (inkrementell)

Kapselt:
- model_wrapper.chat Aufrufe mit Retry-Logik
- Content-basierte Nachbesserung (min. benötigte Items), optional inkrementell
- JSON-Parsing via helpers.json_parser.parse_category_response

Voraussetzung: Qwen2AudioWrapper-kompatibles Interface mit .chat(prompt, audio_files, max_new_tokens, temperature)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple, Callable
import time

from scripts.helpers.json_parser import parse_category_response
from scripts.helpers.logger_setup import get_session_logger, log_exception, log_model_response


class InferenceRunner:
    def __init__(
        self,
        model_wrapper,
        prompts_config: Dict[str, Any],
        count_fn: Callable[[str, Optional[Dict[str, Any]]], int],
        items_fn: Optional[Callable[[str, Optional[Dict[str, Any]]], List[str]]] = None,
        merge_fn: Optional[Callable[[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = None,
    ):
        """
        count_fn: Callable[[str, Optional[Dict[str, Any]]], int]
            Funktion, die die Anzahl sinnvoller Items in einer geparsten Response pro Kategorie zählt.
        items_fn: Callable[[str, Optional[Dict[str, Any]]], List[str]]
            Extrahiert die rohen Items aus der geparsten Response (für inkrementelles Prompting).
        merge_fn: Callable[[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]
            Vereinigt zwei Responses dedupliziert und strukturerhaltend.
        """
        self.model = model_wrapper
        self.cfg = prompts_config or {}
        self.count_items = count_fn
        self.items_fn = items_fn or self._default_items
        self.merge_items = merge_fn or self._default_merge
        self.logger = get_session_logger("InferenceRunner")

    def _get_content_retry_policy(self, category: str) -> Dict[str, Any]:
        wf = self.cfg.get("workflow_config", {})
        policy = wf.get("content_retry", {})
        overrides = (policy.get("overrides") or {}).get(category, {})
        return {
            "enabled": bool(policy.get("enabled", False)),
            "max_attempts": int(overrides.get("max_attempts", policy.get("max_attempts", 0))),
            "delay_seconds": float(overrides.get("delay_seconds", policy.get("delay_seconds", 0.25))),
            "temperature_boost": float(overrides.get("temperature_boost", policy.get("temperature_boost", 0.1))),
        }

    def run(self, *, category: str, prompt: str, audio_files: List[str], template: Dict[str, Any], min_required: int) -> Optional[Dict[str, Any]]:
        """
        Führt eine Kategorien-Inferenz aus, inkl. Retry und optionaler Content-Retry.
        """
        cat_logger = get_session_logger(f"Category.{category}")
        max_retries = int(template.get("retry_count", 2))
        max_tokens = int(template.get("max_tokens", 60))
        temperature = float(template.get("temperature", 0.0))

        parsed: Optional[Dict[str, Any]] = None
        best: Optional[Dict[str, Any]] = None
        have: int = 0

        # 1) Technischer Retry (robust gegen Netzwerk/Parsingprobleme)
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                raw = self.model.chat(
                    prompt=prompt,
                    audio_files=audio_files,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                dt = time.time() - t0
                cat_logger.debug(f"Model response time: {dt:.2f}s")

                parsed = parse_category_response(raw, category)
                log_model_response(cat_logger, category, prompt, raw, parsed)
                if parsed:
                    # initiale Akkumulation
                    best = parsed
                    have = self.count_items(category, best)
                    break
            except Exception as e:
                log_exception(cat_logger, f"{category} execution (attempt {attempt + 1})", e)
                if attempt < max_retries:
                    time.sleep(0.2 * (attempt + 1))

        # 2) Content-Retry, falls erforderlich
        policy = self._get_content_retry_policy(category)
        if not policy.get("enabled", False):
            # Falls vorhanden, gebe bereits partielle/initiale Ergebnisse zurück
            return best or parsed

        # Haben wir genug? Falls nicht, inkrementell nachfordern
        if best is None:
            have = 0
        if best is None or have < int(min_required):
            cr_logger = get_session_logger(f"ContentRetry.{category}")
            reason = "no parse" if best is None else f"have {have}/{min_required}"
            cr_logger.info(f"Content retry due to {reason}. Retrying up to {policy['max_attempts']} times…")

            retry_templates = self.cfg.get("retry_templates", {}) or {}
            base_temp = temperature
            attempts = int(policy.get("max_attempts", 0))
            delay = float(policy.get("delay_seconds", 0.25))
            tboost = float(policy.get("temperature_boost", 0.1))

            for i in range(attempts):
                try:
                    # Fehlende Anzahl gezielt anfordern (inkrementell)
                    missing = max(0, int(min_required) - int(have))
                    if missing <= 0:
                        return best
                    rt_text = (
                        retry_templates.get(category)
                        or retry_templates.get("incremental")
                        or retry_templates.get("default")
                    )
                    if not rt_text:
                        rt_text = (
                            "IMPORTANT: We already have: {existing}. Return ONLY {need} additional unique {category} item(s) as per the JSON schema. "
                            "Do not repeat existing items. JSON only, no explanations."
                        )
                    existing_items = self.items_fn(category, best)
                    # Liste im Prompt begrenzen, um Token zu sparen
                    if len(existing_items) > 20:
                        existing_items = existing_items[:20]
                    rt_text = (
                        rt_text.replace("{need}", str(missing))
                        .replace("{category}", str(category))
                        .replace("{existing}", ", ".join(existing_items))
                    )
                    reinforced_prompt = prompt + "\n\n" + rt_text

                    temp2 = base_temp + tboost * (i + 1)
                    raw = self.model.chat(
                        prompt=reinforced_prompt,
                        audio_files=audio_files,
                        max_new_tokens=max_tokens,
                        temperature=temp2,
                    )
                    parsed2 = parse_category_response(raw, category)
                    log_model_response(cr_logger, category, reinforced_prompt, raw, parsed2)
                    if parsed2:
                        candidate = self.merge_items(category, best, parsed2)
                        have2 = self.count_items(category, candidate)
                        cr_logger.debug(f"Merged items count: {have}->{have2}")
                        if have2 >= int(min_required):
                            return candidate
                        # Falls immer noch zu wenig, behalten wir das beste bisher
                        if not best or have2 > have:
                            best = candidate
                            have = have2
                except Exception as e:
                    log_exception(cr_logger, f"content retry {i+1}", e)
                time.sleep(delay)

        # Gib bestmögliches Ergebnis zurück (auch wenn Mindestanzahl verfehlt)
        return best or parsed

    # -------------------- Default-Hilfsfunktionen --------------------
    @staticmethod
    def _default_items(category: str, parsed: Optional[Dict[str, Any]]) -> List[str]:
        """Standard-Extraktion der Listenitems pro Kategorie.
        Für 'vocal' wird eine leere Liste zurückgegeben, da dort kein Listenfeld existiert.
        """
        if not parsed:
            return []
        cat = (category or "").lower()
        if cat == "genre":
            arr = parsed.get("genres", [])
        elif cat == "mood":
            arr = parsed.get("mood", [])
        elif cat == "instruments":
            arr = parsed.get("instruments", [])
        else:
            return []
        out: List[str] = []
        for x in arr if isinstance(arr, list) else []:
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _default_merge(
        category: str,
        a: Optional[Dict[str, Any]],
        b: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Sehr einfache, schemaorientierte Merge-Strategie mit Deduplikation.
        - Für Listen-Kategorien (genres/mood/instruments): stable order, case-insensitive Dedupe.
        - Für 'vocal': fehlende Felder aus b ergänzen (vocal_type/vocal_style).
        """
        if a is None:
            return b
        if b is None:
            return a

        cat = (category or "").lower()
        if cat == "vocal":
            merged = dict(a)
            for k in ("vocal_type", "vocal_style"):
                v = b.get(k)
                if v:
                    merged[k] = v
            return merged

        key_map = {"genre": "genres", "mood": "mood", "instruments": "instruments"}
        k = key_map.get(cat, cat)
        la = a.get(k, []) if isinstance(a.get(k), list) else []
        lb = b.get(k, []) if isinstance(b.get(k), list) else []
        out: List[Any] = []
        seen = set()
        for x in list(la) + list(lb):
            nx = str(x).strip().lower()
            if nx and nx not in seen:
                seen.add(nx)
                out.append(x)
        merged = dict(a)
        merged[k] = out
        return merged
