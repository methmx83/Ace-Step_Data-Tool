#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/core/inference_runner.py — Inferenz-Runner mit Retry und Content-Retry

Kapselt:
- model_wrapper.chat Aufrufe mit Retry-Logik
- Content-basierte Nachbesserung (min. benötigte Items)
- JSON-Parsing via helpers.json_parser.parse_category_response

Voraussetzung: Qwen2AudioWrapper-kompatibles Interface mit .chat(prompt, audio_files, max_new_tokens, temperature)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import time

from scripts.helpers.json_parser import parse_category_response
from scripts.helpers.logger_setup import get_session_logger, log_exception, log_model_response


class InferenceRunner:
    def __init__(self, model_wrapper, prompts_config: Dict[str, Any], count_fn):
        """
        count_fn: Callable[[str, Optional[Dict[str, Any]]], int]
            Funktion, die die Anzahl sinnvoller Items in einer geparsten Response pro Kategorie zählt.
        """
        self.model = model_wrapper
        self.cfg = prompts_config or {}
        self.count_items = count_fn
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
                    break
            except Exception as e:
                log_exception(cat_logger, f"{category} execution (attempt {attempt + 1})", e)
                if attempt < max_retries:
                    time.sleep(0.2 * (attempt + 1))

        # 2) Content-Retry, falls erforderlich
        policy = self._get_content_retry_policy(category)
        if not policy.get("enabled", False):
            return parsed

        have = self.count_items(category, parsed) if parsed else 0
        if parsed is None or have < int(min_required):
            cr_logger = get_session_logger(f"ContentRetry.{category}")
            reason = "no parse" if parsed is None else f"have {have}/{min_required}"
            cr_logger.info(f"Content retry due to {reason}. Retrying up to {policy['max_attempts']} times…")

            retry_templates = self.cfg.get("retry_templates", {}) or {}
            base_temp = temperature
            attempts = int(policy.get("max_attempts", 0))
            delay = float(policy.get("delay_seconds", 0.25))
            tboost = float(policy.get("temperature_boost", 0.1))

            for i in range(attempts):
                try:
                    rt_text = retry_templates.get(category) or retry_templates.get("default")
                    if not rt_text:
                        rt_text = "IMPORTANT: Return at least {need} distinct {category} item(s) as per the JSON schema. JSON only, no explanations."
                    rt_text = rt_text.replace("{need}", str(min_required)).replace("{category}", str(category))
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
                        have2 = self.count_items(category, parsed2)
                        if have2 >= int(min_required):
                            return parsed2
                        # Falls immer noch zu wenig, behalten wir das Beste bisher
                        if not parsed or have2 > have:
                            parsed = parsed2
                            have = have2
                except Exception as e:
                    log_exception(cr_logger, f"content retry {i+1}", e)
                time.sleep(delay)

        return parsed
