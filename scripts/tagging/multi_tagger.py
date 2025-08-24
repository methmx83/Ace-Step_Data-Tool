#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tagging/multi_tagger.py — Clean Orchestrator mit detailliertem Logging

Verbesserte Version mit:
- Detailliertes Logging in Terminal + Datei
- Debug-Informationen für bessere Problem-Analyse
- Verbesserte Error-Behandlung
- Schrittweise Logging des gesamten Prozesses
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Local modules
from scripts.core.model_loader import load_qwen2audio_model, Qwen2AudioWrapper
from scripts.core.audio_processor import create_audio_processor
from scripts.helpers.json_parser import parse_category_response
from scripts.helpers.tag_processor import create_tag_processor
from scripts.helpers.logger_setup import (
    setup_session_logging, 
    get_session_logger,
    log_exception,
    log_model_response,
    DualLogger
)


class MultiTaggerOrchestrator:
    """
    Hauptklasse: Orchestriert das gesamte Audio-Tagging mit detailliertem Logging
    """
    
    def __init__(self, 
                 model_config_path: str = "config/model_config.json",
                 prompts_config_path: str = "config/prompts.json",
                 moods_file_path: str = "presets/moods.md",
                 allow_tag_extras: bool = False):
        
        self.logger = get_session_logger("Orchestrator")
        self.logger.info("Initializing MultiTaggerOrchestrator...")
        
        # Load configurations
        self.prompts_config = self._load_json_config(prompts_config_path)
        self.logger.info(f"Loaded prompts config with {len(self.prompts_config.get('prompt_templates', {}))} templates")
        
        # Log loaded config
        DualLogger.log_config_info(self.prompts_config, "Prompts Config")
        
        # Initialize components
        try:
            # Audio einmal in 16kHz WAV cachen und für alle Kategorien verwenden
            self.audio_processor = create_audio_processor(enable_compression=False)
            self.logger.info("Audio processor initialized")
        except Exception as e:
            log_exception(self.logger, "audio processor initialization", e)
            raise
        
        try:
            self.tag_processor = create_tag_processor(moods_file_path, allow_extras=allow_tag_extras)
            self.logger.info("Tag processor initialized")
            
            # Log tag statistics
            stats = {
                "genres": len(self.tag_processor.allowed_tags.genres),
                "moods": len(self.tag_processor.allowed_tags.moods),
                "instruments": len(self.tag_processor.allowed_tags.instruments),
                "vocal_types": len(self.tag_processor.allowed_tags.vocal_types)
            }
            self.logger.info(f"Tag processor loaded with: {stats}")
            
        except Exception as e:
            log_exception(self.logger, "tag processor initialization", e)
            raise
        
        # Model wird lazy geladen beim ersten Bedarf
        self.model_wrapper: Optional[Qwen2AudioWrapper] = None
        self.model_config_path = model_config_path
        
        self.logger.info("MultiTaggerOrchestrator initialized successfully")

    def _get_output_policy(self) -> Dict[str, Any]:
        """Liest Output-Format-Policy (Min/Max je Kategorie, Gesamtlimit) aus prompts-config."""
        of = self.prompts_config.get("output_format", {})
        wf = self.prompts_config.get("workflow_config", {})
        return {
            "min_per_cat": of.get("min_tags_per_category", {}),
            "max_per_cat": of.get("max_tags_per_category", {}),
            "max_total": of.get("max_total_tags", None),
            "order": wf.get("default_categories", ["genre", "mood", "instruments", "vocal"]),
        }

    def _get_audio_segments(self) -> List[str]:
        """Liest die zu verwendenden Audiosegmente aus der Konfiguration."""
        wf = self.prompts_config.get("workflow_config", {})
        segments = wf.get("audio_segments", ["middle"])  # default kompatibel
        # validieren
        valid = {"start", "middle", "end", "best", "full"}
        out = [s for s in segments if isinstance(s, str) and s.lower() in valid]
        return out or ["middle"]
    
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt JSON-Konfigurationsdatei mit Logging"""
        try:
            self.logger.debug(f"Loading config from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.debug(f"Config loaded successfully, keys: {list(config.keys())}")
            return config
        except Exception as e:
            log_exception(self.logger, f"loading config from {config_path}", e)
            return {}
    
    def _ensure_model_loaded(self):
        """Lazy Model Loading mit detailliertem Logging"""
        if self.model_wrapper is None:
            self.logger.info("Loading Qwen2-Audio model...")
            start_time = time.time()
            
            try:
                self.model_wrapper, _ = load_qwen2audio_model(config_path=self.model_config_path)
                load_time = time.time() - start_time
                
                self.logger.info(f"Model loaded successfully in {load_time:.1f}s")
                
                # Model-Info loggen
                if hasattr(self.model_wrapper, 'get_model_info'):
                    model_info = self.model_wrapper.get_model_info()
                    DualLogger.log_config_info(model_info, "Model Info")
                
            except Exception as e:
                log_exception(self.logger, "model loading", e)
                raise
    
    def _get_prompt_categories(self) -> List[str]:
        """Gibt die zu verarbeitenden Prompt-Kategorien zurück"""
        workflow_config = self.prompts_config.get("workflow_config", {})
        categories = workflow_config.get("default_categories", ["genre", "mood", "instruments", "vocal"])
        self.logger.debug(f"Using categories: {categories}")
        return categories

    def _get_content_retry_policy(self) -> Dict[str, Any]:
        """Liest die Content-Retry-Policy aus der Konfiguration."""
        wf = self.prompts_config.get("workflow_config", {})
        policy = wf.get("content_retry", {})
        return {
            "enabled": bool(policy.get("enabled", False)),
            "max_attempts": int(policy.get("max_attempts", 0)),
            "delay_seconds": float(policy.get("delay_seconds", 0.25)),
            "temperature_boost": float(policy.get("temperature_boost", 0.1)),
            "overrides": policy.get("overrides", {}),
        }

    def _min_required_for_category(self, category: str) -> int:
        of = self.prompts_config.get("output_format", {})
        return int(of.get("min_tags_per_category", {}).get(category, 0))

    def _count_items_in_response(self, category: str, parsed: Optional[Dict[str, Any]]) -> int:
        if not parsed:
            return 0
        # Hilfsfunktion: zählt nur Tags, die nach Normalisierung gültig bleiben
        def count_normalized(tags: List[str], *, only_vocal_allowed: bool = False) -> int:
            c = 0
            for t in tags:
                nt = self.tag_processor.normalize_tag(str(t).lower())
                if not nt:
                    continue
                if only_vocal_allowed:
                    # Zähle nur, wenn es wirklich ein gültiger Vocal-Typ ist
                    if nt not in self.tag_processor.allowed_tags.vocal_types:
                        continue
                c += 1
            return c

        def coerce_list(val: Any) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val if isinstance(v, (str, int, float))]
            if isinstance(val, str):
                # Split an häufigen Trennern
                import re as _re
                parts = [_s.strip() for _s in _re.split(r"[,;\n\|/]+", val) if _s.strip()]
                return parts
            return []

        if category == "genre":
            return count_normalized(coerce_list(parsed.get("genres", []) or []))
        if category == "mood":
            return count_normalized(coerce_list(parsed.get("mood", []) or []))
        if category == "instruments":
            return count_normalized(coerce_list(parsed.get("instruments", []) or []))
        if category == "vocal":
            # Leite finalen Vocal-Tag ab und prüfe Normalisierung
            vt = (parsed.get("vocal_type") or "").strip().lower()
            vs = (parsed.get("vocal_style") or "").strip().lower()
            derived: List[str] = []
            combined = None
            if vt in ("male", "female"):
                if "rap" in vs:
                    combined = f"{vt} rap"
                elif ("feature" in vs) or ("feat" in vs):
                    combined = f"{vt} feature vocal"
                elif any(k in vs for k in ("sing", "vocal")):
                    combined = f"{vt} vocal"
                elif "spoken" in vs:
                    combined = "spoken word"
                else:
                    combined = f"{vt} vocal"
            elif vt == "instrumental":
                combined = "instrumental"
            if combined:
                derived.append(combined)
            else:
                if vt and vt not in ("none", "unknown"):
                    derived.append(vt)
                # 'rap' alleine nicht als Vocal-Tag verwenden
                if vs and vs not in ("none", "unknown", "rap"):
                    derived.append(vs)
            return count_normalized(derived, only_vocal_allowed=True)
        return 0
    
    def _execute_category_prompt(self, 
                                category: str, 
                                audio_path: str, 
                                context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Führt Prompt für eine spezifische Kategorie aus mit detailliertem Logging
        """
        category_logger = get_session_logger(f"Category.{category}")
        category_logger.info(f"Starting {category} analysis for {Path(audio_path).name}")
        
        self._ensure_model_loaded()
        
        # Template für Kategorie laden
        templates = self.prompts_config.get("prompt_templates", {})
        if category not in templates:
            category_logger.warning(f"No template found for category: {category}")
            return None
        
        template = templates[category]
        category_logger.debug(f"Template loaded: max_tokens={template.get('max_tokens', 'N/A')}, temp={template.get('temperature', 'N/A')}")
        
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        
        # Nur den user_prompt mit Kontext formatieren (system_prompt enthält JSON-Klammern)
        if context and user_prompt:
            try:
                user_prompt = user_prompt.format(**context)
                category_logger.debug(f"User prompt formatted with context: {context}")
            except KeyError as e:
                category_logger.warning(f"Missing context variable for {category}: {e}")
        
        # Kombiniere System + User Prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        category_logger.debug(f"Full prompt length: {len(full_prompt)} chars")
        
        # Führe Model-Chat aus mit Retry-Logic
        max_retries = template.get("retry_count", 2)
        max_tokens = template.get("max_tokens", 60)
        temperature = template.get("temperature", 0.0)
        
        category_logger.debug(f"Executing with: max_tokens={max_tokens}, temp={temperature}, retries={max_retries}")
        
        for attempt in range(max_retries + 1):
            try:
                category_logger.debug(f"Attempt {attempt + 1}/{max_retries + 1}")
                
                response_start = time.time()
                raw_response = self.model_wrapper.chat(
                    prompt=full_prompt,
                    audio_files=[audio_path],
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                response_time = time.time() - response_start
                
                category_logger.debug(f"Model response time: {response_time:.2f}s")
                category_logger.debug(f"Raw response length: {len(raw_response)} chars")
                category_logger.debug(f"Raw response preview: {raw_response[:200]}...")
                
                # Parse JSON response mit Helper-Skript
                parse_start = time.time()
                parsed_response = parse_category_response(raw_response, category)
                parse_time = time.time() - parse_start
                
                # Detailliertes Logging der Model-Response
                log_model_response(category_logger, category, full_prompt, raw_response, parsed_response)
                
                if parsed_response:
                    category_logger.info(f"✓ {category} successful: {parsed_response}")
                    category_logger.debug(f"JSON parsing time: {parse_time:.3f}s")
                    return parsed_response
                else:
                    category_logger.warning(f"JSON parsing failed (attempt {attempt + 1})")
                    category_logger.warning(f"Raw response was: '{raw_response}'")
                    
            except Exception as e:
                log_exception(category_logger, f"{category} execution (attempt {attempt + 1})", e)
                if attempt < max_retries:
                    delay = 0.2 * (attempt + 1)
                    category_logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        category_logger.error(f"✗ {category} failed after {max_retries + 1} attempts")
        return None
    
    def _extract_raw_tags_from_responses(self, responses: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Extrahiert rohe Tags aus allen Category-Responses mit Logging
        """
        extraction_logger = get_session_logger("TagExtraction")
        extraction_logger.info("Extracting raw tags from category responses...")
        
        raw_tags = []
        
        for category, response in responses.items():
            category_tags = []
            
            if not response:
                extraction_logger.warning(f"No response for category: {category}")
                continue
                
            extraction_logger.debug(f"Processing {category} response: {response}")
                
            # Category-spezifische Tag-Extraktion
            if category == "genre":
                genres = response.get("genres", [])
                category_tags = [str(g).lower() for g in genres if g]
                extraction_logger.debug(f"Genre tags: {category_tags}")
                
            elif category == "mood":
                moods = response.get("mood", [])
                category_tags = [str(m).lower() for m in moods if m]
                extraction_logger.debug(f"Mood tags: {category_tags}")
                    
            elif category == "instruments":
                instruments = response.get("instruments", [])
                category_tags = [str(i).lower() for i in instruments if i]
                extraction_logger.debug(f"Instrument tags: {category_tags}")
                
            # 'technical' Kategorie wird nicht mehr verarbeitet (Energy/Key/Time nicht benötigt)
                    
            elif category == "vocal":
                vocal_type = (response.get("vocal_type") or "").strip().lower()
                vocal_style = (response.get("vocal_style") or "").strip().lower()

                combined = None
                if vocal_type in ("male", "female"):
                    if "rap" in vocal_style:
                        combined = f"{vocal_type} rap"
                    elif ("feature" in vocal_style) or ("feat" in vocal_style):
                        combined = f"{vocal_type} feature vocal"
                    elif any(k in vocal_style for k in ("sing", "vocal")):
                        combined = f"{vocal_type} vocal"
                    elif "spoken" in vocal_style:
                        combined = "spoken word"
                elif vocal_type == "instrumental":
                    combined = "instrumental"

                if combined:
                    category_tags.append(combined)
                else:
                    # Fallback auf Einzeltags
                    if vocal_type and vocal_type not in ("none", "unknown"):
                        category_tags.append(f"{vocal_type} vocal" if vocal_type in ("male", "female") else vocal_type)
                    # 'rap' alleine nicht als Vocal-Tag verwenden
                    if vocal_style and vocal_style not in ("none", "unknown", "rap"):
                        category_tags.append(vocal_style)
                extraction_logger.debug(f"Vocal tags: {category_tags}")
                    
            elif category == "production":
                production_styles = response.get("production_style", [])
                sound_quality = response.get("sound_quality")
                
                category_tags.extend([str(s).lower() for s in production_styles if s])
                if sound_quality:
                    category_tags.append(sound_quality)
                extraction_logger.debug(f"Production tags: {category_tags}")
            
            raw_tags.extend(category_tags)
            extraction_logger.info(f"{category}: extracted {len(category_tags)} tags")
        
        extraction_logger.info(f"Total raw tags extracted: {len(raw_tags)}")
        extraction_logger.debug(f"All raw tags: {raw_tags}")
        return raw_tags

    def _build_tags_by_category(self, responses: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extrahiert Tags je Kategorie (lowercased, noch nicht normalisiert)."""
        by_cat: Dict[str, List[str]] = {"genre": [], "mood": [], "instruments": [], "vocal": [], "production": []}

        def coerce_list(val: Any) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val if isinstance(v, (str, int, float))]
            if isinstance(val, str):
                import re as _re
                parts = [_s.strip() for _s in _re.split(r"[,;\n\|/]+", val) if _s.strip()]
                return parts
            return []

        for category, response in responses.items():
            if not response:
                continue
            if category == "genre":
                genres = coerce_list(response.get("genres", []))
                by_cat["genre"] = [str(g).lower() for g in genres if g]
            elif category == "mood":
                moods = coerce_list(response.get("mood", []))
                by_cat["mood"] = [str(m).lower() for m in moods if m]
            elif category == "instruments":
                instruments = coerce_list(response.get("instruments", []))
                by_cat["instruments"] = [str(i).lower() for i in instruments if i]
            elif category == "vocal":
                vocal_type = (response.get("vocal_type") or "").strip().lower()
                vocal_style = (response.get("vocal_style") or "").strip().lower()
                combined = None
                if vocal_type in ("male", "female"):
                    if "rap" in vocal_style:
                        combined = f"{vocal_type} rap"
                    elif ("feature" in vocal_style) or ("feat" in vocal_style):
                        combined = f"{vocal_type} feature vocal"
                    elif any(k in vocal_style for k in ("sing", "vocal")):
                        combined = f"{vocal_type} vocal"
                    elif "spoken" in vocal_style:
                        combined = "spoken word"
                elif vocal_type == "instrumental":
                    combined = "instrumental"

                cat_tags: List[str] = []
                if combined:
                    cat_tags.append(combined)
                else:
                    if vocal_type and vocal_type not in ("none", "unknown"):
                        cat_tags.append(f"{vocal_type} vocal" if vocal_type in ("male", "female") else vocal_type)
                    # 'rap' alleine nicht als Vocal-Tag verwenden
                    if vocal_style and vocal_style not in ("none", "unknown", "rap"):
                        cat_tags.append(vocal_style)
                by_cat["vocal"] = [str(t).lower() for t in cat_tags if t]
            elif category == "production":
                ps = response.get("production_style", [])
                sq = response.get("sound_quality")
                cat_tags = [str(s).lower() for s in ps if s]
                if sq:
                    cat_tags.append(str(sq).lower())
                by_cat["production"] = cat_tags

        return by_cat

    def _normalize_and_dedupe_by_category(self, by_cat: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalisiert alle Tags je Kategorie via TagProcessor und entfernt Duplikate (stabile Reihenfolge)."""
        norm: Dict[str, List[str]] = {}
        for cat, tags in by_cat.items():
            seen = set()
            out: List[str] = []
            for t in tags:
                nt = self.tag_processor.normalize_tag(t)
                if not nt:
                    continue
                # Kategoriefilter: Nur Tags behalten, die zur aktuellen Kategorie gehören
                if cat == "genre" and nt not in self.tag_processor.allowed_tags.genres:
                    continue
                if cat == "mood" and nt not in self.tag_processor.allowed_tags.moods:
                    continue
                if cat == "instruments" and nt not in self.tag_processor.allowed_tags.instruments:
                    continue
                if cat == "vocal" and nt not in self.tag_processor.allowed_tags.vocal_types:
                    continue
                if nt in seen:
                    continue
                seen.add(nt)
                out.append(nt)
            norm[cat] = out
        return norm

    def _select_final_tags(self, tags_by_category: Dict[str, List[str]]) -> List[str]:
        """Wendet Min/Max je Kategorie sowie Gesamtlimit an (in der Reihenfolge der Workflow-Kategorien)."""
        policy = self._get_output_policy()
        order: List[str] = policy["order"]
        min_per = policy["min_per_cat"]
        max_per = policy["max_per_cat"]
        max_total = policy["max_total"]

        selected: List[tuple[str, str]] = []  # (tag, category)
        counts: Dict[str, int] = {c: 0 for c in order}

        def add_tag(tag: str, cat: str) -> bool:
            if any(tag == t for t, _c in selected):
                return False
            if max_total is not None and len(selected) >= max_total:
                return False
            cap = max_per.get(cat, 999)
            if counts.get(cat, 0) >= cap:
                return False
            selected.append((tag, cat))
            counts[cat] = counts.get(cat, 0) + 1
            return True

        # 1) Mindestanzahl je Kategorie erfüllen
        for cat in order:
            need = min_per.get(cat, 0)
            if need <= 0:
                continue
            for tag in tags_by_category.get(cat, []):
                if counts.get(cat, 0) >= need:
                    break
                add_tag(tag, cat)

        # 2) Bis Max je Kategorie auffüllen (in Reihenfolge)
        for cat in order:
            cap = max_per.get(cat, 999)
            for tag in tags_by_category.get(cat, []):
                if counts.get(cat, 0) >= cap:
                    break
                add_tag(tag, cat)

        # 3) Sortierung stabil halten: erst nach order, dann nach ursprünglicher Reihenfolge beibehalten
        order_index = {c: i for i, c in enumerate(order)}
        selected.sort(key=lambda x: order_index.get(x[1], 999))

        final = [t for t, _c in selected]
        # 4) Letzte Conflict-Resolution anwenden (kann selten Tags entfernen)
        final = self.tag_processor.resolve_conflicts(final)
        # 5) Spezielle Genresortierung: 'hip hop' und 'rap' zusammenhalten, 'hip hop' vor 'rap'
        try:
            if "hip hop" in final and "rap" in final:
                # Entferne beide und füge in gewünschter Reihenfolge wieder ein, nahe der ersten Position von beiden
                idxs = [i for i, t in enumerate(final) if t in ("hip hop", "rap")]
                anchor = min(idxs) if idxs else len(final)
                final = [t for t in final if t not in ("hip hop", "rap")]
                # Insert at anchor maintaining order hip hop -> rap
                if anchor <= len(final):
                    final[anchor:anchor] = ["hip hop", "rap"]
                else:
                    final.extend(["hip hop", "rap"])
        except Exception:
            pass
        return final
    
    def process_audio_file(self, 
                          audio_path: str, 
                          context: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
        """
        Hauptmethode: Verarbeitet eine Audio-Datei komplett mit detailliertem Logging
        """
        file_logger = get_session_logger("FileProcessor")
        filename = Path(audio_path).name
        
        audio_path = str(Path(audio_path).resolve())
        
        if not Path(audio_path).exists():
            file_logger.error(f"Audio file not found: {audio_path}")
            return None
        
        file_logger.info(f"🎵 Processing: {filename}")
        if context:
            file_logger.info(f"Context: {context}")
        
        file_start = time.time()
        
        # 0. Audio-Segmente vorbereiten und cachen
        used_audio_paths: List[str] = []
        try:
            segments = self._get_audio_segments()
            file_logger.info(f"Preparing audio segments: {segments}")
            processed_list = self.audio_processor.process_audio_segments(audio_path, segments)
            for p in processed_list:
                pth = p.cache_path or p.source_path
                if pth and Path(pth).exists():
                    used_audio_paths.append(str(Path(pth).resolve()))
            if not used_audio_paths:
                # Fallback: single default processing
                single = self.audio_processor.process_audio_file(audio_path)
                if single and (single.cache_path and Path(single.cache_path).exists()):
                    used_audio_paths = [str(Path(single.cache_path).resolve())]
                else:
                    used_audio_paths = [audio_path]
            file_logger.info(f"Using {len(used_audio_paths)} audio file(s) for inference")
        except Exception as e:
            log_exception(file_logger, "audio preprocessing", e)
            file_logger.warning("Falling back to original audio path")
            used_audio_paths = [audio_path]

        # 1. Alle Kategorien ausführen
        categories = self._get_prompt_categories()
        file_logger.info(f"Processing {len(categories)} categories: {categories}")
        
        responses: Dict[str, Optional[Dict[str, Any]]] = {}
        
        for i, category in enumerate(categories, 1):
            file_logger.info(f"[{i}/{len(categories)}] Processing category: {category}")
            category_start = time.time()

            # Mehrsegment-Unterstützung: direkter Model-Aufruf mit allen Segmenten
            self._ensure_model_loaded()

            templates = self.prompts_config.get("prompt_templates", {})
            template = templates.get(category, {})
            system_prompt = template.get("system_prompt", "")
            user_prompt = template.get("user_prompt", "")
            if context and user_prompt:
                try:
                    user_prompt = user_prompt.format(**context)
                except KeyError:
                    pass
            full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()

            max_retries = template.get("retry_count", 2)
            max_tokens = template.get("max_tokens", 60)
            temperature = template.get("temperature", 0.0)

            response = None
            content_retry = self._get_content_retry_policy()
            overrides = (content_retry.get("overrides") or {}).get(category, {})
            cr_enabled = bool(content_retry.get("enabled", False))
            cr_attempts = int(overrides.get("max_attempts", content_retry.get("max_attempts", 0)))
            cr_delay = float(overrides.get("delay_seconds", content_retry.get("delay_seconds", 0.25)))
            cr_temp_boost = float(overrides.get("temperature_boost", content_retry.get("temperature_boost", 0.1)))
            base_temp = float(temperature)

            for attempt in range(max_retries + 1):
                try:
                    response_start = time.time()
                    raw_response = self.model_wrapper.chat(
                        prompt=full_prompt,
                        audio_files=used_audio_paths,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                    response_time = time.time() - response_start
                    file_logger.debug(f"Model response time: {response_time:.2f}s")

                    parsed = parse_category_response(raw_response, category)
                    log_model_response(get_session_logger(f"Category.{category}"), category, full_prompt, raw_response, parsed)
                    if parsed:
                        response = parsed
                        break
                except Exception as e:
                    log_exception(get_session_logger(f"Category.{category}"), f"{category} execution (attempt {attempt + 1})", e)
                    if attempt < max_retries:
                        time.sleep(0.2 * (attempt + 1))

            # Content-basierter Retry: auch wenn Parsing komplett scheitert (response=None)
            need = self._min_required_for_category(category)
            have = self._count_items_in_response(category, response) if response else 0
            if cr_enabled and (response is None or have < need):
                cr_logger = get_session_logger(f"ContentRetry.{category}")
                reason = "no parse" if response is None else f"have {have}/{need}"
                cr_logger.info(f"Content retry due to {reason}. Retrying up to {cr_attempts} times…")
                for i in range(int(cr_attempts)):
                    try:
                        if category == "mood":
                            reinforced_prompt = (
                                full_prompt
                                + "\n\nIMPORTANT: Return between 2 and 3 distinct mood items as a JSON array under key \"mood\"."
                                + " Use concise, single- or two-word English moods. JSON only, no explanations."
                            )
                        elif category == "genre":
                            reinforced_prompt = (
                                full_prompt
                                + "\n\nIMPORTANT: Respond ONLY with JSON matching exactly {\"genres\": [\"genre1\", \"genre2\"]}."
                                + " Provide 2 specific genres. English only. No explanations."
                            )
                        elif category == "instruments":
                            reinforced_prompt = (
                                full_prompt
                                + "\n\nIMPORTANT: Respond ONLY with JSON matching exactly {\"instruments\": [\"inst1\", \"inst2\", \"inst3\"]}."
                                + " Provide 2-3 real instruments (no genres/vocals). English only. No explanations."
                            )
                        elif category == "vocal":
                            reinforced_prompt = (
                                full_prompt
                                + "\n\nIMPORTANT: Respond ONLY with JSON like {\"vocal_type\": \"male vocal|female vocal|spoken word|male feature vocal|female feature vocal|instrumental\"}."
                                + " Choose exactly one. English only. No explanations."
                            )
                        else:
                            reinforced_prompt = (
                                full_prompt
                                + f"\n\nIMPORTANT: Return at least {need} distinct {category} item(s) as per the JSON schema. JSON only, no explanations."
                            )

                        temp2 = base_temp + float(cr_temp_boost) * (i + 1)
                        raw_response = self.model_wrapper.chat(
                            prompt=reinforced_prompt,
                            audio_files=used_audio_paths,
                            max_new_tokens=max_tokens,
                            temperature=temp2
                        )
                        parsed2 = parse_category_response(raw_response, category)
                        log_model_response(cr_logger, category, reinforced_prompt, raw_response, parsed2)
                        if parsed2:
                            have2 = self._count_items_in_response(category, parsed2)
                            if have2 >= need:
                                response = parsed2
                                break
                    except Exception as e:
                        log_exception(cr_logger, f"content retry {i+1}", e)
                    time.sleep(float(cr_delay))

            responses[category] = response

            category_time = time.time() - category_start
            file_logger.info(f"Category {category} completed in {category_time:.2f}s")

        # 2. Tags je Kategorie extrahieren und normalisieren
        file_logger.info("🏷️  Extracting tags from responses...")
        by_cat = self._build_tags_by_category(responses)
        file_logger.debug(f"Tags by category (raw): {by_cat}")

        norm_by_cat = self._normalize_and_dedupe_by_category(by_cat)
        file_logger.debug(f"Tags by category (normalized): {norm_by_cat}")

        # 3. Auswahl nach Policy treffen
        file_logger.info("🔄 Balancing tags per policy...")
        tagproc_start = time.time()
        final_tags = self._select_final_tags(norm_by_cat)
        processing_time = time.time() - tagproc_start

        if not final_tags:
            file_logger.warning("No valid tags after selection")
            return None
        
        # Tag-Statistiken
        tag_stats = self.tag_processor.get_tag_statistics(final_tags)
        total_time = time.time() - file_start

        file_logger.info(f"✅ Processing completed in {total_time:.2f}s")
        file_logger.info(f"Final tags ({len(final_tags)}): {final_tags}")
        file_logger.info(f"Tag distribution: {tag_stats}")
        
        return final_tags
    
    def write_tags_file(self, audio_path: str, tags: List[str]) -> Optional[Path]:
        """
        Schreibt Tags in _prompt.txt Datei mit Logging
        """
        write_logger = get_session_logger("FileWriter")
        
        try:
            audio_file = Path(audio_path)
            tags_file = audio_file.with_name(f"{audio_file.stem}_prompt.txt")
            
            tags_content = ", ".join(tags)
            tags_file.write_text(tags_content, encoding='utf-8')
            
            write_logger.info(f"💾 Tags written to: {tags_file.name}")
            write_logger.debug(f"Tags content: {tags_content}")
            return tags_file
            
        except Exception as e:
            log_exception(write_logger, "writing tags file", e)
            return None


def create_context_from_filename(audio_path: str) -> Dict[str, Any]:
    """
    Extrahiert Kontext-Informationen aus Dateinamen mit Logging
    """
    context_logger = get_session_logger("ContextExtractor")
    filename = Path(audio_path).stem
    context = {"title": "Unknown", "artist": "Unknown", "bpm": "Unknown"}
    
    context_logger.debug(f"Extracting context from filename: {filename}")
    
    # Pattern: "Artist - Title_BPM_Key" oder "Title_BPM_Key"
    if " - " in filename:
        parts = filename.split(" - ", 1)
        context["artist"] = parts[0].strip()
        filename = parts[1].strip()
        context_logger.debug(f"Found artist: {context['artist']}")
    
    # BPM-Extraktion
    bpm_match = re.search(r'_(\d+)(?:_|\.)', filename)
    if bpm_match:
        context["bpm"] = bpm_match.group(1)
        context_logger.debug(f"Found BPM: {context['bpm']}")
    
    # Title ist der Rest (ohne BPM und Key)
    title = re.sub(r'_\d+(?:_[A-G][b#]?-(?:maj|min))?$', '', filename)
    if title:
        context["title"] = title.replace("_", " ")
        context_logger.debug(f"Extracted title: {context['title']}")
    
    context_logger.debug(f"Final context: {context}")
    return context


def main():
    """Hauptfunktion mit verbessertem Logging"""
    parser = argparse.ArgumentParser(
        description="Clean Multi-Category Audio Tagger für ACE-STEP mit Logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.tagging.multi_tagger --input_dir data/audio
  python -m scripts.tagging.multi_tagger --file "song.mp3" --verbose
        """
    )
    
    parser.add_argument("--input_dir", type=str, default="data/audio",
                       help="Ordner mit Audio-Dateien (.mp3/.wav)")
    parser.add_argument("--file", type=str,
                       help="Einzelne Audio-Datei verarbeiten")
    parser.add_argument("--prompts", type=str, default="config/prompts.json",
                       help="Prompt-Templates Konfiguration")
    parser.add_argument("--model_config", type=str, default="config/model_config.json", 
                       help="Model-Konfiguration")
    parser.add_argument("--moods_file", type=str, default="presets/moods.md",
                       help="Erlaubte Tags Datei")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose Logging (DEBUG level)")
    parser.add_argument("--session_name", type=str,
                       help="Name für diese Session (für Log-Datei)")
    parser.add_argument("--allow_tag_extras", action="store_true",
                       help="Erlaube Tags außerhalb der Whitelist (Standard: aus)")
    
    args = parser.parse_args()
    
    # Logging Setup
    log_file = setup_session_logging(
        session_name=args.session_name,
        verbose=args.verbose
    )
    
    main_logger = get_session_logger("Main")
    main_logger.info("=== MULTI-TAGGER SESSION START ===")
    main_logger.info(f"Log file: {log_file}")
    main_logger.info(f"Arguments: {vars(args)}")
    
    # Orchestrator initialisieren
    try:
        orchestrator = MultiTaggerOrchestrator(
            model_config_path=args.model_config,
            prompts_config_path=args.prompts,
            moods_file_path=args.moods_file,
            allow_tag_extras=args.allow_tag_extras
        )
    except Exception as e:
        log_exception(main_logger, "orchestrator initialization", e)
        main_logger.error("Initialization failed, exiting")
        sys.exit(1)
    
    # Audio-Dateien sammeln
    audio_files = []
    
    if args.file:
        # Einzelne Datei
        file_path = Path(args.file)
        if file_path.exists():
            audio_files.append(file_path)
        else:
            main_logger.error(f"File not found: {args.file}")
            sys.exit(1)
    else:
        # Ordner scannen
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            main_logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        
        audio_files = sorted(list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav")))
    
    if not audio_files:
        main_logger.info("No audio files found")
        sys.exit(0)
    
    # Session Start logging
    DualLogger.log_processing_session_start([str(f) for f in audio_files])
    
    # Verarbeitung
    session_start = time.time()
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        main_logger.info(f"\n{'='*60}")
        main_logger.info(f"FILE {i}/{len(audio_files)}: {audio_file.name}")
        main_logger.info(f"{'='*60}")
        
        try:
            # Kontext aus Filename extrahieren
            context = create_context_from_filename(str(audio_file))
            
            # Audio verarbeiten
            tags = orchestrator.process_audio_file(str(audio_file), context)
            
            if tags:
                # Tags-Datei schreiben
                tags_file = orchestrator.write_tags_file(str(audio_file), tags)
                if tags_file:
                    DualLogger.log_file_processing_result(str(audio_file), tags, True)
                    successful += 1
                else:
                    DualLogger.log_file_processing_result(str(audio_file), [], False, "Failed to write tags file")
                    failed += 1
            else:
                DualLogger.log_file_processing_result(str(audio_file), [], False, "No tags generated")
                failed += 1
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            log_exception(main_logger, f"processing {audio_file.name}", e)
            DualLogger.log_file_processing_result(str(audio_file), [], False, error_msg)
            failed += 1
    
    # Session Summary
    total_time = time.time() - session_start
    DualLogger.log_session_summary(successful, failed, total_time)
    
    main_logger.info(f"\n📊 FINAL SUMMARY:")
    main_logger.info(f"✅ Successful: {successful}")
    main_logger.info(f"❌ Failed: {failed}")
    main_logger.info(f"📁 Log file: {log_file}")
    main_logger.info("=== SESSION END ===")


if __name__ == "__main__":
    main()
