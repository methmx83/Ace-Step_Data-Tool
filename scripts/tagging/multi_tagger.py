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
            self.audio_processor = create_audio_processor()
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
                    if vocal_style and vocal_style not in ("none", "unknown"):
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
        
        # 1. Alle Kategorien ausführen
        categories = self._get_prompt_categories()
        file_logger.info(f"Processing {len(categories)} categories: {categories}")
        
        responses = {}
        
        for i, category in enumerate(categories, 1):
            file_logger.info(f"[{i}/{len(categories)}] Processing category: {category}")
            category_start = time.time()
            
            response = self._execute_category_prompt(category, audio_path, context)
            responses[category] = response
            
            category_time = time.time() - category_start
            file_logger.info(f"Category {category} completed in {category_time:.2f}s")
        
        # 2. Rohe Tags extrahieren
        file_logger.info("🏷️  Extracting tags from responses...")
        raw_tags = self._extract_raw_tags_from_responses(responses)
        
        if not raw_tags:
            file_logger.warning("No tags extracted from any category")
            return None
        
        file_logger.info(f"Extracted {len(raw_tags)} raw tags: {raw_tags}")
        
        # 3. Tag-Nachbearbeitung via Helper-Skript
        file_logger.info("🔄 Processing tags...")
        tagproc_start = time.time()
        
        final_tags = self.tag_processor.process_tags(raw_tags, max_tags=12)
        processing_time = time.time() - tagproc_start
        
        if not final_tags:
            file_logger.warning("No valid tags after processing")
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
