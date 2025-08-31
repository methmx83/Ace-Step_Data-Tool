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
from scripts.core.prompt_builder import PromptBuilder
from scripts.core.inference_runner import InferenceRunner
from scripts.core.segment_planner import SegmentPlanner
from scripts.tagging.tag_pipeline import TagPipeline
from scripts.helpers.context_extractor import extract_from_filename
from scripts.helpers.tag_processor import create_tag_processor
from scripts.helpers.logger_setup import (
    setup_session_logging, 
    get_session_logger,
    log_exception,
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
        self.logger.info("🟡 Initializing MultiTaggerOrchestrator...")

        # Load configurations
        self.prompts_config = self._load_json_config(prompts_config_path)
        self.logger.info(f"🟢 Loaded prompts config with {len(self.prompts_config.get('prompt_templates', {}))} templates")

        # Log loaded config
        DualLogger.log_config_info(self.prompts_config, "Prompts Config")

        # Initialize components
        try:
            # Audio einmal in 16kHz WAV cachen und für alle Kategorien verwenden
            self.audio_processor = create_audio_processor(enable_compression=False)
            self.logger.info("✅ Audio processor initialized")
        except Exception as e:
            log_exception(self.logger, "audio processor initialization", e)
            raise

        try:
            self.tag_processor = create_tag_processor(moods_file_path, allow_extras=allow_tag_extras)
            self.logger.info("✅ Tag processor initialized")

            # Log tag statistics
            stats = {
                "genres": len(self.tag_processor.allowed_tags.genres),
                "moods": len(self.tag_processor.allowed_tags.moods),
                "instruments": len(self.tag_processor.allowed_tags.instruments),
                "vocal_types": len(self.tag_processor.allowed_tags.vocal_types),
                "keys": len(self.tag_processor.allowed_tags.keys),
                "vocal_fx": len(self.tag_processor.allowed_tags.vocal_fx)
            }
            self.logger.info(f"Tag processor loaded with: {stats}")

        except FileNotFoundError as e:
            # Klare Meldung, wenn presets/moods.md fehlt
            self.logger.error(str(e))
            raise
        except Exception as e:
            log_exception(self.logger, "tag processor initialization", e)
            raise

        # Model wird lazy geladen beim ersten Bedarf
        self.model_wrapper = None
        self.model_config_path = model_config_path

        # Keep track of last used audio cache paths for optional cleanup
        self._last_used_audio_paths = []
        # Keep track of processed source files for session-level fallback cleanup
        self._session_source_files = []

        self.logger.info("✅ MultiTaggerOrchestrator initialized successfully")

    
    
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt JSON-Konfigurationsdatei mit Logging"""
        try:
            self.logger.debug(f"🔵 Loading config from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.debug(f"✅ Config loaded successfully, keys: {list(config.keys())}")
            return config
        except Exception as e:
            log_exception(self.logger, f"loading config from {config_path}", e)
            return {}
    
    def _ensure_model_loaded(self):
        """Lazy Model Loading mit detailliertem Logging"""
        if self.model_wrapper is None:
            self.logger.info("ℹ️ Loading Qwen2-Audio model...😊")
            start_time = time.time()
            
            try:
                self.model_wrapper, _ = load_qwen2audio_model(config_path=self.model_config_path)
                load_time = time.time() - start_time

                self.logger.info(f"👍 Model loaded successfully in {load_time:.1f}s")

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
            file_logger.error(f"⚠️ Audio file not found: {audio_path}")
            return None
        
        file_logger.info(f"🎵 Processing: {filename}")
        if context:
            file_logger.info(f"Context: {context}")
        
        file_start = time.time()
        
        # 0. Audio-Segmente vorbereiten und cachen (Union aller kategoriebasierten Anforderungen)
        used_audio_paths: List[str] = []
        by_cat_needed: Dict[str, List[str]] = {}
        seg_path_by_name: Dict[str, str] = {}
        try:
            categories = self._get_prompt_categories()
            seg_planner = SegmentPlanner(self.prompts_config, self.audio_processor)
            by_cat_needed = seg_planner.plan_all(categories)
            seg_path_by_name = seg_planner.prepare_cache(audio_path, categories)
            used_audio_paths = list(seg_path_by_name.values())
            # Save for optional external cleanup (collect union across session)
            self._last_used_audio_paths.extend(used_audio_paths)
            # dedupe while preserving order
            self._last_used_audio_paths = list(dict.fromkeys(self._last_used_audio_paths))
            if not used_audio_paths:
                # Fallback: single default processing
                single = self.audio_processor.process_audio_file(audio_path)
                if single and (single.cache_path and Path(single.cache_path).exists()):
                    used_audio_paths = [str(Path(single.cache_path).resolve())]
                else:
                    used_audio_paths = [audio_path]
                # ensure last_used contains these paths
                self._last_used_audio_paths.extend(used_audio_paths)
                self._last_used_audio_paths = list(dict.fromkeys(self._last_used_audio_paths))
            file_logger.info(f"Using {len(used_audio_paths)} audio file(s) for inference (union prepared)")
            # record source file for session-level cache lookup
            try:
                self._session_source_files.append(audio_path)
                # dedupe
                self._session_source_files = list(dict.fromkeys(self._session_source_files))
            except Exception:
                pass
        except Exception as e:
            log_exception(file_logger, "🎵 audio preprocessing", e)
            file_logger.warning("Falling back to original audio path")
            by_cat_needed = {}
            used_audio_paths = [audio_path]
            self._last_used_audio_paths.extend(used_audio_paths)
            self._last_used_audio_paths = list(dict.fromkeys(self._last_used_audio_paths))
            try:
                self._session_source_files.append(audio_path)
                self._session_source_files = list(dict.fromkeys(self._session_source_files))
            except Exception:
                pass

        # Helper, um pro Kategorie die richtigen Pfade zu bekommen
        def cat_audio_paths(cat: str) -> List[str]:
            return seg_planner.paths_for_category(cat, seg_path_by_name, used_audio_paths)

        # 1. Alle Kategorien ausführen
        categories = self._get_prompt_categories()
        file_logger.info(f"Processing {len(categories)} categories: {categories}")
        
        responses: Dict[str, Optional[Dict[str, Any]]] = {}
        # Bausteine initialisieren
        prompt_builder = PromptBuilder(self.prompts_config)
        self._ensure_model_loaded()
        runner = InferenceRunner(self.model_wrapper, self.prompts_config, self._count_items_in_response)

        for i, category in enumerate(categories, 1):
            file_logger.info(f"[{i}/{len(categories)}] Processing category: {category}")
            category_start = time.time()

            # Prompt + Template
            full_prompt, template = prompt_builder.build(category, context or {})
            # Kategorie-spezifische Pfade
            cat_paths = cat_audio_paths(category)
            # Min-Requirement
            need = self._min_required_for_category(category)
            # Inferenz ausführen (inkl. Content-Retry intern)
            response = runner.run(category=category, prompt=full_prompt, audio_files=cat_paths, template=template, min_required=need)
            responses[category] = response

            category_time = time.time() - category_start
            file_logger.info(f"Category {category} completed in {category_time:.2f}s")

        # 2. Tags je Kategorie extrahieren und normalisieren
        file_logger.info("🏷️  Extracting tags from responses...")
        tag_pipeline = TagPipeline(self.tag_processor, self.prompts_config)
        by_cat = tag_pipeline.build_by_category(responses)
        norm_by_cat = tag_pipeline.normalize_by_category(by_cat)

        # 3. Auswahl nach Policy treffen
        file_logger.info("🔄 Balancing tags per policy...")
        tagproc_start = time.time()
        final_tags = tag_pipeline.select_final(norm_by_cat)
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
    parser.add_argument("--suppress_header", action="store_true",
                       help="Unterdrücke ausführliche Header-Logs (System-Info, Config Dumps)")
    parser.add_argument("--allow_tag_extras", action="store_true",
                       help="Erlaube Tags außerhalb der Whitelist (Standard: aus)")
    parser.add_argument("--cleanup-cache", action="store_true",
                       help="Löscht temporäre konvertierte Cache-Dateien (data/cache) nach erfolgreicher Verarbeitung jeder Datei")
    
    args = parser.parse_args()
    
    # Logging Setup
    log_file = setup_session_logging(
        session_name=args.session_name,
        verbose=args.verbose,
        suppress_header=bool(args.suppress_header)
    )
    
    main_logger = get_session_logger("Main")
    main_logger.info("=== 🎵 MULTI-TAGGER SESSION START 🎵 ===")
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
    except FileNotFoundError as e:
        # Präzise Fehlermeldung, wenn moods.md fehlt
        main_logger.error(str(e))
        sys.exit(2)
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
            main_logger.error(f"⚠️ File not found: {args.file}")
            sys.exit(1)
    else:
        # Ordner scannen
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            main_logger.error(f"⚠️ Input directory not found: {args.input_dir}")
            sys.exit(1)
        
        audio_files = sorted(list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav")))
    
    if not audio_files:
        main_logger.info("⚠️ No audio files found")
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
            context = extract_from_filename(str(audio_file))

            # BPM-Erkennung: Falls im Dateinamen kein BPM gefunden wurde,
            # versuchen wir, ihn aus der Audiodatei zu ermitteln. Lazy import
            # von detect_tempo, damit schwere Dependencies nicht beim Modulimport
            # geladen werden müssen.
            try:
                if not context.get("bpm") or str(context.get("bpm")).lower() in ("unknown", ""):
                    try:
                        from scripts.helpers.bpm import detect_tempo

                        detected_bpm = detect_tempo(str(audio_file))
                        if detected_bpm:
                            context["bpm"] = str(int(detected_bpm))
                            main_logger.info(f"✅ Detected BPM {context['bpm']} for {audio_file.name}")
                        else:
                            main_logger.debug(f"⚠️ No BPM detected for {audio_file.name}")
                    except Exception as e:
                        # Logge, aber fahre fort — BPM-Erkennung ist optional
                        main_logger.debug(f"⚠️ BPM detection skipped/error for {audio_file.name}: {e}")
                else:
                    main_logger.debug(f"⚠️ BPM already present in filename: {context.get('bpm')} for {audio_file.name}")
            except Exception as e:
                main_logger.debug(f"❌ Unexpected error during BPM detection for {audio_file.name}: {e}")

            # Audio verarbeiten
            tags = orchestrator.process_audio_file(str(audio_file), context)
            
            if tags:
                # Sicherstellen, dass ein erkannter BPM-Tag in den finalen Tags
                try:
                    bpm_val = context.get("bpm") if isinstance(context, dict) else None
                    if bpm_val and str(bpm_val).lower() not in ("unknown", ""):
                        try:
                            # Normalisiere auf ganze Zahl
                            bpm_int = int(float(str(bpm_val)))
                            bpm_tag = f"{bpm_int} bpm"

                            # Entferne alte bpm- Tags (falls vorhanden) und füge das neue ein
                            import re as _re
                            tags = [t for t in tags if not _re.match(r"^bpm-\d+", str(t).strip().lower())]
                            # Platziere das bpm-tag bevorzugt nach dem ersten Tag, sonst ans Ende
                            insert_pos = 1 if len(tags) > 0 else len(tags)
                            tags.insert(insert_pos, bpm_tag)
                            main_logger.info(f"✅ Inserted BPM tag '{bpm_tag}' into tags for {audio_file.name}")
                        except Exception as _e:
                            main_logger.debug(f"⚠️ Failed to normalise/insert BPM tag for {audio_file.name}: {_e}")
                except Exception as e:
                    main_logger.debug(f"❌ Error when handling BPM tag for {audio_file.name}: {e}")

                # Tags-Datei schreiben
                tags_file = orchestrator.write_tags_file(str(audio_file), tags)
                if tags_file:
                    DualLogger.log_file_processing_result(str(audio_file), tags, True)
                    successful += 1
                else:
                    DualLogger.log_file_processing_result(str(audio_file), [], False, "❌ Failed to write tags file")
                    failed += 1
                # per-file cleanup disabled: using session-level cleanup
            else:
                DualLogger.log_file_processing_result(str(audio_file), [], False, "⚠️ No tags generated")
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
    # Optional: Session-level cache cleanup
    try:
        if args.cleanup_cache:
            cache_dir = Path(orchestrator.audio_processor.cache_dir).resolve()
            to_delete = []
            for p in orchestrator._last_used_audio_paths:
                try:
                    rp = Path(p).resolve()
                    # Nur Dateien die im Cache-Ordner liegen
                    rp.relative_to(cache_dir)
                    if rp.exists() and rp.is_file():
                        to_delete.append(str(rp))
                except Exception:
                    continue

            if to_delete:
                main_logger.info(f"ℹ️ Performing session cache cleanup: {len(to_delete)} file(s)")
                res = orchestrator.audio_processor.remove_cached_paths(to_delete)
                main_logger.info(f"🆗 Cache cleanup completed: deleted={len(res.get('deleted',[]))}, skipped={len(res.get('skipped_not_in_cache',[]))}, errors={len(res.get('errors',[]))}")
            else:
                # Fallback: try to find cache entries by source_path in audio_processor.cache_info
                try:
                    session_sources = list(getattr(orchestrator, '_session_source_files', []) or [])
                    fallback_files = []
                    if session_sources:
                        for k, entry in orchestrator.audio_processor.cache_info.get('cache_entries', {}).items():
                            src = entry.get('source_path')
                            try:
                                if src and any(str(s) in str(src) or str(src) in str(s) for s in session_sources):
                                    p = Path(orchestrator.audio_processor.cache_dir) / entry.get('filename')
                                    if p.exists():
                                        fallback_files.append(str(p.resolve()))
                            except Exception:
                                continue

                    if fallback_files:
                        main_logger.info(f"Found {len(fallback_files)} fallback cache file(s) by source_path; deleting them")
                        res = orchestrator.audio_processor.remove_cached_paths(fallback_files)
                        main_logger.info(f"✅ Fallback cache cleanup completed: deleted={len(res.get('deleted',[]))}, skipped={len(res.get('skipped_not_in_cache',[]))}, errors={len(res.get('errors',[]))}")
                    else:
                        main_logger.info("⚠️ No cache files found to delete for this session (even by fallback scan)")
                except Exception as e:
                    main_logger.error(f"⚠️ Fallback cache scan failed: {e}")
    except Exception as e:
        main_logger.error(f"❌ Session cache cleanup failed: {e}")


if __name__ == "__main__":
    main()

