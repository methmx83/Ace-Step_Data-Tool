#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/core/segment_planner.py — Plant und bereitet Audio-Segmente vor

Funktionen:
- Liest default/override Segment-Strategien pro Kategorie aus prompts_config
- Erzeugt die Union der benötigten Segmente und lässt sie vom AudioProcessor cachen
- Liefert geordnete Pfade je Kategorie zurück (mit Fallback auf die Union)
"""
from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

from scripts.helpers.logger_setup import get_session_logger, log_exception


class SegmentPlanner:
    def __init__(self, prompts_config: Dict[str, Any], audio_processor):
        self.cfg = prompts_config or {}
        self.audio = audio_processor
        self.logger = get_session_logger("SegmentPlanner")

    def segments_for_category(self, category: str) -> List[str]:
        wf = self.cfg.get("workflow_config", {})
        default = wf.get("audio_segments", ["middle"])  # global default
        overrides = wf.get("audio_segments_overrides", {}) or {}
        chosen = overrides.get(category, default)
        valid = {"start", "middle", "end", "best", "full"}
        out = [s for s in chosen if isinstance(s, str) and s.lower() in valid]
        return out or ["middle"]

    def plan_all(self, categories: List[str]) -> Dict[str, List[str]]:
        return {cat: self.segments_for_category(cat) for cat in categories}

    def prepare_cache(self, audio_path: str, categories: List[str]) -> Dict[str, str]:
        """
        Erzeugt die Union der Segmente und gibt Mapping {segment_name -> cache_path} zurück.
        """
        seg_map: Dict[str, str] = {}
        try:
            needed = []
            seen = set()
            for cat in categories:
                for s in self.segments_for_category(cat):
                    s = (s or "").lower().strip()
                    if s and s not in seen:
                        seen.add(s)
                        needed.append(s)
            self.logger.info(f"Preparing audio segments (union): {needed}")
            processed_list = self.audio.process_audio_segments(audio_path, needed)
            for p in processed_list:
                if not p:
                    continue
                seg_name = (p.processing_info or {}).get("segment_strategy") or (p.processing_info or {}).get("segment_used")
                if not seg_name:
                    continue
                pth = p.cache_path or p.source_path
                if pth and Path(pth).exists():
                    seg_map[seg_name] = str(Path(pth).resolve())
        except Exception as e:
            log_exception(self.logger, "audio preprocessing", e)
        return seg_map

    def paths_for_category(self, category: str, seg_path_by_name: Dict[str, str], fallback_union_paths: List[str]) -> List[str]:
        """
        Liefert Liste der Pfade für die Segmente einer Kategorie.
        Fällt auf fallback_union_paths zurück, wenn kein spezifisches Segment vorhanden ist.
        """
        segs = self.segments_for_category(category)
        out = [seg_path_by_name[s] for s in segs if s in seg_path_by_name]
        return out or list(fallback_union_paths)
