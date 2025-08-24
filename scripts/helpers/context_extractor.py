#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/helpers/context_extractor.py — Kontext aus Dateinamen extrahieren

Beispiel-Patterns:
- "Artist - Title_BPM_Key"
- "Title_BPM_Key"
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any

from scripts.helpers.logger_setup import get_session_logger


def extract_from_filename(audio_path: str) -> Dict[str, Any]:
    """Extrahiert Kontext-Informationen (artist, title, bpm) aus dem Dateinamen."""
    logger = get_session_logger("ContextExtractor")
    filename = Path(audio_path).stem
    context: Dict[str, Any] = {"title": "Unknown", "artist": "Unknown", "bpm": "Unknown"}

    logger.debug(f"Extracting context from filename: {filename}")

    # Pattern: "Artist - Title_BPM_Key" oder "Title_BPM_Key"
    if " - " in filename:
        parts = filename.split(" - ", 1)
        context["artist"] = parts[0].strip()
        filename = parts[1].strip()
        logger.debug(f"Found artist: {context['artist']}")

    # BPM-Extraktion
    bpm_match = re.search(r'_(\d+)(?:_|\.)', filename)
    if bpm_match:
        context["bpm"] = bpm_match.group(1)
        logger.debug(f"Found BPM: {context['bpm']}")

    # Title ist der Rest (ohne BPM und Key)
    title = re.sub(r'_\d+(?:_[A-G][b#]?-(?:maj|min))?$', '', filename)
    if title:
        context["title"] = title.replace("_", " ")
        logger.debug(f"Extracted title: {context['title']}")

    logger.debug(f"Final context: {context}")
    return context
