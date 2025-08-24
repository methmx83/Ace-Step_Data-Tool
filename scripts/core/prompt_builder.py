#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/core/prompt_builder.py — Zentraler Prompt-Builder

Aufgabe:
- Baut für eine Kategorie den finalen Prompt aus prompts.json
- Unterstützt system_prompt_lines und system_prompt
- Formatiert user_prompt sicher mit Context (ohne JSON-Klammern zu stören)

Hinweis: Logging nutzt get_session_logger aus helpers.logger_setup
"""
from typing import Any, Dict, Tuple, Optional

from scripts.helpers.logger_setup import get_session_logger


class PromptBuilder:
    """Erzeugt vollständige Prompts je Kategorie aus der Konfiguration."""

    def __init__(self, prompts_config: Dict[str, Any]):
        self.cfg = prompts_config or {}
        self.logger = get_session_logger("PromptBuilder")

    def build(self, category: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Baut den vollständigen Prompt (system + user) und gibt zusätzlich das Template zurück.

        Rückgabe:
            (full_prompt, template_dict)
        """
        templates = (self.cfg.get("prompt_templates") or {})
        template: Dict[str, Any] = templates.get(category, {})
        if not template:
            self.logger.warning(f"No prompt template found for category '{category}'")
            return ("", {})

        # System prompt: unterstützt system_prompt_lines und system_prompt
        sp_lines = template.get("system_prompt_lines")
        if isinstance(sp_lines, list) and sp_lines:
            system_prompt = "\n".join(str(line) for line in sp_lines)
        else:
            system_prompt = template.get("system_prompt", "")

        # User prompt: mit Context formatieren (nur Platzhalter ersetzen)
        user_prompt = template.get("user_prompt", "")
        if context and user_prompt:
            try:
                user_prompt = user_prompt.format(**context)
            except KeyError as e:
                # Fehlende Platzhalter sind kein harter Fehler; wir loggen nur.
                self.logger.debug(f"Missing context key for category '{category}': {e}")

        full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        self.logger.debug(f"Built prompt for '{category}' (len={len(full_prompt)} chars)")
        return (full_prompt, template)
