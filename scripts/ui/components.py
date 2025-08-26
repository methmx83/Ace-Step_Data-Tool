"""
Minimal UI-Komponenten für die Gradio-Demo.
"""
from typing import List
from scripts.helpers.preset_loader import list_presets, resolve_preset_path


def get_preset_options() -> List[str]:
    opts = list_presets()
    if not opts:
        return ["default"]
    return opts


def get_preset_path(key: str) -> str:
    p = resolve_preset_path(key)
    return p or "presets/moods.md"
