#!/usr/bin/env python3
"""
scripts/dev_repl.py
Startet eine interaktive REPL mit dem Projekt-Root bereits in sys.path und bietet vorbereitete Helfer-Imports.

Usage:
    python scripts/dev_repl.py

Danach im Prompt z.B.:
    from scripts.helpers.preset_loader import list_presets, resolve_preset_path
    print(list_presets())
"""
import sys
from pathlib import Path
import code

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print(f"Added project root to sys.path: {root}")

# Prepare a convenient local namespace
ns = {
    "root": root,
}

try:
    # Pre-import commonly used helpers
    from scripts.helpers.preset_loader import list_presets, resolve_preset_path
    ns["list_presets"] = list_presets
    ns["resolve_preset_path"] = resolve_preset_path
    print("Available helpers: list_presets(), resolve_preset_path(key)")
except Exception:
    print("Could not pre-import helpers; you can still import them manually.")

banner = "Dev REPL — project root added to sys.path. Use list_presets()/resolve_preset_path(). Ctrl-D to exit."
code.interact(banner=banner, local=ns)
