"""
scripts/helpers/preset_loader.py
Kleine Hilfsfunktionen um Preset-Dateien (moods.md) zu finden und zu laden.
"""
from pathlib import Path
from typing import List, Optional
import os


def list_presets(presets_root: Optional[str] = None) -> List[str]:
    """Gibt relative Pfade zu allen gefundenen moods.md zurück (z.B. 'hiphop/moods.md')."""
    root = Path(presets_root) if presets_root else Path(__file__).resolve().parents[2] / "presets"
    out: List[str] = []
    if not root.exists():
        return out
    for p in root.rglob("moods.md"):
        try:
            rel = p.relative_to(root)
            # if moods.md is at presets/moods.md -> key is '' or '.'; normalize to 'default'
            key = str(rel.parent) if str(rel.parent) != '.' else 'default'
            # normalize windows backslashes
            key = key.replace('\\', '/').strip('/')
            out.append(key or 'default')
        except Exception:
            out.append(str(p))
    # remove duplicates and sort
    return sorted(list(dict.fromkeys(out)))


def resolve_preset_path(preset_key: str, presets_root: Optional[str] = None) -> Optional[str]:
    """Gibt den absoluten Pfad zur moods.md für einen gegebenen preset-key zurück (z.B. 'hiphop')."""
    root = Path(presets_root) if presets_root else Path(__file__).resolve().parents[2] / "presets"
    if not preset_key:
        return None
    # normalize key (allow either 'hiphop' or 'hiphop/moods.md' or full path)
    pk = str(preset_key).strip()
    # if full path provided and exists
    if os.path.isabs(pk) and Path(pk).is_file():
        return pk
    # if user passed 'hiphop/moods.md'
    if pk.endswith('moods.md'):
        p = root / pk
        if p.is_file():
            return str(p)
        p2 = Path(pk)
        if p2.is_file():
            return str(p2)
    # try as folder under presets
    candidate = root / pk / "moods.md"
    if candidate.is_file():
        return str(candidate)
    # try direct moods.md in presets root
    candidate2 = root / "moods.md"
    if candidate2.is_file():
        return str(candidate2)
    # try fallback: treat pk as relative path from root
    p_rel = root / pk
    if p_rel.is_file():
        return str(p_rel)
    # nothing
    return None
