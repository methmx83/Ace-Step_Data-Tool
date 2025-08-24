"""
scripts/helpers/items_merge.py

Hilfsfunktionen für inkrementelles Tagging:
- default_items_fn: extrahiert Listen-Items nach Kategorie
- default_merge_fn: mergen mit Dedupe, stabile Reihenfolge

Anpassbar je nach JSON-Schema (siehe json_parser.extract_json Normalisierung)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def default_items_fn(category: str, parsed: Optional[Dict[str, Any]]) -> List[str]:
    if not parsed:
        return []
    cat = (category or "").lower()
    if cat == "genre":
        arr = parsed.get("genres", [])
    elif cat == "mood":
        arr = parsed.get("mood", [])
    elif cat == "instruments":
        arr = parsed.get("instruments", [])
    else:
        return []
    out: List[str] = []
    for x in arr if isinstance(arr, list) else []:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def default_merge_fn(category: str, a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if a is None:
        return b
    if b is None:
        return a

    cat = (category or "").lower()
    if cat == "vocal":
        merged = dict(a)
        for k in ("vocal_type", "vocal_style"):
            v = b.get(k)
            if v:
                merged[k] = v
        return merged

    key_map = {"genre": "genres", "mood": "mood", "instruments": "instruments"}
    k = key_map.get(cat, cat)
    la = a.get(k, []) if isinstance(a.get(k), list) else []
    lb = b.get(k, []) if isinstance(b.get(k), list) else []

    out: List[Any] = []
    seen = set()
    for x in list(la) + list(lb):
        nx = str(x).strip().lower()
        if nx and nx not in seen:
            seen.add(nx)
            out.append(x)
    merged = dict(a)
    merged[k] = out
    return merged
