#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tagging/tag_pipeline.py — Extraktion, Normalisierung und Auswahl von Tags

"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.helpers.logger_setup import get_session_logger


class TagPipeline:
    def __init__(self, tag_processor, prompts_config: Dict[str, Any]):
        self.tp = tag_processor
        self.cfg = prompts_config or {}
        self.logger = get_session_logger("TagPipeline")

    # ------------------ Roh-Extraktion ------------------
    def _coerce_list(self, val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(v) for v in val if isinstance(v, (str, int, float))]
        if isinstance(val, str):
            import re as _re
            parts = [_s.strip() for _s in _re.split(r"[,;\n\|/]+", val) if _s.strip()]
            return parts
        return []

    def build_by_category(self, responses: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, List[str]]:
        by_cat: Dict[str, List[str]] = {"genre": [], "mood": [], "instruments": [], "vocal": [], "key": [], "vocal_fx": [], "rap_style": [], "production": []}
        for category, response in responses.items():
            if not response:
                continue
            if category == "genre":
                genres = self._coerce_list(response.get("genres", []))
                by_cat["genre"] = [str(g).lower() for g in genres if g]
            elif category == "mood":
                moods = self._coerce_list(response.get("mood", []))
                by_cat["mood"] = [str(m).lower() for m in moods if m]
            elif category == "instruments":
                instruments = self._coerce_list(response.get("instruments", []))
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
                    if vocal_style and vocal_style not in ("none", "unknown", "rap"):
                        cat_tags.append(vocal_style)
                by_cat["vocal"] = [str(t).lower() for t in cat_tags if t]
            elif category == "key":
                # key usually returns a single string: 'major' or 'minor'
                k = response.get("key") or response.get("keys")
                if isinstance(k, list):
                    ks = [str(x).lower().strip() for x in k if x]
                else:
                    ks = [str(k).lower().strip()] if k else []
                by_cat["key"] = [x for x in ks if x]
            elif category == "vocal_fx":
                vfx = self._coerce_list(response.get("vocal_fx", []))
                by_cat["vocal_fx"] = [str(v).lower() for v in vfx if v]
            elif category == "rap_style" or category == "rap-style" or category == "rapstyle":
                rs = self._coerce_list(response.get("rap_style", []) or response.get("rapstyle", []) )
                by_cat["rap_style"] = [str(r).lower() for r in rs if r]
            elif category == "production":
                ps = self._coerce_list(response.get("production_style", []))
                sq = response.get("sound_quality")
                cat_tags = [str(s).lower() for s in ps if s]
                if sq:
                    cat_tags.append(str(sq).lower())
                by_cat["production"] = cat_tags
        self.logger.debug(f"Tags by category (raw): {by_cat}")
        return by_cat

    # ------------------ Normalisierung ------------------
    def normalize_by_category(self, by_cat: Dict[str, List[str]]) -> Dict[str, List[str]]:
        norm: Dict[str, List[str]] = {}
        for cat, tags in by_cat.items():
            seen = set()
            out: List[str] = []
            for t in tags:
                # Special-case: key should not be passed through general normalize_tag because
                # TagProcessor historically drops 'major'/'minor' in transformations. Validate directly.
                if cat == "key":
                    # Use TagProcessor normalization so plain 'major'/'minor' become 'major key'/'minor key'
                    nt = self.tp.normalize_tag(t)
                else:
                    nt = self.tp.normalize_tag(t)
                if not nt:
                    continue
                if cat == "genre" and nt not in self.tp.allowed_tags.genres:
                    continue
                if cat == "mood" and nt not in self.tp.allowed_tags.moods:
                    continue
                if cat == "instruments" and nt not in self.tp.allowed_tags.instruments:
                    continue
                if cat == "vocal" and nt not in self.tp.allowed_tags.vocal_types:
                    continue
                if cat == "key" and not self.tp._is_allowed_tag(nt):
                    continue
                if cat == "vocal_fx" and nt not in self.tp.allowed_tags.vocal_fx:
                    continue
                if cat == "rap_style" and nt not in self.tp.allowed_tags.rap_style:
                    continue
                if nt in seen:
                    continue
                seen.add(nt)
                out.append(nt)
            norm[cat] = out
        self.logger.debug(f"Tags by category (normalized): {norm}")
        return norm

    # ------------------ Auswahl/Policy ------------------
    def _policy(self) -> Dict[str, Any]:
        of = self.cfg.get("output_format", {})
        wf = self.cfg.get("workflow_config", {})
        return {
            "min_per_cat": of.get("min_tags_per_category", {}),
            "max_per_cat": of.get("max_tags_per_category", {}),
            "max_total": of.get("max_total_tags", None),
            "order": wf.get("default_categories", ["genre", "mood", "instruments", "vocal"]),
        }

    def select_final(self, tags_by_category: Dict[str, List[str]]) -> List[str]:
        policy = self._policy()
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

        # Sortieren: nach order stabil
        order_index = {c: i for i, c in enumerate(order)}
        selected.sort(key=lambda x: order_index.get(x[1], 999))

        final = [t for t, _c in selected]
        final = self.tp.resolve_conflicts(final)
        try:
            if "hip hop" in final and "rap" in final:
                idxs = [i for i, t in enumerate(final) if t in ("hip hop", "rap")]
                anchor = min(idxs) if idxs else len(final)
                final = [t for t in final if t not in ("hip hop", "rap")]
                if anchor <= len(final):
                    final[anchor:anchor] = ["hip hop", "rap"]
                else:
                    final.extend(["hip hop", "rap"])
        except Exception:
            pass
        return final
