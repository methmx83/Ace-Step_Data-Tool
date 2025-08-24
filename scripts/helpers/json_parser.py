from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONParser:
    """
    Robuster JSON-Parser für LLM-Outputs mit mehreren Fallback-Strategien.
    - Extrahiert JSON aus Text (direkt, in Codeblöcken, eingebettet)
    - Nutzt dynamisch erlaubte Tags aus presets/moods.md (Genres, Moods, Instruments, Vocal Types)
    - Bietet kategoriespezifische Fallbacks
    """

    # Cache der erlaubten Tags (lazy geladen)
    _ALLOWED: Optional[Dict[str, set]] = None

    # --------------- Public API ---------------
    @staticmethod
    def extract_json(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Hauptmethode: Extrahiert JSON aus Text mit mehreren Strategien."""
        if not text or not text.strip():
            return None
        text = text.strip()
        cat = (category or "").strip().lower() or None

        # 1) Direktes JSON-Objekt
        if text.startswith("{") and text.endswith("}"):
            obj = JSONParser._try_parse_json(text)
            if obj:
                return JSONParser._normalize_result_keys(obj, cat)

        # 1b) Direktes JSON-Array → wrap je nach Kategorie
        if text.startswith("[") and text.endswith("]"):
            wrapped = JSONParser._try_parse_list_and_wrap(text, cat)
            if wrapped:
                return JSONParser._normalize_result_keys(wrapped, cat)

        # 2) In Codeblöcken
        obj = JSONParser._extract_from_code_blocks(text, cat)
        if obj:
            return JSONParser._normalize_result_keys(obj, cat)

        # 3) Irgendwo im Text
        obj = JSONParser._find_json_in_text(text, cat)
        if obj:
            return JSONParser._normalize_result_keys(obj, cat)

        # 3b) Quoted JSON (escaped)
        obj = JSONParser._find_quoted_json_in_text(text, cat)
        if obj:
            return JSONParser._normalize_result_keys(obj, cat)

        # 4) Kategorie-Fallbacks über erlaubte Tags
        if cat:
            obj = JSONParser._category_specific_fallback(text, cat)
            if obj:
                return JSONParser._normalize_result_keys(obj, cat)

        # 5) Generisch aus Mustern konstruieren
        obj = JSONParser._construct_from_patterns(text, cat)
        if obj:
            return JSONParser._normalize_result_keys(obj, cat)

        logger.warning("Failed to extract JSON from text: %s...", text[:120])
        return None

    # --------------- Internals ---------------
    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            cleaned = JSONParser._clean_json_text(text)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                return None
        if isinstance(obj, str):
            try:
                obj2 = json.loads(obj)
                obj = obj2
            except Exception:
                pass
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _clean_json_text(text: str) -> str:
        # Entferne trailing commas
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        # Quote unquoted keys (heuristisch)
        text = re.sub(r"(?m)([,{\s])([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*", r"\1""\2"": ", text)
        # Single quotes → double quotes
        text = re.sub(r"'", '"', text)
        return text

    @staticmethod
    def _extract_from_code_blocks(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        patterns = [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if not m:
                continue
            inner = m.group(1).strip()
            if inner.startswith("{") and inner.endswith("}"):
                obj = JSONParser._try_parse_json(inner)
                if obj:
                    return obj
            if inner.startswith("[") and inner.endswith("]"):
                wrapped = JSONParser._try_parse_list_and_wrap(inner, category)
                if wrapped:
                    return wrapped
        return None

    @staticmethod
    def _find_json_in_text(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        patterns = [r"\{[^{}]*\}", r"\[[^\[\]]*\]"]
        for pat in patterns:
            for m in re.finditer(pat, text):
                candidate = m.group(0)
                if candidate.startswith("{"):
                    obj = JSONParser._try_parse_json(candidate)
                    if obj:
                        return obj
                elif candidate.startswith("["):
                    wrapped = JSONParser._try_parse_list_and_wrap(candidate, category)
                    if wrapped:
                        return wrapped
        return None

    @classmethod
    def _get_allowed(cls) -> Dict[str, set]:
        if cls._ALLOWED is not None:
            return cls._ALLOWED

        allowed: Dict[str, set] = {
            "genres": set(),
            "moods": set(),
            "instruments": set(),
            "vocal_types": set(),
        }

        # Kandidatenpfade für presets/moods.md
        candidates: List[Path] = []
        try:
            here = Path(__file__).resolve()
            root = here.parents[2]  # .../data_v2
            candidates.append(root / "presets" / "moods.md")
        except Exception:
            pass
        env_candidate = os.environ.get("ACE_MOODS_MD")
        if env_candidate:
            candidates.append(Path(env_candidate))

        def parse_markdown(md_text: str) -> Dict[str, List[str]]:
            sections: Dict[str, List[str]] = {}
            current: Optional[str] = None
            for raw in md_text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                m = re.match(r"^##\s*([A-Za-z\s/_-]+)\s*:?$", line)
                if m:
                    current = m.group(1).strip().lower()
                    sections.setdefault(current, [])
                    continue
                if current and line.startswith("-"):
                    val = line.lstrip("-").strip().lower()
                    if val:
                        sections[current].append(val)
            return sections

        md_sections: Dict[str, List[str]] = {}
        file_found = False
        for p in candidates:
            try:
                if p.is_file():
                    file_found = True
                    md_text = p.read_text(encoding="utf-8")
                    md_sections = parse_markdown(md_text)
                    break
            except Exception as e:
                logger.debug("Failed reading moods.md at %s: %s", p, e)

        # Wenn keine moods.md gefunden wurde, hart abbrechen
        if not file_found:
            logger.error("moods.md not found!")
            raise FileNotFoundError("moods.md not found!")

        def get_section(keys: List[str]) -> List[str]:
            for k in keys:
                if k in md_sections:
                    return md_sections[k]
            return []

        genres = get_section(["genre", "genres"]) or []
        moods = get_section(["mood", "moods"]) or []
        instruments = get_section(["instrument", "instruments"]) or []
        # Ergänze deutsche Schreibweise "Vocal Typ"
        vocals = get_section(["vocal", "vocals", "vocal type", "vocal types", "vocal typ"]) or []

        allowed["genres"].update(genres)
        allowed["moods"].update(moods)
        allowed["instruments"].update(instruments)
        allowed["vocal_types"].update(vocals)

        # Normalisierung: hip hop, Leerzeichen/Bindestrich
        normed_genres = set()
        for g in allowed["genres"]:
            gg = g.replace("hip-hop", "hip hop")
            normed_genres.add(gg)
        allowed["genres"] = normed_genres

        cls._ALLOWED = allowed
        return cls._ALLOWED

    @staticmethod
    def _token_pattern(token: str) -> str:
        esc = re.escape(token.strip().lower())
        esc = esc.replace(r"\ ", r"[\s\-]+")
        return rf"(?<!\w){esc}(?!\w)"

    @classmethod
    def _find_allowed_in_text(cls, text: str, candidates: set, limit: int) -> List[str]:
        if not text or not candidates:
            return []
        t = text.lower()
        hits: List[tuple[int, str]] = []
        for cand in candidates:
            try:
                pat = cls._token_pattern(cand)
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    hits.append((m.start(), cand))
            except re.error:
                if cand in t:
                    hits.append((t.index(cand), cand))
        hits.sort(key=lambda x: x[0])
        out: List[str] = []
        seen = set()
        for _, val in hits:
            if val not in seen:
                out.append(val)
                seen.add(val)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _category_specific_fallback(text: str, category: str) -> Optional[Dict[str, Any]]:
        category = category.lower()
        if category == "genre":
            return JSONParser._parse_genre_fallback(text)
        if category == "mood":
            return JSONParser._parse_mood_fallback(text)
        if category == "instruments":
            return JSONParser._parse_instruments_fallback(text)
        if category == "vocal":
            return JSONParser._parse_vocal_fallback(text)
        return None

    @staticmethod
    def _parse_genre_fallback(text: str) -> Optional[Dict[str, Any]]:
        allowed = JSONParser._get_allowed().get("genres", set())
        found = JSONParser._find_allowed_in_text(text, allowed, limit=2)
        if found:
            norm = [f.replace("hip-hop", "hip hop") for f in found]
            return {"genres": norm}
        return None

    @staticmethod
    def _parse_mood_fallback(text: str) -> Optional[Dict[str, Any]]:
        allowed = JSONParser._get_allowed().get("moods", set())
        found = JSONParser._find_allowed_in_text(text, allowed, limit=3)
        if found:
            return {"mood": found}
        return None

    @staticmethod
    def _parse_instruments_fallback(text: str) -> Optional[Dict[str, Any]]:
        allowed = JSONParser._get_allowed().get("instruments", set())
        found = JSONParser._find_allowed_in_text(text, allowed, limit=4)
        if found:
            return {"instruments": found}
        return None

    @staticmethod
    def _parse_vocal_fallback(text: str) -> Optional[Dict[str, Any]]:
        t = text.lower()
        allowed = JSONParser._get_allowed().get("vocal_types", set())
        direct = JSONParser._find_allowed_in_text(t, allowed, limit=1)
        if direct:
            d = direct[0]
            if d == "instrumental":
                return {"vocal_type": "instrumental", "vocal_style": "none"}
            if d == "spoken word":
                return {"vocal_type": "mixed", "vocal_style": "spoken"}
            if d == "male rap":
                return {"vocal_type": "male", "vocal_style": "rap"}
            if d == "female rap":
                return {"vocal_type": "female", "vocal_style": "rap"}
            if d == "male feature vocal":
                return {"vocal_type": "male", "vocal_style": "feature"}
            if d == "female feature vocal":
                return {"vocal_type": "female", "vocal_style": "feature"}
            if d == "male vocal":
                return {"vocal_type": "male", "vocal_style": "singing"}
            if d == "female vocal":
                return {"vocal_type": "female", "vocal_style": "singing"}

        # Heuristik
        vocal_type = "instrumental"
        if re.search(r"\b(?:male|männlich)\b", t, re.IGNORECASE):
            vocal_type = "male"
        elif re.search(r"\b(?:female|weiblich)\b", t, re.IGNORECASE):
            vocal_type = "female"
        elif re.search(r"\bmixed\b", t, re.IGNORECASE):
            vocal_type = "mixed"

        vocal_style = "none"
        if re.search(r"\b(?:rap|rapping)\b", t, re.IGNORECASE):
            vocal_style = "rap"
        elif re.search(r"\b(?:sing|singing|song|vocal)\b", t, re.IGNORECASE):
            vocal_style = "singing"
        elif re.search(r"\b(?:spoken|speech)\b", t, re.IGNORECASE):
            vocal_style = "spoken"

        return {"vocal_type": vocal_type, "vocal_style": vocal_style}

    @staticmethod
    def _try_parse_list_and_wrap(text: str, category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            arr = json.loads(text)
            if not isinstance(arr, list):
                return None
        except Exception:
            return None
        cat = (category or "").lower() if category else None
        key_map = {"genre": "genres", "mood": "mood", "instruments": "instruments", "vocal": "vocal"}
        if cat in key_map:
            return {key_map[cat]: arr}
        return {"data": arr}

    @staticmethod
    def _find_quoted_json_in_text(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        quoted = re.findall(r'"(\{.*?\}|\[.*?\])"', text)
        for q in quoted:
            try:
                unescaped = q.encode("utf-8").decode("unicode_escape")
                if unescaped.startswith("{") and unescaped.endswith("}"):
                    obj = JSONParser._try_parse_json(unescaped)
                    if obj:
                        return obj
                elif unescaped.startswith("[") and unescaped.endswith("]"):
                    wrapped = JSONParser._try_parse_list_and_wrap(unescaped, category)
                    if wrapped:
                        return wrapped
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_result_keys(obj: Optional[Dict[str, Any]], category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not obj:
            return obj
        mapping = {
            "genre": "genres",
            "genres": "genres",
            "mood": "mood",
            "moods": "mood",
            "instrument": "instruments",
            "instruments": "instruments",
            "vocaltype": "vocal_type",
            "vocal_type": "vocal_type",
            "vocalstyle": "vocal_style",
            "vocal_style": "vocal_style",
        }
        normalized: Dict[str, Any] = {}
        for k, v in obj.items():
            nk = mapping.get(k.lower().replace(" ", ""), k)
            normalized[nk] = v

        for list_key in ("genres", "mood", "instruments"):
            if list_key in normalized and not isinstance(normalized[list_key], list):
                normalized[list_key] = [normalized[list_key]]

        if "vocal_type" in normalized and "vocal_style" not in normalized:
            normalized["vocal_style"] = "singing" if normalized["vocal_type"] in ("male", "female", "mixed") else "none"
        if "vocal_style" in normalized and "vocal_type" not in normalized:
            normalized["vocal_type"] = "mixed"

        if category:
            if category == "genre":
                return {"genres": normalized.get("genres")} if "genres" in normalized else None
            if category == "mood":
                return {"mood": normalized.get("mood")} if "mood" in normalized else None
            if category == "instruments":
                return {"instruments": normalized.get("instruments")} if "instruments" in normalized else None
            if category == "vocal":
                keys = {k: normalized[k] for k in ("vocal_type", "vocal_style") if k in normalized}
                return keys or None
        return normalized

    @staticmethod
    def _construct_from_patterns(text: str, category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text or not category:
            return None
        allowed = JSONParser._get_allowed()
        cat = category.lower()
        if cat == "genre":
            found = JSONParser._find_allowed_in_text(text, allowed.get("genres", set()), limit=2)
            return {"genres": found} if found else None
        if cat == "mood":
            found = JSONParser._find_allowed_in_text(text, allowed.get("moods", set()), limit=3)
            return {"mood": found} if found else None
        if cat == "instruments":
            found = JSONParser._find_allowed_in_text(text, allowed.get("instruments", set()), limit=4)
            return {"instruments": found} if found else None
        if cat == "vocal":
            voc = JSONParser._parse_vocal_fallback(text)
            return voc
        return None


# Convenience

def parse_category_response(text: str, category: str) -> Optional[Dict[str, Any]]:
    return JSONParser.extract_json(text, category)


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    return JSONParser.extract_json(text)


