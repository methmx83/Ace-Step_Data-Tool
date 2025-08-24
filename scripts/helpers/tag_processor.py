"""
scripts/helpers/tag_processor.py
Tag-Nachbearbeitung und Normalisierung basierend auf erlaubten Tag-Listen

Features:
- Lädt erlaubte Tags aus presets/moods.md
- Normalisiert und validiert generierte Tags
- Entfernt Widersprüche (z.B. "instrumental" + "male vocal")
- Fuzzy-Matching für ähnliche Tags
- ACE-STEP konforme Ausgabe
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TagCategories:
    """Container für erlaubte Tags nach Kategorien"""
    genres: Set[str]
    moods: Set[str] 
    instruments: Set[str]
    vocal_types: Set[str]
    
    @classmethod
    def from_moods_file(cls, moods_file_path: str) -> 'TagCategories':
        """Lädt erlaubte Tags aus presets/moods.md"""
        try:
            content = Path(moods_file_path).read_text(encoding='utf-8')
            
            # Parst Markdown-Sections
            genres = cls._extract_section(content, "Genre:")
            moods = cls._extract_section(content, "Mood:")
            instruments = cls._extract_section(content, "Instrument:")
            vocal_types = cls._extract_section(content, "Vocal Typ:")
            
            logger.info(f"Loaded tags: {len(genres)} genres, {len(moods)} moods, "
                       f"{len(instruments)} instruments, {len(vocal_types)} vocal types")
            
            return cls(genres=genres, moods=moods, instruments=instruments, vocal_types=vocal_types)
            
        except Exception as e:
            logger.error(f"Failed to load allowed tags from {moods_file_path}: {e}")
            return cls._get_fallback_categories()
    
    @staticmethod
    def _extract_section(content: str, section_header: str) -> Set[str]:
        """Extrahiert Tags aus einer Markdown-Section"""
        lines = content.split('\n')
        collecting = False
        tags = set()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith(f"## {section_header}"):
                collecting = True
                continue
            elif line.startswith("## ") and collecting:
                # Nächste Section erreicht
                break
            elif collecting and line.startswith("- "):
                # Tag-Eintrag
                tag = line[2:].strip().lower()
                if tag:
                    tags.add(tag)
        
        return tags
    
    @classmethod 
    def _get_fallback_categories(cls) -> 'TagCategories':
        """Fallback-Tags wenn moods.md nicht geladen werden kann"""
        return cls(
            genres={"electronic", "rock", "pop", "ambient", "experimental"},
            moods={"energetic", "calm", "dark", "bright", "emotional"},
            instruments={"guitar", "piano", "drums", "bass", "synthesizer"},
            vocal_types={"male vocal", "female vocal", "instrumental"}
        )

class TagProcessor:
    """
    Hauptklasse für Tag-Nachbearbeitung und -Normalisierung
    """
    
    def __init__(self, moods_file_path: Optional[str] = None, allow_extras: bool = False, fuzzy_cutoff: float = 0.88):
        # Standard-Pfad falls nicht angegeben
        if moods_file_path is None:
            moods_file_path = "presets/moods.md"
        
        self.allowed_tags = TagCategories.from_moods_file(moods_file_path)
        self.conflict_rules = self._setup_conflict_rules()
        self.allow_extras = allow_extras
        self.fuzzy_cutoff = fuzzy_cutoff
        self._alias_map = self._build_alias_map()
        
    def _setup_conflict_rules(self) -> Dict[str, List[str]]:
        """Definiert widersprüchliche Tag-Kombinationen"""
        return {
            "instrumental": ["male vocal", "female vocal", "male rap", "female rap", "spoken word"],
            "male vocal": ["female vocal", "instrumental"],
            "female vocal": ["male vocal", "instrumental"], 
            "male rap": ["female rap", "instrumental"],
            "female rap": ["male rap", "instrumental"]
        }
    
    def normalize_tag(self, raw_tag: str) -> Optional[str]:
        """
        Normalisiert einen einzelnen Tag
        
        1. Lowercasing und Cleanup
        2. Alias-Mapping
        3. Fuzzy-Matching gegen erlaubte Tags
        4. Spezielle Replacements (Tempo → Mood, Key/Time → DROP)
        """
        if not raw_tag or not isinstance(raw_tag, str):
            return None
        
        # Basic cleanup
        normalized = raw_tag.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces → single space
        normalized = normalized.replace("’", "'").replace("‘", "'").replace("`", "'")

        # Frühzeitige technische Filter: Key / Time
        if re.search(r"\bkey\s+[a-g][#b]?\s*(?:major|minor)\b", normalized):
            return None
        if re.search(r"\btime\s+\d+/\d+\b", normalized):
            return None

        # Alias-Mapping
        if normalized in self._alias_map:
            normalized = self._alias_map[normalized]
        
        # Direkte Matches prüfen
        if self._is_allowed_tag(normalized):
            return normalized
        
        # Fuzzy matching versuchen  (difflib against allowed)
        fuzzy_match = self._closest_allowed(normalized)
        if fuzzy_match is not None:
            return fuzzy_match
        
        # Spezielle Transformationen
        transformed = self._apply_transformations(normalized)
        if transformed == "__DROP__":
            return None
        if transformed and self._is_allowed_tag(transformed):
            return transformed
        
        # Fallback: nur behalten, wenn Extras erlaubt sind
        if self.allow_extras:
            logger.debug(f"Tag not in allowed list: '{raw_tag}' → keeping as-is (allow_extras=True)")
            return normalized
        logger.debug(f"Tag not in allowed list: '{raw_tag}' → dropped")
        return None
    
    def _is_allowed_tag(self, tag: str) -> bool:
        """Prüft ob Tag in erlaubten Listen steht"""
        return (tag in self.allowed_tags.genres or 
                tag in self.allowed_tags.moods or
                tag in self.allowed_tags.instruments or
                tag in self.allowed_tags.vocal_types)
    
    def _find_fuzzy_match(self, tag: str) -> Optional[str]:
        """Abwärtskompatible leichte Fuzzy-Varianten (beibehalten, aber Alias-Map bevorzugen)."""
        all_allowed = (self.allowed_tags.genres |
                       self.allowed_tags.moods |
                       self.allowed_tags.instruments |
                       self.allowed_tags.vocal_types)
        for allowed in all_allowed:
            if tag in allowed or allowed in tag:
                if abs(len(tag) - len(allowed)) <= 3:
                    return allowed
        variations = {
            "male voice": "male vocal",
            "female voice": "female vocal",
            "vocals": "singing",
            "vocal": "singing",
            "spoken": "spoken word",
            "spokenword": "spoken word",
            "hip-hop": "hip hop",
            "chill out": "chill-out",
            "drum n bass": "drum and bass",
            "drum'n'bass": "drum and bass",
            "drum’n’bass": "drum and bass",
            "dnb": "drum and bass",
            "deep house": "deep-house",
            "synthpop": "synth pop",
            "electropop": "electro pop",
            "electro-pop": "electro pop",
            "rnb": "r&b",
            "r and b": "r&b",
        }
        mapped = variations.get(tag)
        return mapped if mapped in all_allowed else None

    def _closest_allowed(self, token: str) -> Optional[str]:
        """Sucht das ähnlichste erlaubte Tag via SequenceMatcher und Grenzwert."""
        from difflib import SequenceMatcher
        t = token
        all_allowed = list((self.allowed_tags.genres |
                            self.allowed_tags.moods |
                            self.allowed_tags.instruments |
                            self.allowed_tags.vocal_types))
        best = None
        best_score = 0.0
        for cand in all_allowed:
            r = SequenceMatcher(None, t, cand).ratio()
            if r > best_score:
                best, best_score = cand, r
        return best if best and best_score >= self.fuzzy_cutoff else None
    
    def _apply_transformations(self, tag: str) -> Optional[str]:
        """Wendet spezifische Transformations-Regeln an"""
        
        # Energy-Level Mappings  
        if "energy" in tag:
            if "low" in tag or "calm" in tag:
                return "chilled" if "chilled" in self.allowed_tags.moods else "calm"
            elif "high" in tag or "intense" in tag:
                return "energetic" if "energetic" in self.allowed_tags.moods else "intense"
            elif "mid" in tag or "medium" in tag:
                return "balanced" if "balanced" in self.allowed_tags.moods else None
        
        # Tempo Mappings
        if "tempo" in tag:
            if "slow" in tag:
                return "slow" if "slow" in self.allowed_tags.moods else "mellow"
            elif "fast" in tag:
                return "energetic" if "energetic" in self.allowed_tags.moods else "driving"
            elif "mid" in tag:
                return "groovy" if "groovy" in self.allowed_tags.moods else None
        
        # Key/Time entfernen (nicht als finale Tags gewünscht)
        if re.search(r"\b(?:major|minor)\b", tag) or tag.startswith("key ") or tag.startswith("time "):
            return "__DROP__"
        
        return None
    
    def resolve_conflicts(self, tags: List[str]) -> List[str]:
        """
        Löst widersprüchliche Tag-Kombinationen auf
        
        Prioritäten:
        1. Vocal Type Tags haben Vorrang vor "instrumental"
        2. Spezifischere Tags haben Vorrang vor allgemeineren
        """
        if not tags:
            return tags
        
        logger.debug(f"Resolving conflicts for: {tags}")
        
        # Conflict-Resolution
        resolved_tags = []
        vocal_tags = []
        
        # Sammle Vocal-Tags separat
        for tag in tags:
            # Als Vocal zählen nur: explizite Vocal-Typen oder klare Begriffe ('vocal', 'spoken', 'singing').
            # 'rap' alleine ist ein Genre und darf hier nicht als Vocal zählen.
            if (tag in self.allowed_tags.vocal_types) or any(v in tag for v in ["vocal", "spoken", "singing"]):
                vocal_tags.append(tag)
                logger.debug(f"Found vocal tag: {tag}")
            else:
                resolved_tags.append(tag)
        
        logger.debug(f"Non-vocal tags: {resolved_tags}")
        logger.debug(f"Vocal tags: {vocal_tags}")
        
        # Vocal-Logik: Wenn echte Vocal-Tags da sind, entferne "instrumental"
        if vocal_tags:
            # Nur "instrumental" entfernen wenn es explizit vorhanden ist UND echte Vocals da sind
            if "instrumental" in resolved_tags:
                resolved_tags.remove("instrumental")
                logger.debug("Removed 'instrumental' due to vocal presence")
            
            # Nur das spezifischste Vocal-Tag behalten falls mehrere vorhanden
            if len(vocal_tags) > 1:
                logger.debug(f"Multiple vocal tags, selecting best: {vocal_tags}")
                # Priorität: [specific rap] > [generic vocal] > [singing/spoken]
                priority_order = ["rap", "vocal", "singing", "spoken"]
                best_vocal = None
                for priority in priority_order:
                    for vocal_tag in vocal_tags:
                        if priority in vocal_tag:
                            best_vocal = vocal_tag
                            logger.debug(f"Selected best vocal: {best_vocal}")
                            break
                    if best_vocal:
                        break
                
                # Falls nichts gefunden, nimm das erste
                if best_vocal:
                    vocal_tags = [best_vocal]
                else:
                    vocal_tags = vocal_tags[:1]
                    logger.debug(f"Fallback: kept first vocal tag: {vocal_tags}")
        
        # Alle Tags kombinieren
        final_tags = resolved_tags + vocal_tags
        logger.debug(f"Combined tags before final conflict check: {final_tags}")
        
        # Weitere Conflicts nach Regeln lösen (aber vorsichtiger)
        conflict_removals = []
        for conflict_tag, conflicting_tags in self.conflict_rules.items():
            if conflict_tag in final_tags:
                for conflicting in conflicting_tags:
                    if conflicting in final_tags:
                        logger.debug(f"Conflict detected: '{conflict_tag}' vs '{conflicting}'")
                        # Nur entfernen wenn es wirklich ein Konflikt ist
                        if conflict_tag == "instrumental" and any(v in conflicting for v in ["vocal", "rap", "singing"]):
                            if conflict_tag not in conflict_removals:
                                conflict_removals.append(conflict_tag)
                                logger.debug(f"Marked for removal: {conflict_tag}")
        
        # Entferne markierte Tags
        for tag_to_remove in conflict_removals:
            if tag_to_remove in final_tags:
                final_tags.remove(tag_to_remove)
                logger.debug(f"Removed conflicting tag: {tag_to_remove}")
        
        logger.debug(f"Final resolved tags: {final_tags}")
        return final_tags
    
    def process_tags(self, raw_tags: List[str], max_tags: int = 12) -> List[str]:
        """
        Hauptmethode: Verarbeitet komplette Tag-Liste
        
        1. Normalisierung
        2. Conflict-Resolution  
        3. Deduplication
        4. Length-Limiting
        """
        if not raw_tags:
            return []
        
        # Step 1: Normalisierung
        normalized_tags = []
        for raw_tag in raw_tags:
            normalized = self.normalize_tag(raw_tag)
            if normalized:
                normalized_tags.append(normalized)
        
        # Step 2: Deduplication (preserve order)
        seen = set()
        deduplicated = []
        for tag in normalized_tags:
            if tag not in seen:
                seen.add(tag)
                deduplicated.append(tag)
        
        # Step 3: Conflict resolution
        resolved_tags = self.resolve_conflicts(deduplicated)
        
        # Step 4: Limit length
        final_tags = resolved_tags[:max_tags]
        
        logger.info(f"Tag processing: {len(raw_tags)} → {len(final_tags)} tags")
        return final_tags

    def _build_alias_map(self) -> Dict[str, str]:
        """Erzeugt ein Alias-Mapping auf erlaubte Tags (nur Ziele, die erlaubt sind)."""
        alias: Dict[str, str] = {}
        def add(a: str, b: str):
            alias[a] = b
        # Genre Aliase
        add("hip-hop", "hip hop")
        add("chill out", "chill-out")
        add("drum n bass", "drum and bass")
        add("drum'n'bass", "drum and bass")
        add("drum’n’bass", "drum and bass")
        add("dnb", "drum and bass")
        add("deep house", "deep-house")
        add("synthpop", "synth pop")
        add("electropop", "electro pop")
        add("electro-pop", "electro pop")
        add("rnb", "r&b")
        add("r and b", "r&b")
        add("film scores", "film score")
        # Mood Aliase
        add("feel-good", "feel good")
        add("high energy", "high-energy")
        add("hard hitting", "hard-hitting")
        # Instrument Aliase
        add("string section", "strings")
        add("strings section", "strings")
        add("brass", "horns")
        add("basses", "bass")
        add("kicks", "bass drum")
        add("choirs", "choir")
        # Vocal Aliase
        add("no vocals", "instrumental")
        add("no vocal", "instrumental")
        add("vocals", "singing")
        add("vocal", "singing")
        add("spoken", "spoken word")
        add("spokenword", "spoken word")

        # Nur Aliase behalten, deren Ziel in erlaubten Tags vorhanden ist
        allowed_all = (self.allowed_tags.genres |
                       self.allowed_tags.moods |
                       self.allowed_tags.instruments |
                       self.allowed_tags.vocal_types)
        return {k: v for k, v in alias.items() if v in allowed_all}
    
    def get_tag_statistics(self, tags: List[str]) -> Dict[str, int]:
        """Gibt Statistiken über Tag-Kategorien zurück"""
        stats = {"genre": 0, "mood": 0, "instrument": 0, "vocal": 0, "other": 0}
        
        for tag in tags:
            if tag in self.allowed_tags.genres:
                stats["genre"] += 1
            elif tag in self.allowed_tags.moods:
                stats["mood"] += 1  
            elif tag in self.allowed_tags.instruments:
                stats["instrument"] += 1
            elif tag in self.allowed_tags.vocal_types:
                stats["vocal"] += 1
            else:
                stats["other"] += 1
        
        return stats


# Convenience Functions
def create_tag_processor(moods_file_path: Optional[str] = None, allow_extras: bool = False) -> TagProcessor:
    """Factory-Funktion für TagProcessor"""
    return TagProcessor(moods_file_path, allow_extras=allow_extras)

def quick_process_tags(raw_tags: List[str], moods_file_path: Optional[str] = None, allow_extras: bool = False) -> List[str]:
    """Schnelle Tag-Verarbeitung für einmalige Nutzung"""
    processor = create_tag_processor(moods_file_path, allow_extras=allow_extras)
    return processor.process_tags(raw_tags)


if __name__ == "__main__":
    # Test-Code
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Test mit den generiederten Tags aus den Beispielen
    processor = create_tag_processor()
    
    test_tags = [
        "instrumental", "ambient", "male vocal", "singing", "synthesizer", 
        "guitar", "drums", "piano", "calm", "relaxing", "peaceful", "low energy"
    ]
    
    processed = processor.process_tags(test_tags)
    stats = processor.get_tag_statistics(processed)
    
    print(f"Original tags: {test_tags}")
    print(f"Processed tags: {processed}")
    print(f"Statistics: {stats}")
