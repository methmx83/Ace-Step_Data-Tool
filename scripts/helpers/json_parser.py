"""
scripts/helpers/json_parser.py
Robuste JSON-Parsing-Utilities für LLM-Outputs

Features:
- Multi-Strategy JSON Extraktion aus LLM-Responses
- Fallback-Parsing für verschiedene Formate
- Category-specific Parsing-Logik
- Error-Recovery und Logging
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class JSONParser:
    """
    Robuster JSON-Parser für LLM-Outputs mit mehreren Fallback-Strategien
    """
    
    @staticmethod
    def extract_json(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Hauptmethode: Extrahiert JSON aus Text mit verschiedenen Strategien
        
        Args:
            text: Roher LLM-Output Text
            category: Prompt-Kategorie für spezifische Fallbacks
            
        Returns:
            Parsed JSON Dict oder None
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Strategy 1: Direct JSON (starts and ends with braces)
        if text.startswith('{') and text.endswith('}'):
            result = JSONParser._try_parse_json(text)
            if result:
                return result
        
        # Strategy 2: JSON in fenced code blocks
        result = JSONParser._extract_from_code_blocks(text)
        if result:
            return result
        
        # Strategy 3: Find JSON anywhere in text
        result = JSONParser._find_json_in_text(text)
        if result:
            return result
        
        # Strategy 4: Category-specific regex fallbacks
        if category:
            result = JSONParser._category_specific_fallback(text, category)
            if result:
                return result
        
        # Strategy 5: Attempt to construct JSON from text patterns
        result = JSONParser._construct_from_patterns(text, category)
        if result:
            return result
        
        logger.warning(f"Failed to extract JSON from text: {text[:100]}...")
        return None
    
    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Versucht JSON zu parsen, inklusive Double-Pass falls der Inhalt ein JSON-String ist."""
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        # Falls ein JSON-String enthalten ist, erneut parsen
        if isinstance(obj, str):
            try:
                inner = json.loads(obj)
                if isinstance(inner, dict):
                    return inner
                return None
            except Exception:
                return None
        return obj if isinstance(obj, dict) else None
    
    @staticmethod
    def _extract_from_code_blocks(text: str) -> Optional[Dict[str, Any]]:
        """Extrahiert JSON aus ```json Code-Blöcken"""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',  
            r'`(\{.*?\})`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                result = JSONParser._try_parse_json(match)
                if result:
                    return result
        
        return None
    
    @staticmethod
    def _find_json_in_text(text: str) -> Optional[Dict[str, Any]]:
        """Findet JSON-ähnliche Strukturen im Text"""
        # Suche nach {key: value} Patterns
        json_patterns = [
            r'\{[^{}]*\}',  # Simple flat JSON
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested JSON (one level)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Clean up common issues
                cleaned = JSONParser._clean_json_text(match)
                result = JSONParser._try_parse_json(cleaned)
                if result:
                    return result
        
        return None
    
    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Bereinigt häufige JSON-Formatierungsfehler"""
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix unquoted keys (common LLM mistake)
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # Fix unquoted string values  
        text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9\s-]*[a-zA-Z0-9])\s*([,}])', r': "\1"\2', text)
        
        return text
    
    @staticmethod
    def _category_specific_fallback(text: str, category: str) -> Optional[Dict[str, Any]]:
        """Category-spezifische Regex-Fallbacks wenn JSON fehlschlägt"""
        
        category = category.lower()
        
        if category == "genre":
            return JSONParser._parse_genre_fallback(text)
        elif category == "mood":
            return JSONParser._parse_mood_fallback(text)
        elif category == "instruments":
            return JSONParser._parse_instruments_fallback(text)
    # 'technical' Kategorie wurde entfernt
        elif category == "vocal":
            return JSONParser._parse_vocal_fallback(text)
        
        return None
    
    @staticmethod
    def _parse_genre_fallback(text: str) -> Optional[Dict[str, Any]]:
        """Fallback-Parsing für Genre-Kategorie"""
        # Suche nach Genre-Wörtern
        genre_keywords = [
            "rock", "pop", "jazz", "classical", "electronic", "hip hop", "rap", 
            "blues", "folk", "country", "metal", "techno", "house", "trance", 
            "ambient", "indie", "punk", "reggae", "soul", "funk", "disco", "dance"
        ]
        
        found_genres = []
        for keyword in genre_keywords:
            # Unterstütze auch Schreibweise mit Bindestrich für hip-hop
            pattern = keyword if keyword != "hip hop" else r"hip[-\s]?hop"
            if re.search(rf'\b{pattern}\b', text, re.IGNORECASE):
                found_genres.append("hip hop" if keyword == "hip hop" else keyword)
        
        if found_genres:
            return {"genres": found_genres[:2]}  # Max 2 genres
        
        return None
    
    @staticmethod
    def _parse_mood_fallback(text: str) -> Optional[Dict[str, Any]]:
        """Fallback-Parsing für Mood-Kategorie"""
        mood_keywords = [
            "happy", "sad", "energetic", "calm", "aggressive", "peaceful", 
            "melancholic", "uplifting", "dramatic", "romantic", "mysterious", 
            "playful", "intense", "relaxing", "dark", "bright"
        ]
        
        energy_keywords = {
            "low": ["low", "calm", "quiet", "soft"],
            "medium": ["medium", "moderate", "balanced"],
            "high": ["high", "intense", "energetic", "powerful"]
        }
        
        found_moods = []
        for keyword in mood_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                found_moods.append(keyword)
        
        if found_moods:
            return {"mood": found_moods[:3]}
        
        return None
    
    @staticmethod
    def _parse_instruments_fallback(text: str) -> Optional[Dict[str, Any]]:
        """Fallback-Parsing für Instruments-Kategorie"""
        instrument_keywords = [
            "guitar", "piano", "drums", "bass", "violin", "saxophone", "trumpet", 
            "synthesizer", "keyboard", "vocal", "voice", "strings", "brass", 
            "percussion", "flute", "clarinet", "organ"
        ]
        
        found_instruments = []
        for keyword in instrument_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                found_instruments.append(keyword)
        
        if found_instruments:
            return {"instruments": found_instruments[:4]}  # Max 4 instruments
        
        return None
    
    @staticmethod
    # Technische Fallbacks entfallen
    
    @staticmethod
    def _parse_vocal_fallback(text: str) -> Optional[Dict[str, Any]]:
        """Fallback-Parsing für Vocal-Kategorie"""
        # Vocal type detection
        vocal_type = "instrumental"  # default
        if re.search(r'\b(?:male|männlich)\b', text, re.IGNORECASE):
            vocal_type = "male"
        elif re.search(r'\b(?:female|weiblich)\b', text, re.IGNORECASE):
            vocal_type = "female"
        elif re.search(r'\bmixed\b', text, re.IGNORECASE):
            vocal_type = "mixed"
        
        # Vocal style detection
        vocal_style = "none"
        if vocal_type != "instrumental":
            if re.search(r'\b(?:rap|rapping)\b', text, re.IGNORECASE):
                vocal_style = "rap"
            elif re.search(r'\b(?:sing|singing|song)\b', text, re.IGNORECASE):
                vocal_style = "singing"
            elif re.search(r'\b(?:spoken|speech)\b', text, re.IGNORECASE):
                vocal_style = "spoken"
        
        return {
            "vocal_type": vocal_type,
            "vocal_style": vocal_style
        }
    
    @staticmethod 
    def _construct_from_patterns(text: str, category: Optional[str]) -> Optional[Dict[str, Any]]:
        """Letzter Fallback: Konstruiert JSON aus erkannten Mustern"""
        if not category:
            return None
        
        # Versuche strukturierte Antworten zu erkennen
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if category == "genre" and len(lines) >= 1:
            # Suche nach Genre-Listen
            for line in lines:
                if any(word in line.lower() for word in ["genre", "style", "music"]):
                    words = re.findall(r'\b[a-zA-Z]+\b', line)
                    genres = [w for w in words if len(w) > 3 and w.lower() not in ["genre", "style", "music"]]
                    if genres:
                        return {"genres": genres[:2]}
        
        return None


# Convenience Functions
def parse_category_response(text: str, category: str) -> Optional[Dict[str, Any]]:
    """Convenience-Funktion für Category-Response-Parsing"""
    return JSONParser.extract_json(text, category)

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Sichere JSON-Parsing ohne Category-Context"""
    return JSONParser.extract_json(text)


if __name__ == "__main__":
    # Test verschiedene JSON-Formate
    test_cases = [
        ('{"genres": ["electronic", "ambient"]}', "genre"),
        ('```json\n{"mood": ["calm", "peaceful"], "energy_level": "low"}\n```', "mood"),
        ('The genres are: electronic and ambient music', "genre"),
        ('This is a calm, peaceful track with low energy', "mood"),
        ('{"instruments": ["piano", "synthesizer", "drums"]}', "instruments"),
    ]
    
    for text, category in test_cases:
        result = parse_category_response(text, category)
        print(f"Input: {text}")
        print(f"Category: {category}")  
        print(f"Result: {result}")
        print("-" * 50)
