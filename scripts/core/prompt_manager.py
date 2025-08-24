"""
scripts/core/prompt_manager.py
Multi-prompt system for structured audio tag generation.

This module orchestrates the generation of tags by issuing multiple
category-specific prompts to the Qwen2-Audio model and parsing the
results. It supports ChatML template formatting, JSON parsing with
fallbacks, retry logic, error recovery and batch orchestration across
multiple prompt categories.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptCategory(Enum):
    """Categories used in the multi-prompt system."""
    GENRE = "genre"
    MOOD = "mood" 
    INSTRUMENTS = "instruments"
    TECHNICAL = "technical"
    VOCAL = "vocal"
    PRODUCTION = "production"
    ENERGY = "energy"

@dataclass
class PromptTemplate:
    """Template for a prompt category."""
    category: PromptCategory
    system_prompt: str
    user_prompt: str = ""
    expected_format: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 60
    temperature: float = 0.0
    retry_count: int = 2
    
    def format_prompt(self, **context) -> str:
        """Format the system and user prompts using provided context variables."""
        try:
            formatted_system = self.system_prompt.format(**context)
            if self.user_prompt:
                formatted_user = self.user_prompt.format(**context)
                return f"{formatted_system}\n\n{formatted_user}"
            return formatted_system
        except KeyError as e:
            logger.warning(f"Missing context variable in prompt template: {e}")
            return self.system_prompt

@dataclass
class PromptResult:
    """Result of a prompt execution."""
    category: PromptCategory
    raw_output: str
    parsed_data: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    processing_time: float = 0.0

class PromptManager:
    """
    Coordinate the multi-prompt system for structured tag generation.

    Workflow:
    1. Load prompt templates from JSON configuration
    2. Execute different categories sequentially
    3. Parse and validate JSON outputs
    4. Combine results into final tag lists
    """
    
    def __init__(self, prompts_config_path: Optional[str] = None):
        self.templates: Dict[PromptCategory, PromptTemplate] = {}
        
        if prompts_config_path and Path(prompts_config_path).exists():
            self.load_templates(prompts_config_path)
        else:
            self._load_default_templates()
        
        logger.info(f"PromptManager initialized with {len(self.templates)} templates")
    
    def load_templates(self, config_path: str):
        """Load prompt templates from a JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for category_name, template_data in config.get("prompt_templates", {}).items():
                try:
                    category = PromptCategory(category_name.lower())
                    template = PromptTemplate(
                        category=category,
                        system_prompt=template_data.get("system_prompt", ""),
                        user_prompt=template_data.get("user_prompt", ""),
                        expected_format=template_data.get("expected_format", {}),
                        max_tokens=template_data.get("max_tokens", 60),
                        temperature=template_data.get("temperature", 0.0),
                        retry_count=template_data.get("retry_count", 2)
                    )
                    self.templates[category] = template
                    logger.debug(f"Loaded template for category: {category_name}")
                    
                except ValueError as e:
                    logger.warning(f"Unknown prompt category: {category_name}, skipping")
                except Exception as e:
                    logger.error(f"Error loading template {category_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load prompt templates from {config_path}: {e}")
            self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default templates if no configuration is provided."""
        
        # Genre Detection Template
        self.templates[PromptCategory.GENRE] = PromptTemplate(
            category=PromptCategory.GENRE,
            system_prompt="""Analyze this audio and identify the musical genre.
Respond ONLY with valid JSON in this exact format:
{{"genres": ["primary_genre", "secondary_genre"]}}

Focus on:
- Primary genre (main style)
- Secondary genre or subgenre if applicable
- Use specific, accurate genre names
- Maximum 2 genres""",
            max_tokens=40,
            temperature=0.0
        )
        
        # Mood Analysis Template
        self.templates[PromptCategory.MOOD] = PromptTemplate(
            category=PromptCategory.MOOD,
            system_prompt="""Analyze the emotional characteristics and mood of this audio.
Respond ONLY with valid JSON in this exact format:
{{"mood": ["mood1", "mood2", "mood3"], "energy_level": "low|medium|high"}}

Mood categories:
- Emotional tone (happy, sad, nostalgic, aggressive, peaceful, etc.)
- Maximum 3 mood descriptors
- Energy level: low, medium, or high""",
            max_tokens=50,
            temperature=0.1
        )
        
        # Instrument Detection Template  
        self.templates[PromptCategory.INSTRUMENTS] = PromptTemplate(
            category=PromptCategory.INSTRUMENTS,
            system_prompt="""Identify the main instruments and sounds in this audio.
Respond ONLY with valid JSON in this exact format:
{{"instruments": ["instrument1", "instrument2", "instrument3"]}}

Focus on:
- Clearly audible instruments
- Most prominent 3-4 instruments
- Use standard instrument names (guitar, drums, piano, synthesizer, etc.)
- Include electronic elements if present""",
            max_tokens=45,
            temperature=0.0
        )
        
        # Technical Analysis Template
        self.templates[PromptCategory.TECHNICAL] = PromptTemplate(
            category=PromptCategory.TECHNICAL,
            system_prompt="""Analyze the technical and musical characteristics of this audio.
Respond ONLY with valid JSON in this exact format:
{{"key": "C major", "tempo": "medium|fast|slow", "time_signature": "4/4"}}

Analyze:
- Musical key (e.g., "C major", "A minor", "F# major")
- Tempo feel: slow, medium, fast
- Time signature (usually 4/4, 3/4, etc.)""",
            max_tokens=35,
            temperature=0.0
        )
        
        # Vocal Analysis Template
        self.templates[PromptCategory.VOCAL] = PromptTemplate(
            category=PromptCategory.VOCAL,
            system_prompt="""Analyze the vocal characteristics in this audio.
Respond ONLY with valid JSON in this exact format:
{{"vocal_type": "male|female|mixed|instrumental", "vocal_style": "style_description"}}

Identify:
- Vocal type: male, female, mixed, or instrumental (no vocals)
- Vocal style: singing style, rap, spoken, etc.
- If instrumental, set vocal_type to "instrumental" and vocal_style to "none" """,
            max_tokens=40,
            temperature=0.0
        )
        
        logger.info("Loaded default prompt templates")
    
    def execute_prompt(self, 
                      template: PromptTemplate, 
                      model_chat_fn, 
                      audio_file: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> PromptResult:
        """
        Execute a single prompt.

        Args:
            template: The prompt template to use.
            model_chat_fn: Callable for the model's ``chat`` method.
            audio_file: Path to the audio file (optional).
            context: Context variables for template formatting.

        Returns:
            A :class:`PromptResult` with parsed data.
        """
        import time
        
        start_time = time.time()
        context = context or {}
        
        # Prompt formatieren
        formatted_prompt = template.format_prompt(**context)
        
        result = PromptResult(
            category=template.category,
            raw_output="",
        )
        
        # Retry logic
        for attempt in range(template.retry_count + 1):
            try:
                logger.debug(f"Executing {template.category.value} prompt (attempt {attempt + 1})")
                
                # Execute model chat
                raw_output = model_chat_fn(
                    prompt=formatted_prompt,
                    audio_files=[audio_file] if audio_file else None,
                    max_new_tokens=template.max_tokens,
                    temperature=template.temperature
                )
                
                result.raw_output = raw_output
                result.processing_time = time.time() - start_time
                
                # Attempt JSON parsing
                parsed_data = self._parse_json_output(raw_output, template.category)
                
                if parsed_data:
                    result.parsed_data = parsed_data
                    result.tags = self._extract_tags_from_parsed(parsed_data, template.category)
                    result.success = True
                    
                    logger.debug(f"{template.category.value} successful: {result.tags}")
                    return result
                else:
                    logger.warning(f"{template.category.value} JSON parsing failed, attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"{template.category.value} execution failed (attempt {attempt + 1}): {e}")
                result.error = str(e)
        
        # All attempts have failed
        result.success = False
        logger.error(f"{template.category.value} failed after {template.retry_count + 1} attempts")
        return result
    
    def _parse_json_output(self, raw_output: str, category: PromptCategory) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from the model output using various fallback strategies.

        1. Search for ```json code blocks
        2. Search for pure JSON enclosed in braces
        3. Regex-based extraction of JSON fragments
        4. Category-specific fallback parsing
        """
        if not raw_output or not raw_output.strip():
            return None
        
        text = raw_output.strip()
        
        # Strategy 1: JSON Code-Block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Pure JSON (starts and ends with braces)
        if text.startswith('{') and text.endswith('}'):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON anywhere in text
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, text)
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Category-specific regex fallbacks
        return self._fallback_parsing(text, category)
    
    def _fallback_parsing(self, text: str, category: PromptCategory) -> Optional[Dict[str, Any]]:
        """Category-specific fallback parsing if JSON parsing fails."""
        
        if category == PromptCategory.GENRE:
            # Look for genre names
            genre_words = re.findall(r'\b(?:rock|pop|jazz|classical|electronic|hip.?hop|rap|blues|folk|country|metal|techno|house|trance|ambient|indie|punk|reggae|soul|funk|disco|dance)\b', text, re.IGNORECASE)
            if genre_words:
                return {"genres": list(set(genre_words[:2]))}
        
        elif category == PromptCategory.MOOD:
            # Look for mood descriptors
            mood_words = re.findall(r'\b(?:happy|sad|energetic|calm|aggressive|peaceful|melancholic|uplifting|dramatic|romantic|mysterious|playful|intense|relaxing|dark|bright)\b', text, re.IGNORECASE)
            energy_words = re.findall(r'\b(?:low|medium|high)\b.*?energy', text, re.IGNORECASE)
            energy = energy_words[0].split()[0].lower() if energy_words else "medium"
            
            if mood_words:
                return {"mood": list(set(mood_words[:3])), "energy_level": energy}
        
        elif category == PromptCategory.INSTRUMENTS:
            # Look for instrument names
            instrument_words = re.findall(r'\b(?:guitar|piano|drums|bass|violin|saxophone|trumpet|synthesizer|keyboard|vocal|voice|strings|brass|percussion|flute|clarinet|organ)\b', text, re.IGNORECASE)
            if instrument_words:
                return {"instruments": list(set(instrument_words[:4]))}
        
        return None
    
    def _extract_tags_from_parsed(self, parsed_data: Dict[str, Any], category: PromptCategory) -> List[str]:
        """Extract a list of tags from parsed data."""
        tags = []
        
        try:
            if category == PromptCategory.GENRE:
                genres = parsed_data.get("genres", [])
                tags.extend([str(g).lower().replace(" ", "-") for g in genres if g])
            
            elif category == PromptCategory.MOOD:
                moods = parsed_data.get("mood", [])
                energy = parsed_data.get("energy_level")
                tags.extend([str(m).lower().replace(" ", "-") for m in moods if m])
                if energy:
                    tags.append(f"energy-{energy}")
            
            elif category == PromptCategory.INSTRUMENTS:
                instruments = parsed_data.get("instruments", [])
                tags.extend([str(i).lower().replace(" ", "-") for i in instruments if i])
            
            elif category == PromptCategory.TECHNICAL:
                key = parsed_data.get("key")
                tempo = parsed_data.get("tempo")
                time_sig = parsed_data.get("time_signature")
                
                if key:
                    tags.append(f"key-{str(key).lower().replace(' ', '-')}")
                if tempo:
                    tags.append(f"tempo-{str(tempo).lower()}")
                if time_sig:
                    tags.append(f"time-{str(time_sig).replace('/', '')}")
            
            elif category == PromptCategory.VOCAL:
                vocal_type = parsed_data.get("vocal_type")
                vocal_style = parsed_data.get("vocal_style")
                
                if vocal_type and vocal_type != "none":
                    tags.append(f"vocal-{str(vocal_type).lower()}")
                if vocal_style and vocal_style != "none":
                    tags.append(f"style-{str(vocal_style).lower().replace(' ', '-')}")
        
        except Exception as e:
            logger.error(f"Tag extraction failed for {category.value}: {e}")
        
        return tags
    
    def execute_multi_prompt_workflow(self, 
                                    model_chat_fn, 
                                    audio_file: str,
                                    categories: Optional[List[PromptCategory]] = None,
                                    context: Optional[Dict[str, Any]] = None) -> Dict[PromptCategory, PromptResult]:
        """
        Execute the multi-prompt workflow.

        Args:
            model_chat_fn: Callable representing the model's chat function.
            audio_file: Path to the audio file to analyse.
            categories: Desired categories (default: all available).
            context: Context for template formatting.

        Returns:
            A dictionary mapping each category to its :class:`PromptResult`.
        """
        if categories is None:
            categories = list(self.templates.keys())
        
        logger.info(f"Starting multi-prompt workflow for {len(categories)} categories")
        
        results = {}
        
        for category in categories:
            if category not in self.templates:
                logger.warning(f"No template found for category: {category}")
                continue
            
            template = self.templates[category]
            result = self.execute_prompt(template, model_chat_fn, audio_file, context)
            results[category] = result
            
            # Brief pause between prompts for stability
            import time
            time.sleep(0.1)
        
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Multi-prompt workflow completed: {successful}/{len(categories)} successful")
        
        return results
    
    def combine_results_to_tags(self, results: Dict[PromptCategory, PromptResult]) -> List[str]:
        """
        Combine all prompt results into a final tag list.

        Duplicate tags are removed and the remaining tags are arranged
        in a logical order:

        1. Technical (BPM, key)
        2. Genre
        3. Instruments
        4. Vocals
        5. Mood and energy
        """
        combined_tags = []
        
        # Define the order for better tag arrangement
        category_order = [
            PromptCategory.TECHNICAL,
            PromptCategory.GENRE, 
            PromptCategory.INSTRUMENTS,
            PromptCategory.VOCAL,
            PromptCategory.MOOD,
            PromptCategory.ENERGY
        ]
        
        for category in category_order:
            if category in results and results[category].success:
                combined_tags.extend(results[category].tags)
        
        # Entferne Duplikate, behalte Reihenfolge
        seen = set()
        unique_tags = []
        for tag in combined_tags:
            if tag and tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags
    
    def get_available_categories(self) -> List[PromptCategory]:
        """Return the available prompt categories."""
        return list(self.templates.keys())
    
    def get_template(self, category: PromptCategory) -> Optional[PromptTemplate]:
        """Return the template for the specified category."""
        return self.templates.get(category)
    
    def add_custom_template(self, template: PromptTemplate):
        """Add a custom template to the manager."""
        self.templates[template.category] = template
        logger.info(f"Added custom template for category: {template.category.value}")


# Convenience Functions
def create_prompt_manager(config_path: Optional[str] = None) -> PromptManager:
    """Factory function for creating a :class:`PromptManager`."""
    return PromptManager(config_path)


def quick_tag_generation(model_chat_fn, audio_file: str, categories: Optional[List[str]] = None) -> List[str]:
    """
    Perform quick tag generation using the default templates.

    Args:
        model_chat_fn: Callable representing the model's chat function.
        audio_file: Path to the audio file to analyse.
        categories: Optional list of category names to restrict processing.

    Returns:
        A list of generated tags.
    """
    manager = PromptManager()
    
    # Convert string category names to enums
    if categories:
        try:
            category_enums = [PromptCategory(cat.lower()) for cat in categories]
        except ValueError as e:
            logger.warning(f"Invalid category in list: {e}")
            category_enums = None
    else:
        category_enums = None
    
    results = manager.execute_multi_prompt_workflow(
        model_chat_fn=model_chat_fn,
        audio_file=audio_file, 
        categories=category_enums
    )
    
    return manager.combine_results_to_tags(results)


if __name__ == "__main__":
    # Test code for development
    logging.basicConfig(level=logging.DEBUG)
    
    # Test Template Loading
    manager = PromptManager()
    print(f"Available categories: {[cat.value for cat in manager.get_available_categories()]}")
    
    # Test Template Formatting
    genre_template = manager.get_template(PromptCategory.GENRE)
    if genre_template:
        formatted = genre_template.format_prompt(artist="Test Artist", title="Test Song")
        print(f"Formatted prompt preview: {formatted[:200]}...")
