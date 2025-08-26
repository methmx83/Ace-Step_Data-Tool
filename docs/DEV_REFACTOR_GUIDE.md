## Developer Refactor & README Ergänzung

Kurz: Dieses Dokument fasst konkrete Refactor-Schritte, Test‑Ideen und schnelle Befehle zusammen, um die Wartbarkeit der `ACE-DATA_v2` Pipeline zu verbessern.

### Ziel
- Bessere Trennung von Verantwortlichkeiten (parsing, normalisierung, heuristiken, IO)
- Einfachere Testbarkeit (Unit-Tests für Parser-/Heuristik-Logik)
- Geringeres Fehlerrisiko bei Änderungen am LLM-Output-Parsing

### Wichtige Dateien (quick map)
- `scripts/helpers/json_parser.py` — LLM-Output parsing + fallbacks (schwerpunkt: robustes JSON extrahieren)
- `scripts/helpers/tag_processor.py` — lädt `moods.md` und stellt erlaubte Tags bereit
- `scripts/tagging/tag_pipeline.py` — Orchestriert Kategorie-Requests und kombiniert Ergebnisse
- `scripts/ui/app.py` — Gradio UI, preset-Auswahl und CLI/Prozess-Aufruf
- `scripts/core/model_loader.py` — Modell-Initialisierung (Qwen2-Audio)

### Kleiner Entwickler-Contract (für Parser-Funktionen)
- Input: `text: str` (LLM-Output), `category: Optional[str]` (z. B. "rap_style")
- Output: `Optional[Dict[str, Any]]` mit normalisierten Schlüsseln (z. B. `{"rap_style": ["trap"]}`)
- Fehler-Modus: liefert `None` bei nicht-parsbarer Antwort; loggt Warnungen
- Erfolgskriterium: gültiges JSON oder plausibles Fallback-Resultat

### Vorschläge für Refactor-Schritte (priorisiert)
1. Extrahiere Parsing-Strategien in Untermodule
   - `scripts/helpers/parser/core.py` → Kernlogik: `extract_json`, `_try_parse_json`, `_clean_json_text`
   - `scripts/helpers/parser/unescape.py` → alle heuristiken für escaped/double-escaped JSON
   - `scripts/helpers/parser/fallbacks.py` → kategoriespezifische Fallbacks (`_parse_rap_style_fallback`, `_parse_vocal_fx_fallback`)
   Vorteil: kleinere Dateien, leichteres Testen einzelner Strategien.

2. Kapsle Normalisierung
   - `scripts/helpers/normalizer.py` mit Funktionen `normalize_keys(dict)`, `wrap_single_values(dict, keys=[])`
   - Testen: Input-Varianten ("Rap Style", "rap-style", `rap_style`) → immer `rap_style` key

3. TagProcessor als dataclass
   - `@dataclass` mit Feldern für erlaubte sets (`genres`, `moods`, `instruments`, `vocal_types`, `keys`, `vocal_fx`, `rap_style`)
   - Liefert Methoden: `is_allowed(tag, category)`, `find_allowed_in_text(text, category)`

4. Tests
   - Unit-Tests für `json_parser` (happy path + edge cases)
   - Integrationstest: kleine Audio-/mock-run der Pipeline, prüft erzeugte Prompt-Datei enthält `rap_style` wenn aktiviert

### Konkrete kleine Code-Hinweise (copy-paste friendly)
- Sicheres Quoting in `_clean_json_text`:
```python
def _clean_json_text(text: str) -> str:
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # quote only unquoted keys
    def _q(m):
        return f"{m.group('prefix')}\"{m.group('key')}\": "
    text = re.sub(r'(?m)(?P<prefix>[\{,\s])(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*', _q, text)
    text = text.replace("'", '"')
    return text
```

- Isoliere unescape-Logik in eine Funktion, iteriere `replace('\\"', '"')` und `unicode_escape` decoding — so fängst du doppelt-escaped Variationen.

### Tests: Minimal pytest Beispiele
- Datei: `tests/test_json_parser_basic.py`
```python
from scripts.helpers.json_parser import JSONParser

def test_rap_style_variants():
    inputs = [
        '{"rap_style": ["trap"]}',
        '{"Rap Style": ["lyrical"]}',
        '"{\\"Rap Style\\": [\\"oldschool\\"]}"',
    ]
    for s in inputs:
        res = JSONParser.extract_json(s, 'rap_style')
        assert res and 'rap_style' in res
```

### Quick commands (Windows `cmd.exe`)
Führe Unit-Tests:
```bat
REM Umgebung aktivieren falls nötig
CALL RUN_env.bat
python -m pytest -q
```

Einzeldatei taggen (CLI) — Beispiel:
```bat
REM erzeugt tags für eine Datei und nutzt den HipHop-Preset
RUN.bat --file "data/audio/03 - Deine Schuld.mp3" --moods_file "presets/hiphop/moods.md" --verbose
```

### Edge-Cases / Risiken
- Model kann unvorhersehbare Keys liefern (z. B. `RapStyle` ohne Leerzeichen). Investiere in Normalisierung und Tests.  
- Grobe Regex-Substitutionen können gültiges JSON zerstören — vermeide globale `re.sub` ohne Kontext.  
- Performance: viele Unicode-Decodes / Iterationen sind lichtgewichtig im Vergleich zu Modell-Inferenz; dennoch nur bei Bedarf ausführen.

### Vorschlag: kleine Roadmap (2–4 Stunden Arbeit)
1. Extrahiere `unescape` und `_clean_json_text` in eigenes Modul (30–60min) + unit tests (30 min).  
2. Extrahiere `fallbacks` (30–60min) und füge Tests für heuristics (15–30min).  
3. Schreibe Integrationstest, starte Tagging für 1–2 Dateien (20–40min).

Wenn du willst, kann ich die Refactor-PRs schrittweise erstellen (eine Datei pro PR) und jeweils Tests hinzufügen.

---
Kurzer Abschluss: Sag mir, ob ich eine erste PR mit nur dem `unescape`-Modul und zugehörigen Tests anlegen soll — oder ob du lieber zuerst lokal testen möchtest. 
