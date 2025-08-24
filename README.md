# ACE-DATA v2

Tooling, um automatisch Ace-Step kompatible Trainingsdaten aus Audiodateien zu erzeugen (Tags, optional Lyrics/BPM). Nutzt Qwen2-Audio (4-bit) und strikte Tag-Whitelist.

## Features
- Mehrkategorien-Tagging (Genre, Mood, Instruments, Vocal) mit Qwen2-Audio 7B Instruct (4-bit)
- Strikte Validierung gegen `presets/moods.md` (Alias/Fuzzy, Vocal-Whitelist)
- Robustes JSON-Parsing: Objekte/Arrays, Codeblöcke, gequotetes/escaptes JSON, Key-Normalisierung, Fallbacks
- Content-basierter Retry pro Kategorie (konfigurierbar, z. B. Mood-Overrides)
- Audio-Caching + Multi-Segment (z. B. `best`, `middle`), 16 kHz Mono WAV in `data/cache`
- Sauberes Logging (Konsole + Datei)
- Konfigurierbare Prompts in `config/prompts.json`

## Projektstruktur (Kurz)
- `scripts/` Orchestrator, Core (Model/Audio), Helpers (Parser/Tags/Logger)
- `config/` Prompts/Modell-Konfig
- `presets/` Whitelist-Tags (Genres/Moods/Instruments/Vocal)
- `data/` Audio und generierte `_prompt.txt` (nicht im Repo)

## Setup
1) Conda-Env aktivieren (Python 3.11)
- conda activate ace-data_v2_env
2) Abhängigkeiten installieren
- pip install -r requirements.txt
3) Optional (Windows): Batch starten
- RUN_env.bat (Environment) / RUN.bat (Standardlauf)

## Run (Windows, cmd)
- Kompletten Ordner verarbeiten:
	```bat
	python -m scripts.tagging.multi_tagger --input_dir data\audio --verbose
	```
- Einzeldatei:
	```bat
	python -m scripts.tagging.multi_tagger --file "data\audio\yourfile.mp3" --verbose
	```
- Tags außerhalb Whitelist erlauben:
	```bat
	python -m scripts.tagging.multi_tagger --file "data\audio\yourfile.mp3" --allow_tag_extras
	```

Ergebnis: `yourfile_prompt.txt` neben der Audio-Datei.

## Konfiguration (Kurz)
- `workflow_config.default_categories`: Reihenfolge der Kategorien (Standard: genre, mood, instruments, vocal)
- `workflow_config.audio_segments`: Welche Segmente verarbeitet werden (z. B. ["best", "middle"]) – werden gecacht
- `workflow_config.content_retry`: Content-basierter Retry (enabled, max_attempts, delay_seconds, temperature_boost, overrides pro Kategorie)
- `output_format.min_tags_per_category`/`max_tags_per_category`: Min/Max je Kategorie (z. B. Genre 2, Mood 2–3, Instruments 2–3, Vocal 1)
- `output_format.max_total_tags`: Gesamtlimit

Hinweis: Genres und Vocals werden strikt validiert; „rap“ ist ein Genre, Vocal-Typen sind whitelisted (z. B. „male vocal“, „female vocal“, „spoken word“, Feature-Varianten, „instrumental“).

## Troubleshooting
- Genres fehlen: Siehe `ContentRetry.genre` im Log – der Parser akzeptiert jetzt Arrays/escaped JSON, bei Parsing-Fehlern wird nachgefordert.
- Moods fehlen: Mood-Override im `content_retry` aktiv, Fallback erkennt u. a. „relaxed“, „excited“.
- Vocals uneindeutig: Whitelist greift; „rap“ allein ist kein Vocal-Typ.

## Lizenz
MIT (sofern nicht anders angegeben).
