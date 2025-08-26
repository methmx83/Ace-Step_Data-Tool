# ACE-DATA v2

Tooling, um automatisch ACE-STEP-kompatible Trainingsdaten aus Audiodateien zu erzeugen (Tags, optional Lyrics/BPM). Nutzt Qwen2-Audio (4-bit/8-bit) und eine strikte Tag-Whitelist in `presets/moods.md`.

## Kurzüberblick
- Mehrkategorien-Tagging: `genre`, `key` (major/minor), `mood`, `instruments`, `vocal` und `vocal_fx` (z. B. `autotune`, `harmony`, `pitch-up`).
- Starke Validierung gegen `presets/moods.md` (Alias/Fuzzy-Matching, Whitelist-Regeln).
- Robustes JSON-Parsing für LLM-Outputs (Objekte/Arrays, Codeblöcke, gequotetes JSON, Fallbacks).
- Content-basiertes Retry pro Kategorie (konfigurierbar) und Audio-Caching / Multi-Segment-Processing.

## Features
- Modularer Orchestrator: PromptBuilder, InferenceRunner, SegmentPlanner, TagPipeline, ContextExtractor
- Konfigurierbare Prompts in `config/prompts.json` (inkl. neue Templates für `key` und `vocal_fx`)
- Sauberes Logging (Konsole + Datei)

## Projektstruktur (Kurz)
- `scripts/`
	- `tagging/multi_tagger.py` — Orchestrator und CLI
	- `tagging/tag_pipeline.py` — Extraktion / Normalisierung / Policy-Auswahl
	- `core/model_loader.py` — Qwen2-Audio-Integration (Quantisierungsoptionen)
	- `core/audio_processor.py` — Preprocessing, Segmentierung, Cache
	- `core/prompt_builder.py` — Prompt-Templates aus `config/prompts.json`
	- `core/inference_runner.py` — Model-Calls, Retry-Logik, Parsing
	- `core/segment_planner.py` — Planung der Segmente pro Kategorie
	- `helpers/json_parser.py` — Robuste Extraktion & Fallbacks (jetzt auch `key`/`vocal_fx`)
	- `helpers/tag_processor.py` — Normalisierung, Alias-Map, Conflict-Resolution (inkl. `keys` + `vocal_fx`)
	- `helpers/context_extractor.py` — Kontext aus Dateinamen
	- `helpers/logger_setup.py` — Session-Logging
- `config/` — Prompt- & Modell-Configs (`config/prompts.json` enthält neue Kategorien)
- `presets/` — `moods.md` mit Whitelist-Tags (Genres, Moods, Instruments, Vocal Types, Keys, Vocal Fx)
- `data/` — `audio/`, `cache/`, `output/` (Audio und generierte `_prompt.txt`)

## Architektur & Flow
1. `ContextExtractor` liest Artist/Title/BPM aus Dateinamen.
2. `SegmentPlanner` plant Segmente laut `workflow_config` und cached die Union via `AudioProcessor`.
3. `PromptBuilder` erzeugt system+user prompts je Kategorie.
4. `InferenceRunner` ruft das Modell auf (mehrere Audiopfade pro Kategorie möglich), inklusive technischer und content-basierter Retries.
5. `TagPipeline` extrahiert Roh-Tags je Kategorie, normalisiert gegen Whitelist (in `presets/moods.md`), wendet Min/Max/Order/Gesamtlimits an und löst Konflikte.
6. Orchestrator schreibt finale Tags als `*_prompt.txt` neben die Audio-Datei.

## Setup
1) Conda-Env (Python 3.11) aktivieren / erstellen

```bat
conda create -n ace-data_v2_env python=3.11 -y
conda activate ace-data_v2_env
```

2) Abhängigkeiten installieren

```bat
pip install -r requirements.txt
```

3) (Optional Windows helpers)

```bat
RUN_env.bat   # setzt lokale ENV-Variablen
RUN.bat       # Beispiel-Run (konfiguriert)
```

## Run (Windows, cmd)
- Kompletten Ordner verarbeiten:

```bat
python -m scripts.tagging.multi_tagger --input_dir data\audio --verbose
```

- Einzeldatei verarbeiten:

```bat
python -m scripts.tagging.multi_tagger --file "data\audio\yourfile.mp3" --verbose
```

- Tags außerhalb der Whitelist erlauben:

```bat
python -m scripts.tagging.multi_tagger --file "data\audio\yourfile.mp3" --allow_tag_extras --verbose
```

Ergebnis: `yourfile_prompt.txt` neben der Audio-Datei.

## Wichtige Konfig-Optionen
- `config/prompts.json` — Prompt-Templates und `workflow_config.default_categories` (Standard enthält jetzt `key` und `vocal_fx`).
- `workflow_config.audio_segments` — z. B. `["best","middle"]` (werden gecacht).
- `output_format.min_tags_per_category` / `max_tags_per_category` — Min/Max je Kategorie.

## Presets / Whitelist
- `presets/moods.md` enthält die erlaubten Tags für `genres`, `moods`, `instruments`, `vocal types`, `keys` und `vocal_fx`.
- Du kannst einen alternativen Pfad per Umgebungsvariable `ACE_MOODS_MD` setzen.

## Tests & Smoke-Checks
- Empfohlen: kurze Unit-Tests für `JSONParser` (z. B. `key` / `vocal_fx` Fallbacks) und `TagProcessor` (Whitelist/Alias/Fuzzy).
- Wenn du Tests möchtest, generiere ich gerne passende `pytest`-Dateien.

## Performance / Hinweise
- Qwen2-Audio-Modelle (7B) können viel VRAM benötigen; verwende 4-bit/8-bit-Quantisierung oder CPU-Offload in `core/model_loader.py` falls nötig.
- Für schnelle Runs: `workflow_config.audio_segments` auf `best`/`middle` setzen statt `full`.

## Troubleshooting
- Fehlende Tags: Prüfe Logs (Konsole + Logdatei). Der Parser versucht mehrere Fallbacks: JSON-Objekte, Arrays, Codeblöcke, gequotetes JSON und heuristische Textsuche.
- Vocals vs Genre: `rap` ist ein Genre; Vocal-Typen sind eigene Whitelist-Einträge (z. B. `male vocal`).

## Weiteres
- Falls du möchtest, erstelle ich Unit-Tests oder ein kurzes E2E-Skript, das eine Beispiel-Audiodatei durch die Pipeline jagt (ohne Modell-Load für schnelle lokale Checks).

## Lizenz
MIT
