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
- `scripts/` — enthält Tagging, Training-Helpers und WebUI
- `presets/` — Whitelists und Spezial-Presets
- `config/` — Prompt- und Modellkonfigurationen
- `third_party/` — Third-party license attributions and copies

## Quickstart (Windows, cmd)
1) Conda-Env (Python 3.11) aktivieren / erstellen

```bat
conda create -n ace-data_v2_env python=3.11 -y
conda activate ace-data_v2_env
```

2) Abhängigkeiten installieren

```bat
pip install -r requirements.txt
```

3) WebUI starten (optional)

```bat
cd scripts\ui
python ui.py
```

## License & Third-Party Attributions

This repository and the included code are distributed under the Apache License, Version 2.0. The full license text is included in the `LICENSE` file at the repository root.

Third-party components included in this project are documented in `third_party/THIRD_PARTY_LICENSES.md` and `NOTICE`. Several files and modules were derived from or inspired by other projects that are themselves licensed under Apache-2.0. Those original copyright notices and license headers are retained in the copied files where present.

If you are a contributor or a copyright owner of code referenced in `third_party/THIRD_PARTY_LICENSES.md` and require changes to attribution, please open an issue.

### How this affects you

- You may use, modify and redistribute this code under the terms of the Apache-2.0 license.
- If you redistribute or modify the code, keep the existing NOTICE and license headers in files that were copied or derived from Apache-2.0 projects.
- This repository contains both original code (authored by the repository owner) and third-party code that remains under Apache-2.0. See `third_party/THIRD_PARTY_LICENSES.md` for per-file provenance and notes.

### Contributing

Please follow these simple rules when contributing:

- Add a header comment to any file that copies or modifies third-party code describing the original source, a short change description and the license (Apache-2.0). Example header (add at top of modified files):

```text
# Original: <upstream path> from <upstream repo URL>
# Upstream commit: <REPLACE_WITH_COMMIT_SHA>
# Modifications: (brief list of changes applied, e.g. "Windows path normalization; VRAM optimizations; HDF5 changes")
# License: Apache-2.0 (see repository LICENSE)
```

- Keep changes small and focused and include a short description in the PR about any third-party code touched.

### Setting the repository license on GitHub

To make the license visible on the project page on GitHub, ensure the `LICENSE` file is committed at the repository root. GitHub will detect the license automatically and show it on the repo main page.

If you prefer to release your own files under a different license (for example MIT), document this clearly in `README.md` and in `third_party/THIRD_PARTY_LICENSES.md` which files are Apache-2.0 and which are under your chosen license. Mixing licenses is allowed, but requires explicit documentation.

## Further help

If you want, I can:

- Try to locate and replace the upstream commit SHAs for the files listed in `third_party/THIRD_PARTY_LICENSES.md`.
- Insert header templates into the specific modified files (`scripts/train/convert2hf_dataset_new.py`, `scripts/train/preprocess_dataset_new.py`, `scripts/train/trainer_optimized.py`).
- Create a small CI/Dev-check script that verifies presence of header comments in modified files.
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
 - Neuerungen: spezialisiertes Hiphop-Preset & `rap_style` Kategorie
	 - `presets/hiphop/moods.md` enthält ein spezialisiertes Set an Moods/Genres/rap-styles für Hip-Hop-Workflows.
	 - Neue Kategorie `rap_style` (Prompt und Parser) erkennt Stile wie `trap`, `mumble rap`, `lyrical rap` und wird optional per UI/CLI aktiviert.

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

	## Neu

	- BPM-Erkennung: Es gibt jetzt ein dediziertes Skript zur BPM-Analyse unter `scripts/helpers/bpm.py`.
		- Funktion: `detect_tempo(audio_path: str) -> Optional[float]` erkennt das Tempo und gibt bei Erfolg eine Zahl zurück.
		- Integration: Die Pipeline ruft die Erkennung vor der Prompt-/Tag-Erzeugung auf und fügt ein normalisiertes Tag im Format `bpm-XXX` in die generierten `_prompt.txt` Dateien ein.

	- Neues WebUI: `scripts/ui/ui.py` (Gradio) ist das aktuelle Interface.
		- Start: `python -m scripts.ui.ui` startet die WebUI.
		- Die UI startet intern `multi_tagger` als Subprozess (mit `--suppress_header`), zeigt Live-Logs und bietet einen Prompt-Editor zum Nachbearbeiten der `_prompt.txt` Dateien.

	Hinweis: Wenn du die Logs beim direkten CLI-Start weniger ausführlich haben willst, nutze `--suppress_header` für `multi_tagger`.

## Lyrics Scraper (neu)

Kurz: Das Projekt enthält jetzt einen eigenständigen Lyric‑Scraper, der Lyrics aus dem Web (z. B. Genius.com) holt, die Dateien bereinigt und über die WebUI editierbar macht. Die relevanten Module liegen unter `scripts/helpers/`.

Wichtige Module
- `scripts/helpers/lyrics.py` — Kernlogik zum Ermitteln von Artist/Title, Konstruktion von Genius‑URLs, Fallback‑Suche und Schreiben von `<audio-basename>_lyrics.txt`.
- `scripts/helpers/clean_lyrics.py` — Post‑Processing: entfernt Kopfzeilen/Metadaten und schreibt bereinigte Lyrics (z. B. alles vor erstem `[Verse]`-Marker entfernen).
- `scripts/helpers/metadata.py` — String‑Normalisierung für Dateinamen/URLs (z. B. `normalize_string`, `clean_filename`, `clean_rap_metadata`).
- `scripts/helpers/shared_logs.py` — Zentraler Log‑Puffer (`LOGS`) und `log_message()` für konsistente UI‑Anzeige.

Verwendung
- WebUI (empfohlen für interaktives Arbeiten):
	```bat
	cd scripts\ui
	python ui.py
	```
	Im Browser Tab "Lyrics" findest du: `Get Lyrics` (läuft Scraper + speichert), `Overwrite lyrics` (steuert Überschreiben), Live‑Log und `Save Lyrics` (speichert manuell editierten Text).

- Programmgesteuert / CLI: Es gibt helper‑Funktionen in `scripts/helpers/lyrics.py` — z. B. `process_single_file(path)` oder `get_lyrics(artist, title)` für Scripting in eigenen Tools oder Tests.

Ausgabe
- `<audio-basename>_lyrics.txt` — die rohe bzw. bereinigte Lyrics‑Datei (UTF‑8).
- Optional werden Begleitdateien wie `_prompt.txt` oder Backup‑Dateien (`.bak`) erzeugt, abhängig von Pipeline‑Schritten.

Wichtig für Entwickler
- Logging: Verwende `shared_logs.log_message()` anstelle von `print()` in allen Lyrics‑Modulen, damit die WebUI die Logs konsistent anzeigen kann.
- HTTP/Robustheit: Beim Scrapen empfiehlt sich eine Retry/Backoff‑Logik bei 429/5xx Antworten und ein moderates `REQUEST_DELAY` (z. B. 1.0–2.0s), um IP‑Sperren zu vermeiden.
- Tests: Erstelle Unit‑Tests für `normalize_string`, `clean_rap_metadata` und `bereinige_datei` (temporäre Dateien als Fixtures). Für `scrape_genius_lyrics` nutze lokale HTML‑Fixtures statt Live‑Requests in CI.

Rechtliches / Hinweise
- Web‑Scraping von Seiten wie Genius kann Einschränkungen durch deren AGB unterliegen. Bitte prüfe die rechtliche Lage bevor du automatisierte Scrapes in großem Maßstab fährst.

Weiteres
- Empfehlung: Kleine Code‑Änderung — ersetze verbleibende `print()`‑Aufrufe in `scripts/helpers/lyrics.py` durch `shared_logs.log_message()`; das verbessert die UI‑Integration.

	## Beispiel: Inhalt einer _prompt.txt

	Wenn die Pipeline eine Audiodatei verarbeitet, wird neben der Datei eine `_prompt.txt` erzeugt. Sie enthält eine einfache, kommagetrennte Liste von Tags. Beispiel:

	pop, bpm-114, electronic, minor, sad, piano, synth-pad, female-vocal

	Hinweise:
	- Das BPM-Tag hat das Format `bpm-<INT>` (z. B. `bpm-114`).
	- Tags sind lowercase und idealerweise hyphen-separated (`synth-pad`, `female-vocal`).
	- Die Pipeline fügt das BPM-Tag automatisch hinzu (wenn erkannt) und entfernt doppelte `bpm-*` Einträge.
	- Wenn du die Reihenfolge in der Datei manuell anpassen willst, kannst du den Prompt-Editor in der WebUI verwenden.
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
 - Neu: `presets/hiphop/moods.md` ist ein Beispiel-Preset für Hip-Hop-spezifische Tags. Wähle es in der UI oder via `--moods_file presets/hiphop/moods.md` beim CLI-Run.

### Rap Style aktivieren
- In der WebUI gibt es eine Checkbox `Enable rap_style`. Wenn aktiviert, wird die Kategorie `rap_style` an das Modell angefragt und die Parser-Fallbacks greifen.
- CLI: aktiviere durch Hinzufügen von `--moods_file presets/hiphop/moods.md` oder setze `workflow_config.default_categories` in `config/prompts.json` so, dass `rap_style` enthalten ist.

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
Apache-2.0 license
