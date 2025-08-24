# ACE-DATA v2

Tooling, um automatisch Ace-Step kompatible Trainingsdaten aus Audiodateien zu erzeugen (Tags, optional Lyrics/BPM). Nutzt Qwen2-Audio (4-bit) und strikte Tag-Whitelist.

## Features
- Mehrkategorien-Tagging (Genre, Mood, Instruments, Vocal) mit Qwen2-Audio 7B Instruct
- Strikte Validierung gegen `presets/moods.md` (Alias/Fuzzy)
- Sauberes Logging (Konsole + Datei), Caching-Hooks vorbereitet
- Konfigurierbare Prompts in `config/prompts.json`

## Projektstruktur (Kurz)
- `scripts/` Orchestrator, Core (Model/Audio), Helpers (Parser/Tags/Logger)
- `config/` Prompts/Modell-Konfig
- `presets/` Whitelist-Tags (Genres/Moods/Instruments/Vocal)
- `data/` Audio und generierte `_prompt.txt` (nicht im Repo)

## Setup
1. Conda-Env aktivieren (Python 3.11):
- conda activate ace-data_v2_env
2. Abhängigkeiten installieren:
- pip install -r requirements.txt

## Run
- Ordner verarbeiten:
- python -m scripts.tagging.multi_tagger --input_dir data/audio --verbose
- Einzeldatei:
- python -m scripts.tagging.multi_tagger --file "data/audio/yourfile.mp3" --verbose
- Optional: Tags außerhalb Whitelist erlauben
- python -m scripts.tagging.multi_tagger --file "..." --allow_tag_extras

Ergebnis: `yourfile_prompt.txt` neben der Audio-Datei.

## Hinweise
- Logs und große Dateien sind per `.gitignore` ausgeschlossen.
- Wenn du Audio/Weights versionieren willst, nutze Git LFS (siehe `.gitattributes`).
- Energy/Technical-Tags sind ausgeschaltet (Ace-Step Fokus auf Genre/Mood/Instruments/Vocal).

## Lizenz
MIT (sofern nicht anders angegeben).
