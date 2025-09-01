<h1 align="center">🎵 ACE-Step Data-Tool</h1>
<p align="center">
  <strong>Generate clean, structured audio metadata.</strong><br>
  <em>*Extracts lyrics, tags & BPM from audio files – fully automated*</em>
</p>

<p align="center">
  <img src="./docs/screenshot.jpg" alt="ACE-Step Data Tool Screenshot" width="50%">
</p>


### ✨ Features
- 🧠 **LLM-powered Tag Generator** – (genre, moods, bpm, key, instruments, vocals and rap style)
- 🎙️  **Lyric Detection** – automatically via Genius.com
- 🕺   **BPM Analysis** – via Librosa
- 🖥️ **Modern WebUI** – with mood slider, genre presets & custom prompt field
- 🗂️  **Export to ACE-Step training format**
- 🔁  **Retry logic & logging built-in**

### 💻 Recommended Setup
	| Component  | Recommended   |
	|------------|---------------|
	| OS         | Windows 10 Pro|
	| GPU        | 12 GB VRAM    |
	| RAM        | 32 GB         |
	| Python     | 3.11          |
	| CUDA       | 12.9          |
	| Model      | `Qwen2-Audio-7B-Instruct`|

### Windows Installation
1. *Install NVIDIA Video Driver:*
	- You should install the latest version of your GPUs driver. Download drivers here: [NVIDIA GPU Drive](https://www.nvidia.com/Download/index.aspx).

2. *Install CUDA Toolkit:*
	- Follow the instructions to install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

3. *Install PyTorch:*
	- Install `torch` and `triton`. 
	- Go to https://pytorch.org to install it. For example `pip install torch torchvision torchaudio triton`
	- You will need the correct version of PyTorch that is compatible with your CUDA drivers, so make sure to select them carefully.
	- [Install PyTorch](https://pytorch.org/get-started/locally/).
	- Confirm if CUDA is installed correctly. Try `nvcc`. If that fails, you need to install `cudatoolkit` or CUDA drivers.

4. *Install BitsandBytes:*
	- Install `bitsandbytes` and check it with `python -m bitsandbytes`
	
### Installation
**Conda Installation** *(recommended)*
```bash
conda create --name acedata python=3.11
conda activate acedata
```

*Install Pytorch*
```bash
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

*Clone the repository*
```bash
git clone https://github.com/methmx83/Ace-Step_Data-Tool.git
cd Ace-Step_Data-Tool
```

*Install dependencies*
```bash
pip install -e .
```

### 🚀 Quickstart
	
**Launch the WebUI**
```bash
conda activate acedata
acedata
```
**Alternative**
```bash
conda activate acedata
python start.py
```
*Open WebUI* [http://localhost:7860]
  

### Example: 
	*Content of a* **_prompt.txt**
	When the pipeline processes an audio file, a `_prompt.txt` is created next to the file. It contains a simple, comma-separated list of tags. Example:
	`pop, bpm-114, electronic, minor, sad, piano, synth-pad, female-vocal`

*Notes:*
	- The BPM tag has the format `bpm-<INT>` (e.g., `bpm-114`).
	- Tags are lowercase and ideally hyphen-separated (`synth-pad`, `female-vocal`).
	- The pipeline automatically adds the BPM tag (if detected) and removes duplicate `bpm-*` entries.
	- If you want to manually adjust the order in the file, you can use the prompt editor in the WebUI.
	- `config/` — Prompt & Model Configs (`config/prompts.json` contains new categories)
	- `presets/` — `moods.md` with whitelist tags (Genres, Moods, Instruments, Vocal Types, Keys, Vocal Fx)
	- `data/` — `audio/`, `cache/`, `output/` (Audio and generated `_prompt.txt`)

## Quick Overview
	- Multi-category tagging: `genre`, `key` (major/minor), `mood`, `instruments`, `vocal`, and `vocal_fx` (e.g., `autotune`, `harmony`, `pitch-up`).
	- Configurable prompts in `config/prompts.json` (includes new templates for `key` and `vocal_fx`)
	- Content-based retry per category (configurable) and audio caching / multi-segment processing.
	- Modular orchestrator: PromptBuilder, InferenceRunner, SegmentPlanner, TagPipeline, ContextExtractor
	- Clean logging (console + file)
	
## BPM Detection: There is now a dedicated script for BPM analysis at `scripts/helpers/bpm.py`
	- Function: `detect_tempo(audio_path: str) -> Optional[float]` detects the tempo and returns a number on success.
	- Integration: The pipeline calls the detection before prompt/tag generation and adds a normalized tag in the format `bpm-XXX` to the generated `_prompt.txt` files.

## Key Modules
	- `scripts/helpers/lyrics.py` — Core logic for determining artist/title, constructing Genius URLs, fallback search, and writing `<audio-basename>_lyrics.txt`.
	- `scripts/helpers/clean_lyrics.py` — Post-processing: removes headers/metadata and writes cleaned lyrics (e.g., remove everything before the first `[Verse]` marker).
	- `scripts/helpers/metadata.py` — String normalization for filenames/URLs (e.g., `normalize_string`, `clean_filename`, `clean_rap_metadata`).
	- `scripts/helpers/shared_logs.py` — Central log buffer (`LOGS`) and `log_message()` for consistent UI display.

## New WebUI: `scripts/ui/ui.py` (Gradio) is the current interface
    - Start: `python -m scripts.ui.ui` starts the WebUI.
    - The UI internally starts `multi_tagger` as a subprocess (with `--suppress_header`), shows live logs, and offers a prompt editor for post-processing the `_prompt.txt` files.

## Lyrics Acquisition (Lyrics Scraping
	- The tool attempts to automatically retrieve the lyrics for the respective piece from the internet. It uses Genius.com as the source, first forming a suitable lyrics URL from the artist/title data. 
	- If the direct URL is unsuccessful, a search query is executed on Genius and the first matching result is used. The lyrics are extracted as plain text (using `Requests` + `BeautifulSoup4`) and saved in a file `<Name>_lyrics.txt`.
	**File Format:** *Each `<Name>_lyrics.txt` file initially records the artist and title, followed by the complete lyrics (UTF-8 encoded).
	**Note:** *A short pause is inserted between requests to avoid overloading the target website. If no text is found, the tool aborts processing of that song (i.e., no tag generation occurs without lyrics – purely instrumental tracks or unknown songs are skipped with a corresponding note).

## Directory Scan & File Processing
	- By default, the tool expects a folder (`data/audio` in the project directory) containing audio files (supported: `.mp3`, `.wav`, `.flac`, `.m4a`).
	- All files (recursively in subfolders) are read and processed one after another. Intermediate results and logs are displayed for each track.	

## Architecture & Flow
	1. `ContextExtractor` reads Artist/Title/BPM from filename.
	2. `SegmentPlanner` plans segments according to `workflow_config` and caches the union via `AudioProcessor`.
	3. `PromptBuilder` generates system+user prompts per category.
	4. `InferenceRunner` calls the model (multiple audio paths per category possible), including technical and content-based retries.
	5. `TagPipeline` extracts raw tags per category, normalizes against the whitelist (in `presets/moods.md`), applies Min/Max/Order/Overall limits, and resolves conflicts.
	6. Orchestrator writes final tags as `*_prompt.txt` next to the audio file.

## Important Config Options
	- `config/prompts.json` — Prompt templates and `workflow_config.default_categories` (standard now includes `key` and `vocal_fx`).
	- `workflow_config.audio_segments` — e.g., `["best","middle"]` (are cached).
	- `output_format.min_tags_per_category` / `max_tags_per_category` — Min/Max per category.

## Presets / Whitelist
	- `presets/moods.md` contains the allowed tags for `genres`, `moods`, `instruments`, `vocal types`, `keys`, and `vocal_fx`.
	- New: `presets/hiphop/moods.md` is an example preset for Hip-Hop-specific tags. Select it in the UI or via `--moods_file presets/hiphop/moods.md` in the CLI run.

### Enable Rap Style
	- CLI: Enable by adding `--moods_file presets/hiphop/moods.md` or set `workflow_config.default_categories` in `config/prompts.json` to include `rap_style`.

## Performance / Notes
	- Qwen2-Audio models (7B) can require a lot of VRAM; we use 4-bit quantization.
	- For fast runs: set `workflow_config.audio_segments` to `best`/`middle` instead of `full`.

## Troubleshooting
	- Missing tags: Check logs (console + log file). The parser attempts several fallbacks: JSON objects, arrays, code blocks, quoted JSON, and heuristic text search.

## Legal / Notes
	- Web scraping of sites like Genius may be subject to restrictions by their Terms of Service. Please check the legal situation before running automated scrapes on a large scale.




### License
	Apache-2.0 license

## License & Third-Party Attributions
	This repository and the included code are distributed under the Apache License, Version 2.0. The full license text is included in the `LICENSE` file at the repository root.

	Third-party components included in this project are documented in `third_party/THIRD_PARTY_LICENSES.md` and `NOTICE`. Several files and modules were derived from or inspired by other projects that are themselves licensed under Apache-2.0. Those original copyright notices and license headers are retained in the copied files where present.
	- [ACE-Step] (https://github.com/ace-step/ACE-Step)
	- [woctordho] (https://github.com/woct0rdho/ACE-Step)
