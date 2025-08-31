# lyrics_ui.py
# Python 3.11 / Gradio 5.x
from __future__ import annotations
import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # .../data_v2
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Iterator, List, Tuple

import gradio as gr

# ==== Imports aus deinem extrahierten Tool (1:1-Logik beibehalten) ====
# Falls du statt lyrics_tool die alten Einzeldateien verwendest,
# ersetze die Importe entsprechend (z. B. `from lyrics import ...`).
from scripts.helpers.shared_logs import log_message, LOGS
from scripts.helpers.lyrics import get_audio_metadata, fetch_and_save_lyrics
from scripts.helpers.clean_lyrics import bereinige_datei

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # your_project/
AUDIO_DIR = (Path(os.getenv("LYRICS_AUDIO_DIR"))
            if os.getenv("LYRICS_AUDIO_DIR")
            else ROOT / "data" / "audio")

# ---------- Hilfen ----------

def list_audio_files() -> List[Path]:
    exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    if not AUDIO_DIR.exists():
        return []
    return sorted([p for p in AUDIO_DIR.rglob("*") if p.suffix.lower() in exts])

def scan_lyrics_files() -> List[Path]:
    """Return a sorted list of Path objects for all *_lyrics.txt files under AUDIO_DIR."""
    if not AUDIO_DIR.exists():
        return []
    return sorted([p for p in AUDIO_DIR.rglob("*_lyrics.txt")])


def find_lyrics_display_names() -> List[str]:
    """Return display names for the dropdown: relative path to AUDIO_DIR or filename.

    Examples: "11 - Solo_lyrics.txt" or "subdir/11 - Solo_lyrics.txt" (forward slashes).
    """
    files = scan_lyrics_files()
    names: List[str] = []
    for p in files:
        try:
            rel = p.relative_to(AUDIO_DIR)
            names.append(rel.as_posix())
        except Exception:
            names.append(p.name)
    return sorted(names)


def resolve_lyrics_path(selection: str) -> Path:
    """Resolve a dropdown selection (display name) back to an absolute Path.

    Strategy:
    - If selection can be joined to AUDIO_DIR and exists, return that.
    - Otherwise search scanned files for a matching name or trailing match.
    - If nothing found, return the candidate path (will raise on IO operations).
    """
    if not selection:
        return Path("")
    candidate = AUDIO_DIR / Path(selection)
    if candidate.exists():
        return candidate
    # fallback: match by name or by path suffix
    for p in scan_lyrics_files():
        if p.name == selection or p.as_posix().endswith(selection.replace("\\", "/")):
            return p
    return candidate

def build_progress_html(pct: float) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return f"""
    <div class="progress-wrap">
        <div class="progress-bar" style="width:{pct:.1f}%"></div>
    </div>
    """

CSS = """
.progress-wrap { width: 100%; background: #2e2e2e; border-radius: 6px; height: 18px; overflow: hidden; }
.progress-bar  { height: 100%; width: 0%; background: #f29d38; transition: width .15s linear; }
.box-title     { font-weight: 600; color:#ffd43b; margin-bottom:6px; }
"""

# ---------- Generator: verarbeitet alle Dateien + streamt UI-Updates ----------

def process_all_files(overwrite: bool) -> Iterator[Tuple[str, str, str, dict, str]]:
    """
    Streamt (progress_html, status_text, log_text, dropdown_update, log_state)
    """
    logs = ""
    files = list_audio_files()
    total = len(files)

    if total == 0:
        logs += "⚠️ No audio files found in data/audio\n"
        yield build_progress_html(0.0), "**Status:** 🔵 Ready.", logs, gr.update(), logs
        return

    # Start-Status
    logs += f"🔍 Found {total} audio file(s)\n"
    yield build_progress_html(0.0), "**Status:** 🟢 working ⏳…", logs, gr.update(choices=find_lyrics_display_names()), logs

    done = 0
    for fpath in files:
        meta = get_audio_metadata(str(fpath))
        artist, title = meta.get("artist") or "", meta.get("title") or ""
        base = fpath.with_suffix("").name
        out_txt = fpath.with_name(f"{base}_lyrics.txt")

        # Überspringen, wenn vorhanden & overwrite=False
        if out_txt.exists() and not overwrite:
            msg = f"↩️ Skip existing: {out_txt.name}\n"
            logs += msg
            done += 1
            pct = 100.0 * done / total
            yield build_progress_html(pct), f"**Status:** {pct:.1f}%  |  skipped", logs, gr.update(choices=find_lyrics_display_names()), logs
            continue

        logs += f"🔄 Fetch: {artist} - {title}\n"
        ok = fetch_and_save_lyrics(artist, title, str(out_txt))
        if ok:
            # direkt säubern, identisch zur Pipeline-Logik
            bereinige_datei(str(out_txt))
            logs += f"✅ Saved & cleaned: {out_txt.name}\n"
        else:
            logs += f"✗ Not found: {artist} - {title}\n"

        done += 1
        pct = 100.0 * done / total
        status = f"**Status:** {pct:.1f}%  |  file {done}/{total}"
    yield build_progress_html(pct), status, logs, gr.update(choices=find_lyrics_display_names()), logs

    # Abschluss
    yield build_progress_html(100.0), "**Status:** ✅ Done.", logs, gr.update(choices=find_lyrics_display_names()), logs

# ---------- Dropdown -> Lyrics laden ----------

def load_lyrics_file(lyrics_path: str | None) -> str:
    if not lyrics_path:
        return ""
    try:
        full = resolve_lyrics_path(lyrics_path)
        return Path(full).read_text(encoding="utf-8")
    except Exception as e:
        return f"❌ Error loading file: {e}"

# ---------- Speichern aus dem Editor ----------

def save_lyrics_file(lyrics_path: str | None, text: str, log_state: str) -> Tuple[str, str]:
    if not lyrics_path:
        return log_state + "⚠️ No file selected.\n", log_state
    try:
        full = resolve_lyrics_path(lyrics_path)
        Path(full).parent.mkdir(parents=True, exist_ok=True)
        Path(full).write_text(text, encoding="utf-8", newline="\n")
        msg = f"💾 Saved: {Path(full).name}\n"
        return log_state + msg, log_state + msg
    except Exception as e:
        msg = f"❌ Save error: {e}\n"
        return log_state + msg, log_state + msg

# ---------- UI ----------

def build_interface():
    with gr.Blocks(css=CSS, title="Lyrics Scraper") as demo:
        gr.Markdown("### 🎶 Lyrics Scraper — fetch, clean & edit")

        # Eigene Progressbar + Status (kein Overlay!)
        progress_html = gr.HTML(build_progress_html(0.0))
        status_md = gr.Markdown("**Status:** 🔵 Ready.")

        # Zwei Spalten: links Log, rechts Editor
        log_state = gr.State("")
        with gr.Row():
            log_box = gr.Textbox(
                label="Live Log",
                value="",
                lines=18,
                interactive=False,
                show_copy_button=True,
            )
            with gr.Column():
                file_dropdown = gr.Dropdown(
                    label="Lyrics files",
                    choices=find_lyrics_display_names(),
                    value=None,
                    interactive=True,
                )
                editor_box = gr.Textbox(
                    label="Lyrics Editor",
                    value="",
                    lines=18,
                    interactive=True,
                )
                save_button = gr.Button("Save Lyrics")

        # Controls unter den Boxen
        overwrite_checkbox = gr.Checkbox(label="Overwrite lyrics", value=False)
        fetch_button = gr.Button("Get Lyrics", variant="primary")

        # Callbacks
        fetch_button.click(
            fn=process_all_files,
            inputs=[overwrite_checkbox],
            outputs=[progress_html, status_md, log_box, file_dropdown, log_state],
            queue=True,
            show_progress=False,   # <<< wichtig: Overlay AUS
        )

        file_dropdown.change(
            fn=load_lyrics_file,
            inputs=[file_dropdown],
            outputs=[editor_box],
            show_progress=False,   # sauberer ohne Overlay
        )

        save_button.click(
            fn=save_lyrics_file,
            inputs=[file_dropdown, editor_box, log_state],
            outputs=[log_box, log_state],
            show_progress=False,
        )

        return demo

if __name__ == "__main__":
    demo = build_interface()
    # Eigener Port, damit nichts mit anderen UIs kollidiert
    demo.launch(server_port=7860, share=False)
