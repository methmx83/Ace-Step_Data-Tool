# scripts/ui/optimized_web_ui.py
# Python 3.11 / Gradio 5.x
from __future__ import annotations

import io
import os
import re
import sys
import time
import subprocess
import shutil
from pathlib import Path
from typing import Iterator, Tuple, List

# Projektwurzel für Imports sicherstellen
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # 5.x

# Projekt-Helfer
from scripts.helpers.preset_loader import list_presets, resolve_preset_path
from scripts.helpers.logger_setup import get_session_logger

LOGGER = get_session_logger("WebUI")

CSS = """
.progress-wrap { width: 100%; background: #2e2e2e; border-radius: 6px; height: 18px; overflow: hidden; }
.progress-bar  { height: 100%; width: 0%; background: #f29d38; transition: width .15s linear; }
.box-title     { font-weight: 600; color:#ffd43b; margin-bottom:6px; }
"""

INSTRUCTIONS_DEFAULT = (
    "Anleitung:\n"
    "1) Preset wählen (lädt passende moods.md, z. B. default/hiphop).\n"
    "2) 'Start Tagging' startet die Vollanalyse (multi_tagger) für data/audio mit --verbose.\n"
    "3) Fortschritt, laufende Datei und Logs siehst du links live.\n"
)

FILE_LINE_RE = re.compile(r"FILE\s+(\d+)\s*/\s*(\d+)\s*:\s*(.+)")

# --- Prompt-Editor helpers ---
# We want to allow the user to list, view and edit any generated `_prompt.txt` files.
# The following functions scan the audio directory for prompt files, load their contents,
# and save modifications back to disk in a safe (atomic) manner. They are defined here
# because they don't depend on Gradio and can be easily tested or reused.  The
# audio directory is relative to the project root and matches the default input_dir
# for the tagging pipeline.

BASE_INPUT_DIR = "data/audio"

def _scan_prompt_files(base_dir: str = BASE_INPUT_DIR) -> Tuple[List[str], List[str]]:
    """Recursively find all *_prompt.txt files under the given base_dir.

    Returns a tuple of (labels, abs_paths). `labels` are the relative paths from
    the base_dir, suitable for display in a dropdown. `abs_paths` are the full
    paths to each file, used internally when loading or saving. If no files are
    found, both lists are empty.
    """
    base = (ROOT / base_dir).resolve()
    try:
        files = [p for p in base.rglob("*_prompt.txt") if p.is_file()]
    except Exception:
        files = []
    labels = [str(p.relative_to(base)) for p in files]
    abs_paths = [str(p) for p in files]
    return labels, abs_paths

def _load_prompt_text(rel_label: str, abs_paths: List[str]) -> Tuple[str, str]:
    """Load the text of the selected prompt file.

    The `rel_label` is the label chosen from the dropdown. We match it to an
    absolute path by checking for a suffix match in the stored `abs_paths` list.
    Returns a tuple (text, status_message). The status message can be shown
    alongside the text box to inform the user whether loading succeeded.
    """
    # Find the matching absolute path. We use endswith so that labels
    # containing nested folders still match correctly.
    match = None
    for ap in abs_paths:
        if ap.endswith(rel_label):
            match = ap
            break
    if not match:
        return "", f"⚠️ Datei nicht gefunden: {rel_label}"
    try:
        # Read the file as UTF-8 text
        with open(match, "r", encoding="utf-8") as fh:
            text = fh.read()
        return text, f"✅ Geladen: {rel_label}"
    except Exception as e:
        return "", f"❌ Fehler beim Laden: {e}"

def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically.

    Creates a temporary file in the same directory and then moves it into place.
    This avoids partially written files if the process is interrupted.
    """
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time())}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    os.replace(tmp, path)

def _save_prompt_text(rel_label: str, content: str, abs_paths: List[str]) -> str:
    """Save edited prompt text back to its file.

    Tries to locate the absolute path corresponding to `rel_label` from the list
    of absolute paths. Performs a simple sanitization (trim whitespace, unify
    line endings, collapse extra spaces and commas). Writes the file
    atomically. Returns a status message describing the outcome.
    """
    match = None
    for ap in abs_paths:
        if ap.endswith(rel_label):
            match = ap
            break
    if not match:
        return "⚠️ Datei nicht gefunden."
    dest = Path(match)
    try:
        # Take a backup of the file if it doesn't already exist
        bak = dest.with_suffix(dest.suffix + ".bak")
        if not bak.exists() and dest.exists():
            shutil.copyfile(dest, bak)
        # Normalize the input: convert CRLF to LF and collapse whitespace around commas
        text = content.replace("\r\n", "\n").strip()
        # Build a single comma-separated line of tags, trimming each
        parts = [p.strip() for p in re.split(r",|\n", text) if p.strip()]
        normalized = ", ".join(parts)
        _atomic_write(dest, normalized)
        return f"💾 Gespeichert: {rel_label}"
    except Exception as e:
        return f"❌ Fehler beim Speichern: {e}"

def _build_progress_html(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return f'''
    <div class="progress-wrap">
        <div class="progress-bar" style="width:{pct:.1f}%"></div>
    </div>
    '''

def _parse_progress_from_line(line: str) -> Tuple[int | None, int | None, str | None]:
    m = FILE_LINE_RE.search(line)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), m.group(3).strip()

def _eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0:
        return "--:--"
    rate = elapsed / done
    remain = max(0.0, rate * (total - done))
    mm, ss = divmod(int(remain), 60)
    return f"{mm:02d}:{ss:02d}"

def _run_cli_stream(preset_key: str, rap_style_enabled: bool, input_dir: str) -> Iterator[tuple]:
    # Preset-Pfad auflösen (unterstützt presets/moods.md UND presets/<sub>/moods.md)
    # Sicherstellen, dass wir immer einen string übergeben
    preset_path = resolve_preset_path(preset_key) or (ROOT / "presets" / "moods.md")
    preset_path = str(preset_path)

    # Vollanalyse EXAKT wie gewünscht; plus --moods_file für das gewählte Preset
    cmd = [
        sys.executable, "-m", "scripts.tagging.multi_tagger",
        "--input_dir", input_dir, "--verbose",
    "--suppress_header",
    "--moods_file", preset_path,
    ]
    LOGGER.info(f"Launching CLI: {' '.join(cmd)} (cwd={ROOT})")

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    log_buf = io.StringIO()
    seen_done = 0
    total = None
    current_file = ""

    # Initiales UI
    yield _build_progress_html(0.0), "**Status:** Starte …", "", "**Aktuelle Datei:** –"

    try:
        assert proc.stdout is not None
        start = time.time()
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            log_buf.write(line + "\n")
            # zusätzlich ins zentrale Log
            LOGGER.info(line)

            i, n, fname = _parse_progress_from_line(line)
            # explizite None-Checks (sicherer)
            if i is not None and n is not None:
                seen_done, total = i, n
                current_file = fname or current_file

            pct = 0.0
            if total and total > 0:
                pct = 100.0 * float(seen_done) / float(total)
                status = f"**Status:** Processing Songs – {pct:.1f}%  |  Datei {seen_done}/{total}"
            else:
                status = f"**Status:** {line[:160] if line else 'Läuft …'}"

            if total and total > 0:
                status += f"  |  ETA {_eta(time.time()-start, seen_done, total)}"

            current_md = f"**Aktuelle Datei:** {current_file or '–'}"
            yield _build_progress_html(pct), status, log_buf.getvalue(), current_md

        rc = proc.wait()
        elapsed = time.time() - start
        end_status = f"**Status:** Fertig (rc={rc}) – Dauer {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
        yield _build_progress_html(100.0), end_status, log_buf.getvalue(), f"**Aktuelle Datei:** {current_file or '–'}"

    except Exception as exc:
        LOGGER.exception("Fehler im CLI-Stream")
        end_status = f"**Status:** Fehler: {exc}"
        yield _build_progress_html(100.0), end_status, log_buf.getvalue(), f"**Aktuelle Datei:** {current_file or '–'}"

    finally:
        # Sicherstellen, dass Kindprozess beendet wird, falls UI abbricht
        try:
            if proc.poll() is None:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
        except Exception as exc:
            LOGGER.warning(f"Fehler beim Beenden des Subprozesses: {exc}")

def launch_ui():
    presets = list_presets() or ["default"]
    default_preset = presets[0]

    with gr.Blocks(css=CSS, title="ACE-DATA_v2 — Tagging") as demo:
        # Kopfzeile
        with gr.Row():
            preset_dd = gr.Dropdown(choices=presets, value=default_preset, label="Genre-Preset", interactive=True)
            rap_toggle = gr.Checkbox(value=False, label="Rap Style", interactive=True)

        # Breite Progressbar + Status
        progress_html = gr.HTML(_build_progress_html(0.0))
        status_md = gr.Markdown("**Status:** Bereit.")

        # Untere zwei Boxen
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('<div class="box-title">Process Log Box</div>')
                current_file_md = gr.Markdown("**Aktuelle Datei:** –")
                log_text = gr.Textbox(label="Live Log", value="", lines=18, interactive=False)
            with gr.Column(scale=1):
                gr.Markdown('<div class="box-title">Info Text Box</div>')
                info_text = gr.Textbox(label="Anleitung", value=INSTRUCTIONS_DEFAULT, lines=18, interactive=True)

        # Start-Button (nur Vollanalyse)
        start_btn = gr.Button(value="Start Tagging")

        def on_start(preset_key: str, rap_style: bool) -> Iterator[tuple]:
            input_dir = "data/audio"
            yield from _run_cli_stream(preset_key, rap_style, input_dir)

        # ------------------------------------------------------------------
        gr.Markdown("### 📝 Prompt-Editor (post-processing)")
        with gr.Row():
            prompt_files_state = gr.State([])
            prompt_file_dd = gr.Dropdown(
                label="Prompt-Datei",
                choices=[],
                value=None,
                interactive=True,
                scale=2,
            )
            reload_btn = gr.Button("↻ Liste neu laden", variant="secondary")

        prompt_text = gr.Textbox(
            label="Tags in dieser Datei",
            value="",
            lines=2,
            interactive=True,
        )
        with gr.Row():
            save_btn = gr.Button("💾 Speichern", variant="secondary")
            status_out = gr.Markdown("")

        # Callback to scan the directory for prompt files
        def ui_scan() -> tuple:
            labels, abs_paths = _scan_prompt_files()
            # If there is at least one file, select the first one so the user
            # sees something immediately; otherwise leave selection empty.
            value = labels[0] if labels else None
            return (
                gr.update(choices=labels, value=value),  # prompt_file_dd update
                abs_paths,  # prompt_files_state update
                "",       # prompt_text clear
                ""        # status_out clear
            )

        # Callback when the user selects a file from the dropdown
        def ui_load(selected_label: str, abs_paths: List[str]) -> tuple:
            if not selected_label:
                return "", "⚠️ Keine Datei gewählt."
            text, msg = _load_prompt_text(selected_label, abs_paths)
            return text, msg

        # Callback for saving the edited prompt text
        def ui_save(selected_label: str, text: str, abs_paths: List[str]) -> str:
            if not selected_label:
                return "⚠️ Keine Datei gewählt."
            return _save_prompt_text(selected_label, text, abs_paths)

        # Bind the reload, dropdown change, and save actions to the callbacks
        reload_btn.click(
            fn=ui_scan,
            inputs=[],
            outputs=[prompt_file_dd, prompt_files_state, prompt_text, status_out],
        )
        prompt_file_dd.change(
            fn=ui_load,
            inputs=[prompt_file_dd, prompt_files_state],
            outputs=[prompt_text, status_out],
        )
        save_btn.click(
            fn=ui_save,
            inputs=[prompt_file_dd, prompt_text, prompt_files_state],
            outputs=[status_out],
        )

        # Click the start button to run the full tagging.  After the run
        # completes we chain ui_scan() to automatically refresh the list of
        # prompt files.
        run_event = start_btn.click(
            fn=on_start,
            inputs=[preset_dd, rap_toggle],
            outputs=[progress_html, status_md, log_text, current_file_md],
            show_progress=False,
        )
        run_event.then(
            fn=ui_scan,
            inputs=[],
            outputs=[prompt_file_dd, prompt_files_state, prompt_text, status_out],
        )

    # In Gradio 5: queue()-Parameter wie concurrency_count entfallen.
    demo.launch()

if __name__ == "__main__":
    launch_ui()
