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
    "Instructions:\n"
    "1) Copy audio files (MP3) to data/audio.\n"
    "2) Select a preset / default for mixed songs.\n"
    "3) Click on 'start tagging'.\n" 
    "- You can see the progress, current file, and logs live on the left.\n"
)

FILE_LINE_RE = re.compile(r"FILE\s+(\d+)\s*/\s*(\d+)\s*:\s*(.+)")

# --- Laufsteuerung für Start/Stop ---
RUN_PROC: subprocess.Popen | None = None
STOP_REQUESTED: bool = False

def _terminate_process(proc: subprocess.Popen) -> None:
    """Robustes Beenden des Subprozesses (Windows-Fallback inkl.)."""
    try:
        if proc.poll() is None:
            proc.terminate()
            time.sleep(0.6)
    except Exception as e:
        LOGGER.warning(f"terminate() failed: {e}")

    if proc.poll() is None:
        try:
            proc.kill()
            time.sleep(0.3)
        except Exception as e:
            LOGGER.warning(f"kill() failed: {e}")

    if proc.poll() is None and os.name == "nt":
        # Windows: notfalls taskkill
        try:
            os.system(f'taskkill /PID {proc.pid} /T /F')
        except Exception as e:
            LOGGER.warning(f"taskkill failed: {e}")

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

def _run_cli_stream(preset_key: str, input_dir: str) -> Iterator[tuple]:
    """Startet den echten CLI-Prozess als Subprozess und streamt STDOUT live."""
    global RUN_PROC, STOP_REQUESTED

    # Preset-Pfad (unterstützt presets/moods.md UND presets/<sub>/moods.md)
    preset_path = resolve_preset_path(preset_key) or (ROOT / "presets" / "moods.md")
    preset_path = str(preset_path)

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
    RUN_PROC = proc
    STOP_REQUESTED = False

    log_buf = io.StringIO()
    seen_done = 0
    total = None
    current_file = ""

    # Initial
    yield _build_progress_html(0.0), "**Status:** Starting …", "", "**Current file:** –"

    try:
        assert proc.stdout is not None
        start = time.time()
        for raw_line in proc.stdout:
            if STOP_REQUESTED:
                LOGGER.info("Stop requested by user. Terminating process …")
                _terminate_process(proc)
                status = "**Status:** ⏹️ Aborted"
                yield _build_progress_html(100.0), status, log_buf.getvalue(), f"**Current file:** {current_file or '–'}"
                break

            line = raw_line.rstrip("\n")
            log_buf.write(line + "\n")
            LOGGER.info(line)

            i, n, fname = _parse_progress_from_line(line)
            if i is not None and n is not None:
                seen_done, total = i, n
                current_file = fname or current_file

            pct = 0.0
            if total and total > 0:
                pct = 100.0 * float(seen_done) / float(total)
                status = f"**Status:** Processing Songs – {pct:.1f}%  |  file {seen_done}/{total}"
                status += f"  |  ETA {_eta(time.time()-start, seen_done, total)}"
            else:
                status = f"**Status:** {line[:160] if line else 'Running …'}"

            yield _build_progress_html(pct), status, log_buf.getvalue(), f"**Current file:** {current_file or '–'}"

        # Wenn nicht vorher abgebrochen wurde, normalen Abschluss signalisieren
        if not STOP_REQUESTED:
            rc = proc.wait()
            elapsed = time.time() - start
            end_status = f"**Status:** Fertig (rc={rc}) – Dauer {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
            yield _build_progress_html(100.0), end_status, log_buf.getvalue(), f"**Aktuelle Datei:** {current_file or '–'}"

    except Exception as exc:
        LOGGER.exception("Error in CLI stream")
        end_status = f"**Status:** Error: {exc}"
        yield _build_progress_html(100.0), end_status, log_buf.getvalue(), f"**Current file:** {current_file or '–'}"

    finally:
        try:
            if proc.poll() is None:
                _terminate_process(proc)
        finally:
            RUN_PROC = None
            STOP_REQUESTED = False

# -------- Prompt-Editor Helpers --------

BASE_INPUT_DIR = "data/audio"

def _scan_prompt_files(base_dir: str = BASE_INPUT_DIR) -> Tuple[List[str], List[str]]:
    base = Path(ROOT / base_dir).resolve()
    files = [p for p in base.rglob("*_prompt.txt") if p.is_file()]
    labels = [str(p.relative_to(base)) for p in files]
    abs_paths = [str(p) for p in files]
    return labels, abs_paths

def _load_prompt_text(rel_label: str, abs_paths: List[str]) -> Tuple[str, str]:
    match = None
    for ap in abs_paths:
        try:
            if str(Path(ap)).endswith(rel_label):
                match = ap; break
        except Exception:
            continue
    if not match:
        return "", f"⚠️ File not found: {rel_label}"
    try:
        text = Path(match).read_text(encoding="utf-8")
        return text, f"✅ Loaded: {rel_label}"
    except Exception as e:
        return "", f"❌ Error loading: {e}"

def _atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time())}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)

def _save_prompt_text(rel_label: str, content: str, abs_paths: List[str]) -> str:
    match = None
    for ap in abs_paths:
        if str(Path(ap)).endswith(rel_label):
            match = ap; break
    if not match:
        return "⚠️ File not found."
    dest = Path(match)
    try:
        bak = dest.with_suffix(dest.suffix + ".bak")
        if not bak.exists():
            shutil.copyfile(dest, bak)

        text = content.replace("\r\n", "\n").strip()
        text = ", ".join([t.strip() for t in text.split(",") if t.strip()])
        _atomic_write(dest, text)
        return f"💾 Saved: {rel_label}"
    except Exception as e:
        return f"❌ Error saving: {e}"

# ------------- UI -------------

def launch_ui():
    presets = list_presets() or ["default"]
    default_preset = presets[0]

    with gr.Blocks(css=CSS, title="ACE-DATA_v2 — Tagging") as demo:
        # Kopfzeile
        with gr.Row():
            preset_dd = gr.Dropdown(choices=presets, value=default_preset, label="Genre-Preset", interactive=True)
            rap_toggle = gr.Checkbox(value=False, label="Rap Style", interactive=True, visible=False)  # optional ausblenden

        # Breite Progressbar + Status
        progress_html = gr.HTML(_build_progress_html(0.0))
        status_md = gr.Markdown("**Status:** Ready.")

        # Untere zwei Boxen (links breiter, rechts schmaler)
        with gr.Row():
            with gr.Column(scale=2):
                current_file_md = gr.Markdown("**Current file:** –")
                log_text = gr.Textbox(label="Live Log", value="", lines=18, interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("<br>")
                gr.Markdown('<div class="box-title"></div>')
                info_text = gr.Textbox(label="Information", value=INSTRUCTIONS_DEFAULT, lines=11, interactive=True)

                # Buttons unter der rechten Text-Box (vertikal)
                with gr.Column():
                    start_btn = gr.Button(value="Start Tagging", variant="secondary")
                    stop_btn  = gr.Button(value="Stop", variant="stop")

        # ---- Start/Stop Logik ----

        def on_start(preset_key: str) -> Iterator[tuple]:
            # rap_toggle ist logische Deko; Preset steuert Kategorien
            input_dir = "data/audio"
            yield from _run_cli_stream(preset_key, input_dir)

        def on_stop() -> str:
            global STOP_REQUESTED, RUN_PROC
            if RUN_PROC is None or RUN_PROC.poll() is not None:
                return "**Status:** No active run."
            STOP_REQUESTED = True
            try:
                _terminate_process(RUN_PROC)
            except Exception as e:
                LOGGER.warning(f"Stop error: {e}")
            return "**Status:** ⏹️ Cancellation requested …"

        run_event = start_btn.click(
            fn=on_start,
            inputs=[preset_dd],
            outputs=[progress_html, status_md, log_text, current_file_md],
            show_progress=False,
        )
        stop_btn.click(
            fn=on_stop,
            inputs=[],
            outputs=[status_md],
            show_progress=False,
        )

        # --- Prompt-Editor (unterhalb der Buttons / dem zweispaltigen Bereich) ---
        gr.Markdown("### 📝 Prompt-Editor (post-processing)")
        with gr.Row():
            prompt_files_state = gr.State([])
            prompt_file_dd = gr.Dropdown(label="Prompt-file", choices=[], value=None, interactive=True, scale=2)
            reload_btn = gr.Button("↻ Reload list", variant="secondary", scale=1)

        prompt_text = gr.Textbox(label="Tags in this file", value="", lines=2, interactive=True)
        with gr.Row():
            save_btn = gr.Button("💾 Save", variant="secondary")
            status_out = gr.Markdown("")

        def ui_scan() -> tuple:
            labels, abs_paths = _scan_prompt_files()
            value = labels[0] if labels else None
            return gr.update(choices=labels, value=value), abs_paths, "", ""

        def ui_load(selected_label: str, abs_paths: List[str]) -> tuple:
            if not selected_label:
                return "", "⚠️ No file selected."
            text, msg = _load_prompt_text(selected_label, abs_paths)
            return text, msg

        def ui_save(selected_label: str, text: str, abs_paths: List[str]) -> str:
            if not selected_label:
                return "⚠️ No file selected."
            return _save_prompt_text(selected_label, text, abs_paths)

        reload_btn.click(fn=ui_scan, inputs=[], outputs=[prompt_file_dd, prompt_files_state, prompt_text, status_out])
        prompt_file_dd.change(fn=ui_load, inputs=[prompt_file_dd, prompt_files_state], outputs=[prompt_text, status_out])
        save_btn.click(fn=ui_save, inputs=[prompt_file_dd, prompt_text, prompt_files_state], outputs=[status_out])

        # Nach dem Lauf automatisch die Liste der Prompt-Dateien aktualisieren
        run_event.then(fn=ui_scan, inputs=[], outputs=[prompt_file_dd, prompt_files_state, prompt_text, status_out])

    demo.launch()

if __name__ == "__main__":
    launch_ui()
