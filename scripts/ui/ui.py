from __future__ import annotations
import io
import os
import re
import sys
import time
import subprocess
import shutil
import json
from pathlib import Path
from typing import Iterator, Tuple, List

# Projektwurzel für Imports sicherstellen
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # 5.x

# Projekt-Helfer
from scripts.ui import lyrics_ui as lyr  # reuse der Lyrics-Callbacks & Utils
from scripts.helpers.preset_loader import list_presets, resolve_preset_path
from scripts.helpers.logger_setup import get_session_logger

LOGGER = get_session_logger("WebUI")
# Header-Bild
HEADER_URL = "https://raw.githubusercontent.com/methmx83/Ace-Step_Data-Tool/refs/heads/main/scripts/ui/assets/top.png"

CSS = """
.progress-wrap { width: 100%; background: #024142; border-radius: 6px; height: 18px; overflow: hidden; }
.progress-bar  { height: 100%; width: 0%; background: #0a5baf; transition: width .15s linear; }

/* --- Global: max width for the entire Gradio app --- */
.gradio-container, .gradio-root, .container {
    max-width: 1500px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 12px;
    padding-right: 12px;
    box-sizing: border-box;
}

/* --- CSS für das Header-Bild --- */
.app-header-image {{
    width: 100%;
    height: auto;
    background: transparent !important;
    display: block; /* Verhindert kleine Lücken unter dem Bild */
}}


/* --- Styling für den Inhaltsbereich der Tabs --- */
.tabitem {{
    background-color: rgba(0, 0, 0, 0.5);
    border-radius: 12px;
    padding: 20px;
    margin-top: 10px;
}}
"""

INSTRUCTIONS_DEFAULT = (
    "📝 Instructions:\n"
    "\n"
    "1) Copy audio files (MP3) to data/audio.\n"
    "2) Select a preset / default for mixed songs.\n"
    "3) Click on 'start tagging'.\n"
    "\n"
    "➡️ Hip-Hop preset includes a Rap-Style Tag.\n"
    "➡️ See the progress, current file, and logs live on the left.\n"
)
AUDIO_DIR = ROOT / "data" / "audio"
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
        "--cleanup-cache",
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
    yield _build_progress_html(0.0), "**Status:** 🟢 working ⏳…", "", "**Current file:** –"

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
                status = f"**Status:** ▶️ Processing Songs – {pct:.1f}%  |  file {seen_done}/{total}"
                status += f"  |  ETA {_eta(time.time()-start, seen_done, total)}"
            else:
                status = f"**Status:** {line[:160] if line else '🟢 Running …'}"

            yield _build_progress_html(pct), status, log_buf.getvalue(), f"**Current file:** {current_file or '–'}"

        # Wenn nicht vorher abgebrochen wurde, normalen Abschluss signalisieren
        if not STOP_REQUESTED:
            rc = proc.wait()
            elapsed = time.time() - start
            end_status = f"**Status:** ✅ Finished (rc={rc}) – Duration {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
            yield _build_progress_html(100.0), end_status, log_buf.getvalue(), f"**Current file:** {current_file or '–'}"

    except Exception as exc:
        LOGGER.exception("Error in CLI stream")
        end_status = f"**Status:** ❌ Error: {exc}"
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
    # --- Collect available LoRA config JSON files ---
    # We scan the 'config/lora' folder for .json files to populate the dropdown
    lora_dir = ROOT / "config" / "lora"
    if lora_dir.exists():
        try:
            lora_jsons = [p.name for p in lora_dir.glob("*.json") if p.is_file()]
        except Exception:
            lora_jsons = []
    else:
        lora_jsons = []

    theme = None  # Standardwert, falls das Laden fehlschlägt
    theme_path = Path(__file__).parent / "acedata.json"
    
    try:
        with open(theme_path, "r", encoding="utf-8") as f:
            theme_dict = json.load(f)
            theme = gr.Theme.from_dict(theme_dict)
        print("Theme 'acedata.json' successfully loaded.")
    except Exception as e:
        print(f"Error loading theme: {e}")
        # Fallback auf Standard-Theme
        theme = gr.themes.Default()

    # =======================================================

    with gr.Blocks(css=CSS, title="Ace-Step Data-Tool", theme=theme) as demo:

        # BILD-HEADER
        gr.HTML(f'<img src="{HEADER_URL}" class="app-header-image">')

        with gr.Tabs():
            # --- Tagging Tab ---
            with gr.TabItem("🎼 Tagging"):
                # Kopfzeile
                with gr.Row():
                    preset_dd = gr.Dropdown(choices=presets, value=default_preset, label="🎤 Genre-Preset", interactive=True)
                    rap_toggle = gr.Checkbox(value=False, label="Rap Style", interactive=True, visible=False)  # optional ausblenden

                # Breite Progressbar + Status
                progress_html = gr.HTML(_build_progress_html(0.0))
                status_md = gr.Markdown("**Status:** 🔵 Ready.")

                # Untere zwei Boxen (links breiter, rechts schmaler)
                with gr.Row():
                    with gr.Column(scale=2):
                        current_file_md = gr.Markdown("**Current file:** –")
                        log_text = gr.Textbox(label="📝 Live Log", value="", lines=18, interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("<br>")
                        gr.Markdown('<div></div>')
                        info_text = gr.Textbox(label="ℹ️ Information", value=INSTRUCTIONS_DEFAULT, lines=11, interactive=True)

                        # Buttons unter der rechten Text-Box (vertikal)
                        with gr.Column():
                            start_btn = gr.Button(value="Start Tagging", variant="secondary")
                            stop_btn  = gr.Button(value="Stop", variant="stop")
                            clear_logs_btn = gr.Button(value="Clear Logs", variant="danger")

                # ---- Start/Stop Logik ----

                def on_start(preset_key: str) -> Iterator[tuple]:
                    # rap_toggle ist logische Deko; Preset steuert Kategorien
                    input_dir = "data/audio"
                    yield from _run_cli_stream(preset_key, input_dir)

                def on_stop() -> str:
                    global STOP_REQUESTED, RUN_PROC
                    if RUN_PROC is None or RUN_PROC.poll() is not None:
                        return "**Status:** 🟡 No active run."
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

                def on_clear_logs() -> str:
                    """Recursively deletes all .log files in the project logs folder and returns a status."""
                    try:
                        logs_dir = Path(ROOT) / "logs"
                        if not logs_dir.exists():
                            return "**Status:** ⚠️ No 'logs' folder found."
                        removed = 0
                        errors = []
                        for pattern in ("*.log", "*.txt"):
                            for p in logs_dir.rglob(pattern):
                                try:
                                    p.unlink()
                                    removed += 1
                                except Exception as e:
                                    errors.append(f"{p}: {e}")
                        msg = f"**Status:** 🗑️ Deleted: {removed} .log files."
                        if errors:
                            LOGGER.warning(f"Log cleanup errors: {errors[:5]}")
                            msg += f" Errors occurred while deleting: {len(errors)} files (see server logs)."
                        return msg
                    except Exception as e:
                        LOGGER.exception("Error clearing logs")
                        return f"**Status:** ❌ Error clearing logs: {e}"

                clear_logs_btn.click(
                    fn=on_clear_logs,
                    inputs=[],
                    outputs=[status_md],
                    show_progress=False,
                )

                # --- Prompt-Editor (post-processing) in Tagging tab ---
                gr.Markdown("### 🎼 Prompt-Editor (post-processing)")
                with gr.Row():
                    prompt_files_state = gr.State([])
                    prompt_file_dd = gr.Dropdown(label="📝 Prompt-file", choices=[], value=None, interactive=True, scale=2)
                    reload_btn = gr.Button("↻ Reload list", variant="secondary", scale=1)

                prompt_text = gr.Textbox(label="▶️ Tags in this file", value="", lines=2, interactive=True)
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

            # --- Lyrics Tab ---
            with gr.TabItem("🎶 Lyrics"):
                l_progress_html = gr.HTML(lyr.build_progress_html(0.0))
                l_status_md = gr.Markdown("**Status:** 🔵 Ready.")
                l_log_state = gr.State("")
                with gr.Row():
                    l_log_box = gr.Textbox(label="📝 Live Log", value="", lines=18, interactive=False, show_copy_button=True)
                    with gr.Column():
                        l_file_dropdown = gr.Dropdown(label="📁 Lyrics files", choices=lyr.find_lyrics_display_names(), value=None, interactive=True)
                        l_editor_box = gr.Textbox(label="📝 Lyrics Editor", value="", lines=18, interactive=True)
                        l_save_button = gr.Button("💾 Save Lyrics")
                l_overwrite_checkbox = gr.Checkbox(label="Overwrite 🎶 Lyrics", value=False)
                l_fetch_button = gr.Button("Get Lyrics", variant="primary")

                l_fetch_button.click(
                    fn=lyr.process_all_files,
                    inputs=[l_overwrite_checkbox],
                    outputs=[l_progress_html, l_status_md, l_log_box, l_file_dropdown, l_log_state],
                    queue=True,
                    show_progress=False,
                )
                l_file_dropdown.change(
                    fn=lyr.load_lyrics_file,
                    inputs=[l_file_dropdown],
                    outputs=[l_editor_box],
                    show_progress=False,
                )
                l_save_button.click(
                    fn=lyr.save_lyrics_file,
                    inputs=[l_file_dropdown, l_editor_box, l_log_state],
                    outputs=[l_log_box, l_log_state],
                    show_progress=False,
                )        
                # Prompt-Editor removed from Lyrics tab (moved to Tagging tab)
            # --- Finetuning Tab ---
            with gr.TabItem("🧬 Finetuning 𓀋"):
                # Erklärung: Diese Registerkarte erlaubt die Ausführung der Datensatz-Erstellung und des Finetunings.
                # Aufbau: Links ein Live-Log, rechts Buttons für die drei Schritte.
                with gr.Row():
                    with gr.Column(scale=2):
                        finetune_log = gr.Textbox(label="📝 Live Log", value="", lines=20, interactive=False)
                    with gr.Column(scale=1):
                        # Buttons für die Finetuning-Pipeline. Die Buttons werden vertikal angeordnet.
                        convert_btn = gr.Button(value="Convert Dataset", variant="secondary")
                        create_btn = gr.Button(value="Create Dataset", variant="secondary")
                        train_btn = gr.Button(value="Start Finetuning", variant="secondary")
                        stop_ft_btn = gr.Button(value="Stop", variant="stop")

                # Parameter-Slider für Trainingsoptionen
                # Ermöglichen die Anpassung von max_steps und num_workers durch den Benutzer.
                with gr.Row():
                    max_steps_slider = gr.Slider(
                        minimum=100,
                        maximum=20000,
                        step=100,
                        value=1000,
                        label="⚙️ Max Steps"
                    )
                    num_workers_slider = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=8,
                        label="👷 Num Workers"
                    )
                # Neue Eingabefelder für Checkpoint-Einstellungen
                with gr.Row():
                    save_every_n_steps_slider = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        step=10,
                        value=100,
                        label="💾 Save every N steps"
                    )
                    save_last_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                        label="📊 Keep last N checkpoints"
                    )

                # Auswahl für LoRA-Konfigurationsdatei und Eingabe für Checkpoint-Pfad
                # Die Dropdown-Liste zeigt alle gefundenen .json Dateien im Ordner config/lora an
                with gr.Row():
                    lora_config_dd = gr.Dropdown(
                        choices=lora_jsons,
                        value=lora_jsons[0] if lora_jsons else "",
                        label="📑 LoRA Config",
                        interactive=True,
                    )
                    ckpt_path_input = gr.Textbox(
                        value="data/lora/r256_8bit",
                        label="💾 LoRa Checkpoints Path",
                        interactive=True,
                    )
                # Neues Textfeld für last_lora_path
                with gr.Row():
                    last_lora_path_input = gr.Textbox(
                        value="",
                        label="📁 Resume from LoRA (optional)",
                        placeholder="data/lora/.../pytorch_lora_weights.safetensors",
                        interactive=True,
                    )    

                # Hilfsfunktion: generischer Prozess-Runner für Finetuning-Befehle
                def _run_finetune_command(cmd: List[str]) -> Iterator[str]:
                    """Starts a subprocess for finetuning tasks and streams stdout live."""
                    global RUN_PROC, STOP_REQUESTED
                    # Verhindere parallele Ausführung
                    if RUN_PROC is not None and RUN_PROC.poll() is None:
                        # Es läuft bereits ein Prozess
                        yield "⚠️ Another process is already running. Please stop it before starting a new one.\n"
                        return
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            cwd=str(ROOT),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                    except Exception as exc:
                        LOGGER.exception("⚠️ Failed to start process")
                        yield f"❌ Could not start process: {exc}\n"
                        return
                    RUN_PROC = proc
                    STOP_REQUESTED = False
                    log_buf = io.StringIO()
                    try:
                        assert proc.stdout is not None
                        for raw_line in proc.stdout:
                            # Prüfe, ob ein Abbruch gewünscht ist
                            if STOP_REQUESTED:
                                LOGGER.info("⚠️ Stop requested for Finetuning. Terminating process …")
                                _terminate_process(proc)
                                log_buf.write("\n[⚠️ Process aborted]\n")
                                yield log_buf.getvalue()
                                break
                            line = raw_line.rstrip("\n")
                            log_buf.write(line + "\n")
                            LOGGER.info(line)
                            yield log_buf.getvalue()
                        # Prozess beendet sich regulär
                        if not STOP_REQUESTED:
                            rc = proc.wait()
                            log_buf.write(f"\n[✅ Process finished (rc={rc})]\n")
                            yield log_buf.getvalue()
                    except Exception as exc:
                        LOGGER.exception("❌ Error while streaming process output")
                        log_buf.write(f"\n[Error: {exc}]\n")
                        yield log_buf.getvalue()
                    finally:
                        # Aufräumen
                        try:
                            if proc.poll() is None:
                                _terminate_process(proc)
                        finally:
                            RUN_PROC = None
                            STOP_REQUESTED = False

                def on_convert() -> Iterator[str]:
                    """Execution of the convert2hf dataset script with fixed parameters."""

                    # Initiale Log-Nachricht
                    yield "🟢 Convert files to json... please wait...⏳\n"

                    cmd = [
                        sys.executable,
                        "scripts/train/convert2hf_dataset_new.py",
                        "--data_dir",
                        "data/audio",
                        "--output_name",
                        "data/data_sets/jsons_sets",
                    ]
                    yield from _run_finetune_command(cmd)

                def on_create() -> Iterator[str]:
                    """Execution of the preprocess_dataset script with fixed parameters."""

                    # Initiale Log-Nachricht
                    yield "🟢 Creating the hdf5 dataset... please wait...⏳\n"

                    cmd = [
                        sys.executable,
                        "scripts/train/preprocess_dataset_new.py",
                        "--input_name",
                        "data/data_sets/jsons_sets",
                        "--output_dir",
                        "data/data_sets/train_set",
                    ]
                    yield from _run_finetune_command(cmd)


                def on_train(max_steps_val: int, num_workers_val: int, save_every_n_steps_val: int, save_last_val: int, selected_lora_config: str, ckpt_path_val: str, last_lora_path_val: str) -> Iterator[str]:
                    """
                    Startet trainer_optimized.py direkt in Python und streamt Logs über den
                    gemeinsamen Runner (_run_finetune_command). Die Werte von max_steps und
                    num_workers werden aus den Slider-Eingaben übernommen.
                    """

                    # Initiale Log-Nachricht
                    yield "🟢 Finetuning script starting... please wait...⏳\n"
                    yield "🔄 loading model from huggingface...⏳\n"
                    
                    # Das Dropdown liefert nur den Dateinamen; erstelle vollständigen Pfad zum lora-JSON
                    lora_path = f"config/lora/{selected_lora_config}" if selected_lora_config else ""
                    ckpt_path = ckpt_path_val or "data/lora/r256_8bit"
                    cmd = [
                        sys.executable,
                        "scripts/train/trainer_optimized.py",
                        "--dataset_path", "data/data_sets/train_set",
                        "--lora_config_path", lora_path,
                        "--ckpt_path", ckpt_path,
                        "--learning_rate", "1e-4",
                        "--optimizer", "adamw8bit",
                        "--max_steps", str(max_steps_val),
                        "--batch_size", "1",
                        "--num_workers", str(num_workers_val),
                        "--save_every_n_train_steps", str(save_every_n_steps_val),
                        "--save_last", str(save_last_val),
                        "--precision", 'bf16-mixed',
                        "--accumulate_grad_batches", "2",
                        "--text_encoder_device", "cuda",
                        "--exp_name", "LoRa_r256_8bit",
                    ]

                    # Füge last_lora_path nur hinzu, wenn es nicht leer ist
                    if last_lora_path_val and last_lora_path_val.strip():
                        cmd.extend(["--last_lora_path", last_lora_path_val.strip()])

                    # Nutzt den generischen Runner für konsistentes Logging und Stop-Funktion
                    yield from _run_finetune_command(cmd)


                def on_stop_finetune():
                    """Stops a running finetuning process."""
                    global STOP_REQUESTED, RUN_PROC
                    if RUN_PROC is None or RUN_PROC.poll() is not None:
                        return  # no active process
                    STOP_REQUESTED = True
                    try:
                        _terminate_process(RUN_PROC)
                    except Exception as e:
                        LOGGER.warning(f"❌ Stop error: {e}")

                # Button-Klicks an Callback-Funktionen binden
                convert_btn.click(fn=on_convert, inputs=[], outputs=[finetune_log], show_progress=False)
                create_btn.click(fn=on_create, inputs=[], outputs=[finetune_log], show_progress=False)
                train_btn.click(fn=on_train, inputs=[max_steps_slider, num_workers_slider, save_every_n_steps_slider, save_last_slider, lora_config_dd, ckpt_path_input, last_lora_path_input], outputs=[finetune_log], show_progress=False)
                stop_ft_btn.click(fn=on_stop_finetune, inputs=[], outputs=[], show_progress=False)

    demo.launch()

if __name__ == "__main__":
    launch_ui()
