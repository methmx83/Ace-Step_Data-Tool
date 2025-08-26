"""
Minimal Gradio WebUI to demo preset selection and tag processing.

Note: This is a lightweight demo — no LLM calls. It lets you input raw tags (comma-separated) and
process them through the existing TagProcessor and TagPipeline using a selected preset (moods.md).
"""
from typing import List
import logging
import sys
import subprocess
from pathlib import Path

# Ensure project root is on sys.path so 'scripts.*' imports work when running this file directly
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    import gradio as gr
except Exception:
    gr = None

from scripts.helpers.preset_loader import list_presets, resolve_preset_path
from scripts.helpers.tag_processor import TagProcessor
from scripts.tagging.tag_pipeline import TagPipeline

logger = logging.getLogger(__name__)


def process_demo(preset: str, enable_rap_style: bool, raw_tags_text: str, run_script: bool = False, input_dir: str = "data/audio") -> dict:
    # Resolve preset path
    presets_available = list_presets()
    preset_path = resolve_preset_path(preset) or "presets/moods.md"
    tp = TagProcessor(moods_file_path=preset_path, allow_extras=False)
    # optional: if rap_style disabled, empty allowed set
    if not enable_rap_style:
        tp.allowed_tags.rap_style = set()
    # parse raw tags input (comma separated)
    raw = [t.strip() for t in raw_tags_text.split(',') if t.strip()]

    # Use TagProcessor.process_tags which runs normalization, dedupe and conflict resolution
    final = tp.process_tags(raw)
    stats = tp.get_tag_statistics(final)
    result = {
        "debug": {
            "preset_selected": preset,
            "resolved_preset_path": preset_path,
            "presets_available": presets_available,
        },
        "preset_path": preset_path,
        "input_raw_count": len(raw),
        "final_tags": final,
        "stats": stats,
    }

    # If no raw tags were provided, assume user wants to run the real pipeline
    if not run_script and len(raw) == 0:
        run_script = True

    if run_script:
        # Run the actual multi_tagger CLI in a subprocess using the same interpreter
        # Pass the selected moods file so the CLI uses the chosen preset
        cmd = [sys.executable, "-m", "scripts.tagging.multi_tagger", "--input_dir", input_dir, "--verbose", "--moods_file", preset_path]
        try:
            # Run from project root so python -m scripts.tagging... finds the package
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(root))
            result["cli_returncode"] = proc.returncode
            result["cli_stdout"] = proc.stdout
            result["cli_stderr"] = proc.stderr
        except Exception as e:
            result["cli_exception"] = str(e)

    return result


def launch_gradio():
    if gr is None:
        print("Gradio is not installed. Install it with: pip install gradio")
        return

    presets = list_presets()
    if not presets:
        presets = ["default"]

    with gr.Blocks(title="ACE-DATA_v2 Tagging Demo") as demo:
        gr.Markdown("# ACE-DATA_v2 — Tagging Demo")
        with gr.Row():
            preset_dropdown = gr.Dropdown(choices=presets, label="Preset (moods.md)", value=presets[0])
            rap_checkbox = gr.Checkbox(value=True, label="Enable rap_style")
            run_cli = gr.Checkbox(value=False, label="Run multi_tagger CLI")
            input_dir_in = gr.Textbox(value="data/audio", label="input_dir (for CLI)")
        raw_tags = gr.Textbox(lines=3, label="Raw tags (comma separated)")
        out_json = gr.JSON(label="Result")
        cli_output = gr.Textbox(lines=10, label="CLI Output (stdout / stderr)")
        run_btn = gr.Button("Process")

        def _run(preset, rap, tags, run_cli_flag, input_dir_val):
            try:
                res = process_demo(preset, rap, tags, run_script=run_cli_flag, input_dir=input_dir_val)
                cli_text = ""
                if run_cli_flag:
                    stdout = res.get("cli_stdout") or ""
                    stderr = res.get("cli_stderr") or ""
                    cli_text = stdout + ("\n--- STDERR ---\n" + stderr if stderr else "")
                return res, cli_text
            except Exception as e:
                logger.exception("Processing failed")
                return {"error": str(e)}, str(e)

        run_btn.click(_run, inputs=[preset_dropdown, rap_checkbox, raw_tags, run_cli, input_dir_in], outputs=[out_json, cli_output])

    demo.launch()


if __name__ == "__main__":
    launch_gradio()
