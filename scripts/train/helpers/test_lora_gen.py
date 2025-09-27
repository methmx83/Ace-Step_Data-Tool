# test_lora_generation.py
# 🔊 Mini-Skript zum Testen deines LoRA-Adapters in ACE-STEP
import json
import torch
from peft import LoraConfig
from scripts.train.acestep.pipeline_ace_step import ACEStepPipeline
from safetensors.torch import load_file
import soundfile as sf
import os

# === USER KONFIGURATION ===
CHECKPOINT_DIR = r"C:\Users\methm\.cache\ace-step\checkpoints"
LORA_PATH = os.path.join(CHECKPOINT_DIR, "data/lora/train_9", "pytorch_lora9_weights.safetensors")
LORA_CONFIG_PATH = "config/lora/Ace-LoRa.json"
PROMPT = "Ein düsterer Boom-bap Hip-Hop Beat deepen Rap Lyrics auf dem Beat"
OUTPUT_PATH = "data/tests/test_new3.wav"
BPM = 83
DURATION = 20  # Sekunden (optional)

# Stelle sicher, dass der Ausgabeordner existiert
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === PIPELINE LADEN ===
print("🔄 Lade ACE-STEP Pipeline ...")
pipeline = ACEStepPipeline(CHECKPOINT_DIR)
pipeline.load_checkpoint(CHECKPOINT_DIR)

# === LoRA CONFIG UND ADAPTER LADEN ===
print("🎚️  Lade LoRA Adapter Konfiguration ...")

# Konfiguration aus JSON laden
with open(LORA_CONFIG_PATH, "r") as f:
    config_dict = json.load(f)

config = LoraConfig(**config_dict)

# Adapter hinzufügen
print("➕ Füge LoRA-Adapter zum Transformer hinzu ...")
pipeline.ace_step_transformer.add_adapter(adapter_config=config, adapter_name="sido")

# Gewichte laden
print("📥 Lade LoRA-Gewichte aus:", LORA_PATH)
lora_weights = load_file(LORA_PATH)

# 🔍 DEBUG: Zeige die tatsächlichen Keys der LoRA-Gewichte an
print("🔍 Gefundene LoRA-Gewichts-Keys:")
for key in lora_weights.keys():
    print(f"    {key}")

# Mögliche Anpassung der Keys – je nachdem, wie sie gespeichert wurden
# Beispiel: Wenn die Keys mit 'base_model.model.ace_step_transformer.' anfangen, musst du kürzen
# Oder wenn sie einfach nur 'query.weight' heißen, musst du den Pfad ergänzen

# Annahme: Die Gewichte sind nur mit z. B. 'transformer.layers.0.self_attn.q_proj.weight' gespeichert
# Wir versuchen, sie unter 'ace_step_transformer.' einzuhängen
adapted_weights = {}
for key, value in lora_weights.items():
    # Option 1: Wenn die Keys bereits LoRA-Form haben, z. B. `query.lora_A.weight`
    # und du nur den Modulnamen voranstellen musst
    new_key = f"ace_step_transformer.{key}"
    adapted_weights[new_key] = value

print(f"🔧 Setze {len(adapted_weights)} LoRA-Gewichte ein (strict=False) ...")
pipeline.ace_step_transformer.load_state_dict(adapted_weights, strict=False)

# Adapter aktivieren
pipeline.ace_step_transformer.set_adapter("sido")
pipeline.ace_step_transformer.eval()

print("Pipeline __call__ Signatur:", pipeline.__call__.__code__.co_varnames)

# === GENERIEREN ===
print(f"🚀 Generiere Musik mit Prompt: '{PROMPT}'")
with torch.no_grad():
    out = pipeline(
        prompt=PROMPT,
        lyrics= "Erschreck dich nicht, Sido ist schrecklich frisch, Ich hab dir schon auf dem ersten Tape gesagt, dass du zu hässlich bist."  # Wichtig: nicht None!
    )

# === SPEICHERN / KOPIEREN ===
print(f"\n🔍 Pipeline hat Audio bereits gespeichert!")
gen_audio_path = out[0]  # Das ist der Pfad zur WAV-Datei
print(f"📁 Generierte Datei: {gen_audio_path}")

if os.path.exists(gen_audio_path):
    print(f"💾 Kopiere nach: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import shutil
    shutil.copy(gen_audio_path, OUTPUT_PATH)
    print("✅ Fertig! Audio erfolgreich kopiert.")
else:
    raise FileNotFoundError(f"Generierte Audio-Datei nicht gefunden: {gen_audio_path}")

