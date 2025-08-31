from safetensors.torch import load_file
import os
import re

# ✅ Pfad zu deiner .safetensors Datei:
file_path = r"Z:\AI\projects\music\ace-lora\loras\chinesrap\pytorch_lora_weights.safetensors"

# Datei laden
state_dict = load_file(file_path)

# LoRA Layer extrahieren
lora_layers = [k for k in state_dict.keys() if ".lora_A.weight" in k or ".lora_B.weight" in k]
layer_names = sorted(set(re.sub(r"\.lora_[AB]\.weight", "", k) for k in lora_layers))

# Analyse
print(f"\n🔍 LoRA-Analyse für: {os.path.basename(file_path)}\n")
print(f"📦 Gefundene Layer: {len(layer_names)}\n")

total_params = 0
total_size_mb = 0.0

for layer in layer_names:
    A_key = f"{layer}.lora_A.weight"
    B_key = f"{layer}.lora_B.weight"
    if A_key in state_dict and B_key in state_dict:
        A = state_dict[A_key]
        B = state_dict[B_key]
        r, dim_in = B.shape
        dim_out, _ = A.shape
        params = A.numel() + B.numel()
        size_mb = params * 2 / (1024 ** 2)  # 2 Bytes für bf16
        total_params += params
        total_size_mb += size_mb
        print(f"• {layer:60s} | r={r:<3d} | {dim_out}×{dim_in} | ~{size_mb:.2f} MB")

print(f"\n🧮 Gesamt: {total_params:,} Parameter ≈ {total_size_mb:.2f} MB\n")