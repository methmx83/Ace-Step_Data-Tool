from safetensors.torch import load_file

# Pfad zu deiner .ckpt Datei
ckpt_path = r"Z:\AI\projects\music\DATA_TOOL\data/lora/train_20/pytorch_lora_weights.safetensors"

# Lade Safetensors
state_dict = load_file(ckpt_path)

# Zeige alle Keys
print("Alle Keys im Safetensors:\n")
for k in state_dict.keys():
    print(k)

# Extrahiere LoRA-relevante Layer
print("\nExtrahierte LoRA-Gewichte:\n")
state_dict_filtered = {
    k: v
    for k, v in state_dict.items()
    if "lora" in k or ".adapter" in k
}
for k in state_dict_filtered.keys():
    print(k)