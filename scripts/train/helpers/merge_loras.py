# merge_loras_safe.py
from safetensors.torch import load_file, save_file
import torch
import sys

def merge_loras_safe(lora_paths, alphas, output_path):
    assert len(lora_paths) == len(alphas), "Länge von LoRA-Dateien und Alphas muss übereinstimmen"
    state_dicts = [load_file(p) for p in lora_paths]

    merged = {}
    skipped = []

    all_keys = set().union(*[sd.keys() for sd in state_dicts])
    for key in sorted(all_keys):
        # Prüfe, ob Key in allen vorhanden ist
        tensors = []
        shapes = []
        for sd, alpha in zip(state_dicts, alphas):
            if key in sd:
                tensors.append(alpha * sd[key])
                shapes.append(sd[key].shape)
            else:
                tensors.append(None)
                shapes.append(None)

        # Nur mergen wenn alle vorhandenen Tensors die gleiche Shape haben
        shapes_set = {tuple(s.shape) for s in tensors if s is not None}
        if len(shapes_set) == 1:
            # Summe aller nicht-leeren Tensors
            merged[key] = sum(t for t in tensors if t is not None)
        else:
            skipped.append((key, shapes))

    save_file(merged, output_path)

    print(f"\n✅ Merge abgeschlossen: {output_path}")
    print(f"🔢 Gemergte Layer: {len(merged)}")
    print(f"⚠️  Übersprungene Layer wegen Shape-Unterschied: {len(skipped)}")
    for key, shapes in skipped:
        print(f"  ⛔ {key} → Shapes: {shapes}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Verwendung:")
        print("  python merge_loras_safe.py <lorafile1> <lorafile2> <alpha1> <alpha2> <output_path>")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]
    alpha1, alpha2 = float(sys.argv[3]), float(sys.argv[4])
    output_path = sys.argv[5]

    merge_loras_safe([path1, path2], [alpha1, alpha2], output_path)