# Python 3.11+, safetensors 0.4.3
# Datei: inspect_true_rank.py
from safetensors.torch import load_file
import sys

def infer_r(shapeA, shapeB):
    # LoRA speichert üblicherweise 2D-Matrizen A und B.
    # Gängig: A: (out, r), B: (r, in)  ODER A: (r, in), B: (out, r)
    a0,a1 = shapeA
    b0,b1 = shapeB
    if a1 == b0: return a1
    if a0 == b1: return a0
    # Fallback: gemeinsame Dimension r suchen
    for dA in (a0,a1):
        if dA in (b0,b1): return dA
    return None

def main(path):
    sd = load_file(path, device="cpu")
    pairs, seen = {}, set()
    for k in sorted(sd.keys()):
        if "lora_A.weight" in k:
            base = k.replace("lora_A.weight", "")
            a = sd[k].shape
            b = sd.get(base + "lora_B.weight")
            if b is None: 
                print(f"[WARN] Kein B für {k}")
                continue
            r = infer_r(a, b.shape)
            print(f"{base}r={r} | A{a} B{b.shape} | dtype={sd[k].dtype}")
            seen.add(base)
    print(f"Total LoRA layers: {len(seen)}")

if __name__ == "__main__":
    main(sys.argv[1])
