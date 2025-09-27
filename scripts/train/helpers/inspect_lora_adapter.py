# inspect_lora_adapter.py
# Stand: 2025-07 (getestet PyTorch >= 2.1, safetensors >= 0.4)
import argparse
import re
from collections import Counter
from pathlib import Path

from safetensors import safe_open

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, default=r"Z:\AI\projects\music\DATA_TOOL\data\lora\train_20\pytorch_lora_weights.safetensors", help="Pfad zu .safetensors")
    ap.add_argument("--expected", default="LoRa_r256_8bit", help="Erwarteter Adapter-Name (optional)")
    ap.add_argument("--samples", type=int, default=10, help="Anzahl Beispiel-Keys")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(p)

    adapters = Counter()
    lora_keys = []
    meta_adapter = None

    # 1) Metadata lesen (falls vorhanden)
    with safe_open(str(p), framework="pt", device="cpu") as f:
        md = f.metadata()
        if isinstance(md, dict):
            meta_adapter = md.get("adapter_name", None)
        keys = list(f.keys())

    # 2) Adapter-Name aus Keys extrahieren
    # Formate:
    #   A) ...linear_q.<adapter>.lora_A.weight
    #   B) ...linear_q.lora_A.weight              (ohne Adapter)
    pat_with = re.compile(r"\.([^.]+)\.(lora_[AB]\.weight)$")
    pat_noad = re.compile(r"\.(lora_[AB]\.weight)$")

    for k in keys:
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            lora_keys.append(k)
            m = pat_with.search(k)
            if m:
                adapters[m.group(1)] += 1
            else:
                # Kein Adapter-Token gefunden
                if pat_noad.search(k):
                    adapters["<none>"] += 1
                else:
                    adapters["<unknown>"] += 1

    total = sum(adapters.values())
    print(f"[INFO] Datei: {p.name}")
    print(f"[INFO] LoRA-Tensoren gefunden: {total} (Keys mit lora_A/B)")
    if meta_adapter is not None:
        print(f"[INFO] Metadata adapter_name: {meta_adapter}")
    else:
        print(f"[INFO] Metadata adapter_name: <none>")

    if not lora_keys:
        print("[WARN] Keine LoRA-Gewichte gefunden (keine *.lora_A/B.weight Keys).")
        return

    print("\n[STATS] Adapter-Verteilung:")
    for name, cnt in adapters.most_common():
        print(f"  - {name}: {cnt}")

    # Beispiel-Keys
    print("\n[EXAMPLES]")
    for k in lora_keys[: min(len(lora_keys), args.samples)]:
        print(" ", k)

    # 3) Validierung gegen expected (falls gegeben)
    if args.expected:
        expected = args.expected
        ok = (len(adapters) == 1) and (expected in adapters) and ("<unknown>" not in adapters)
        # Sonderfall: Keys ohne Adapter-Name
        if expected not in adapters and "<none>" in adapters and len(adapters) == 1:
            print("\n[NOTE] Alle Keys haben keinen Adapter-Namen (.lora_A/B direkt).")
            print("       Das ist technisch gültig, aber 'expected' kann so nicht verifiziert werden.")
            ok = False

        print(f"\n[CHECK] expected='{expected}'  ->  {'OK' if ok else 'MISMATCH'}")
        # Optionaler Exit-Code für CI/Batch
        import sys
        sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
