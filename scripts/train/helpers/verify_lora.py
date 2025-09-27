import sys, math
import os
from pathlib import Path
import torch
from safetensors.torch import load_file

# Projektwurzel für Imports sicherstellen
def find_project_root(start: Path, markers=("setup.py", "README.md", ".git")) -> Path:
    """Gehe von `start` nach oben und suche nach einer Datei/Ordner, der das Projekt markiert.
    Falls nichts gefunden wird, versuche einen vernünftigen Fallback (vier Ebenen hoch).
    """
    p = start
    for _ in range(16):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    # Fallback: projektebene 3 Ebenen höher (scripts -> train -> helpers -> ..)
    parents = Path(__file__).resolve().parents
    if len(parents) > 3:
        return parents[3]
    return Path(__file__).resolve().parents[-1]

ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.helpers.shared_logs import log_message

# Relative Pfade innerhalb des Projekts (nur Projekt-internen Pfad angeben)
# Beispiel: "data/lora/train_3/Sido_lora3.safetensors"
old_rel = Path("data/lora/train_7/pytorch_lora_weights.safetensors")
new_rel = Path("data/lora/train_20/pytorch_lora_weights.safetensors")

# Vollständige Pfade relativ zur Projektwurzel
old_path = (ROOT / old_rel).resolve()
new_path = (ROOT / new_rel).resolve()

# Existenz prüfen und im Live-Log melden
if not old_path.exists():
    log_message(f"❌ Datei nicht gefunden: {old_path}")
    raise FileNotFoundError(f"Old LoRA file not found: {old_path}")
if not new_path.exists():
    log_message(f"❌ Datei nicht gefunden: {new_path}")
    raise FileNotFoundError(f"New LoRA file not found: {new_path}")

old_sd = load_file(str(old_path))
new_sd = load_file(str(new_path))

# Gemeinsame Keys
common = sorted(set(old_sd.keys()) & set(new_sd.keys()))
print(f"Gemeinsame Layer: {len(common)} / old={len(old_sd)} / new={len(new_sd)}")

# Differenz-Statistik
diff_norm = 0.0
old_norm = 0.0
new_norm = 0.0
cos_sims = []

for k in common:
    a = old_sd[k].to(torch.float32).flatten()
    b = new_sd[k].to(torch.float32).flatten()
    da = torch.linalg.norm(a)
    db = torch.linalg.norm(b)
    dn = torch.linalg.norm(b - a)
    diff_norm += dn.item() ** 2
    old_norm  += da.item() ** 2
    new_norm  += db.item() ** 2
    # Kosinus-Ähnlichkeit (nur wenn Normen > 0)
    if da.item() > 0 and db.item() > 0:
        cos = torch.dot(a, b) / (da * db)
        cos_sims.append(cos.item())

diff_norm = math.sqrt(diff_norm)
old_norm  = math.sqrt(old_norm)
new_norm  = math.sqrt(new_norm)

print(f"||new - old||_2 = {diff_norm:.4f}")
print(f"||old||_2       = {old_norm:.4f}")
print(f"||new||_2       = {new_norm:.4f}")
if cos_sims:
    import statistics as st
    print(f"cosine similarity: mean={st.mean(cos_sims):.4f}, median={st.median(cos_sims):.4f}")

# Heuristik:
# - Resume: hohe cosine similarity (≈ 0.8–0.99) und ||new|| nahe ||old||, diff moderat
# - Fresh:  cosine ~ 0 (teils sogar negativ), ||new|| von ||old|| deutlich verschieden