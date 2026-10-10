import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [2, 4, 6, 8, 10, 12, 14]:
    cell = nb["cells"][i]
    src = "".join(cell.get("source", []))
    print(f"=== Cell {i} ===")
    print(src[:400])
    print("...\n")
