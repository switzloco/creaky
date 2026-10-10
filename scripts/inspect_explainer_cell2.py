import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell 2 ===")
print("".join(nb["cells"][2]["source"]))
print("=== Cell 4 (first 10 lines) ===")
print("\n".join("".join(nb["cells"][4]["source"]).splitlines()[:20]))
