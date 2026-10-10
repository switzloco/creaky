import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("\n".join("".join(nb["cells"][2]["source"]).splitlines()[:35]))
