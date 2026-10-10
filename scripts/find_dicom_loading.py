import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "dcmread" in src or "pydicom" in src or "dicom" in src.lower():
            print(f"Cell {i} has DICOM references:")
            lines = [l for l in src.splitlines() if "dcm" in l or "dicom" in l or "glob" in l or "path" in l.lower()]
            print("\n".join(lines[:15]))
            print("-" * 50)
