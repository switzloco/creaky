import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def inspect_nb(path):
    print(f"=== {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb.get('cells', []))}")
    for i, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        first_line = src.strip().split("\n")[0] if src.strip() else "(empty)"
        print(f"  [{i:02d}] {ctype:8s} | {first_line[:80]}")

if __name__ == "__main__":
    for p in ["knee_mri_patient_explainer.ipynb", "notebooks/03_discrepancy_audit.ipynb", "notebooks/04_cleanlab_audit.ipynb"]:
        try:
            inspect_nb(p)
        except Exception as e:
            print(f"Error {p}: {e}")
