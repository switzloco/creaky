import json

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "def load_study_slice" in src:
            for line in src.splitlines():
                if "def load_study_slice" in line:
                    idx = src.splitlines().index(line)
                    print("\n".join(src.splitlines()[idx:idx+40]))
                    break
