import json

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][14]["source"])
for line in src.splitlines():
    if "patient_raw" in line or "patient_annotated" in line or "load_" in line or "extract_" in line:
        print("Cell 14:", line)
