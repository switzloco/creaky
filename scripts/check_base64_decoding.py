import json

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "base64" in src and "b64decode" in src:
            for line in src.splitlines():
                if "b64decode" in line or "CACHED" in line:
                    print("Found:", line[:100])
