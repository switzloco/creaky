import json

with open("knee_mri_patient_explainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell 6 ===")
print("".join(nb["cells"][6]["source"]))
print("=== Cell 14 ===")
print("".join(nb["cells"][14]["source"]))
