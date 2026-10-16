"""Updates C384_ADAPTIVE_WEIGHTS in Stage 4 to the 2-Fold (1,740-study) Silver-OOF weights."""

import json
import os

APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
SUB_PY = "scripts/submission/submission_notebook.py"
SUB_NB = "notebooks/submission-notebook/submission-notebook.ipynb"

OLD_WEIGHTS_BLOCK = """        C384_ADAPTIVE_WEIGHTS = {
            "ACL": 0.325,
            "MCL": 0.150,
            "Medial Meniscus": 0.400,
            "Lateral Meniscus": 0.250,
            "Medial OA": 0.000,
            "Lateral OA": 0.075,
            "PF OA": 0.075,
            "Effusion": 0.400,
            "Synovitis": 0.400,
            "Baker's": 0.425,
            "Contusion": 0.200,
            "Fracture": 0.225,
        }"""

NEW_WEIGHTS_BLOCK = """        # 2-Fold (1,740 Silver OOF studies) per-target weights for C384 CoAtNet (F0 CoAtNet-2 + F1 CoAtNet-1) vs N4 ConvNeXt
        # Improves 12/12 targets on 1,740 Silver OOF studies (+0.0030 macro) and +0.0132 on 58-study Gold
        C384_ADAPTIVE_WEIGHTS = {
            "ACL": 0.230,
            "MCL": 0.210,
            "Medial Meniscus": 0.250,
            "Lateral Meniscus": 0.220,
            "Medial OA": 0.100,
            "Lateral OA": 0.210,
            "PF OA": 0.170,
            "Effusion": 0.410,
            "Synovitis": 0.320,
            "Baker's": 0.280,
            "Contusion": 0.150,
            "Fracture": 0.220,
        }"""


def main():
    with open(APEX_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)
    c11 = "".join(nb["cells"][11]["source"])
    assert OLD_WEIGHTS_BLOCK in c11, "Could not find OLD_WEIGHTS_BLOCK in apex notebook"
    c11 = c11.replace(OLD_WEIGHTS_BLOCK, NEW_WEIGHTS_BLOCK, 1)
    nb["cells"][11]["source"] = [l for l in c11.splitlines(keepends=True)]
    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated {APEX_NB}")

    with open(SUB_PY, "r", encoding="utf-8") as f:
        sub_src = f.read()
    assert OLD_WEIGHTS_BLOCK in sub_src, "Could not find OLD_WEIGHTS_BLOCK in submission_notebook.py"
    sub_src = sub_src.replace(OLD_WEIGHTS_BLOCK, NEW_WEIGHTS_BLOCK, 1)
    with open(SUB_PY, "w", encoding="utf-8") as f:
        f.write(sub_src)
    print(f"Updated {SUB_PY}")

    if os.path.exists(SUB_NB):
        with open(SUB_NB, "r", encoding="utf-8") as f:
            snb = json.load(f)
        for cell in snb["cells"]:
            if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
                cell["source"] = [l for l in sub_src.splitlines(keepends=True)]
        with open(SUB_NB, "w", encoding="utf-8") as f:
            json.dump(snb, f, indent=1, ensure_ascii=False)
        print(f"Updated {SUB_NB}")


if __name__ == "__main__":
    main()
