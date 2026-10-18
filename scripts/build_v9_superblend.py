"""Builds Version 9 of `notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb`:
Combines the 0.9586 4-family public stack (Raptor-384 + Raptor-224 + R6 4-Fold Flip-TTA + 2.5D MIL)
with our 6-Model Cleanlab-Pruned Multi-Resolution Anatomical MoE (3x N4 256px ConvNeXt-Small +
3x C384 384px CoAtNet, 0.8676 Silver OOF / 0.8568 Gold AUC, 2x In-VRAM Zoom TTA + OOF Z-Logit Calibration).
"""

import json
import re

GOODPJW_NB = "checkpoints/ref_0958_goodpjw/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb"
APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
APEX_META = "notebooks/apex_cleanlab_blend/kernel-metadata.json"


def main():
    with open(GOODPJW_NB, "r", encoding="utf-8") as f:
        base_nb = json.load(f)

    with open(APEX_NB, "r", encoding="utf-8") as f:
        our_nb = json.load(f)

    # Extract our Stage 4 + Stage 5 code cell (Cell 11 in current creaky-apex-cleanlab-blend.ipynb)
    our_stage4_cell = None
    for cell in our_nb["cells"]:
        if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
            our_stage4_cell = cell
            break
    assert our_stage4_cell is not None, "Could not find Stage 4 inference cell in current notebook"

    # 1. Upgrade Part A3 (Cell 12 in base_nb) to use heliosli's V2 0.9586 per-finding weights:
    # W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.30}
    c12_src = "".join(base_nb["cells"][12]["source"])
    old_w = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
    new_w = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}"
    assert old_w in c12_src, "Expected W_RAPTOR_C in base_nb cell 12"
    c12_src = c12_src.replace(old_w, new_w, 1)
    compile(c12_src, "<part_a3>", "exec")
    base_nb["cells"][12]["source"] = [l for l in c12_src.splitlines(keepends=True)]

    # 2. Tune Part B time guard in Cell 18 from 8.2h to 7.8h so Part C (our 6-Model Cleanlab MoE) always has >=1.0h headroom
    c18_src = "".join(base_nb["cells"][18]["source"])
    c18_src = c18_src.replace("> 8.2 * 3600:", "> 7.8 * 3600:")
    compile(c18_src, "<part_b>", "exec")
    base_nb["cells"][18]["source"] = [l for l in c18_src.splitlines(keepends=True)]

    # 3. Calibrate Stage 5 OPTIMIZED_TARGET_WEIGHTS in our_stage4_cell for the 0.9586 4-family stack
    s4_src = "".join(our_stage4_cell["source"])
    old_opt_weights = re.compile(r"        # Micro-injection weights[\s\S]*?        OPTIMIZED_TARGET_WEIGHTS = \{[\s\S]*?        \}")
    new_opt_weights = """        # Micro-injection weights calibrated to 6-Model Z-Logit Gold Benchmark (0.8568 Gold Macro AUC / 0.8676 Silver OOF)
        # on top of the 0.9586 4-family stack (Raptor-384 + Raptor-224 + R6 Flip-TTA + 2.5D MIL):
        # Elite tier (>0.92 Gold AUC): Medial OA (0.9845), Baker's (0.9801), Fracture (0.9208) -> 0.040-0.045
        # Strong tier (0.85-0.90 Gold AUC): Contusion (0.8853), ACL (0.8848), MCL (0.8753), Effusion (0.8821 OOF, +0.0160 lift) -> 0.035
        # Mid/Conservative tier: Medial Meniscus (0.030), Lateral OA (0.025), PF OA (0.020), Lateral Meniscus (0.015), Synovitis (0.015)
        OPTIMIZED_TARGET_WEIGHTS = {
            "Medial OA": 0.0450,
            "Baker's": 0.0450,
            "Fracture": 0.0400,
            "Contusion": 0.0350,
            "ACL": 0.0350,
            "MCL": 0.0350,
            "Effusion": 0.0350,
            "Medial Meniscus": 0.0300,
            "Lateral OA": 0.0250,
            "PF OA": 0.0200,
            "Lateral Meniscus": 0.0150,
            "Synovitis": 0.0150,
        }"""
    s4_src, n_rep = old_opt_weights.subn(new_opt_weights, s4_src, count=1)
    assert n_rep == 1, f"Expected 1 replacement of OPTIMIZED_TARGET_WEIGHTS, got {n_rep}"

    # Add 8.5h time guard inside the try block right after backing up public_submission_backup.csv
    old_try_header = """import shutil as _sh
_sh.copy('submission.csv', 'public_submission_backup.csv')
try:"""
    new_try_header = """import shutil as _sh, time as _c_time
_sh.copy('submission.csv', 'public_submission_backup.csv')
try:
    if _c_time.time() - globals().get('NB_T0', _c_time.time()) > 8.5 * 3600:
        raise RuntimeError('Time budget (>8.5h) reached before Part C; keeping 0.9586 stack submission')"""
    assert old_try_header in s4_src, "Expected old_try_header in Stage 4 cell"
    s4_src = s4_src.replace(old_try_header, new_try_header, 1)
    compile(s4_src, "<part_c_stage4_5>", "exec")

    part_c_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Part C · Creaky 6-Model Cleanlab-Pruned Multi-Resolution Anatomical MoE (`0.8676` Silver OOF / `0.8568` Gold AUC)\n",
            "\n",
            "Fuses our 6-model Cleanlab-pruned 3-plane Cross-Plane Transformer + Anatomical Gated MoE (`3x N4` 256×256 ConvNeXt-Small + `3x C384` 384×384 CoAtNet, with 2× In-VRAM Multi-Scale Zoom TTA and OOF-Anchored Per-Checkpoint Z-Logit Calibration) into the `0.9586` stack in percentile-rank space.\n"
        ],
    }
    part_c_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l for l in s4_src.splitlines(keepends=True)],
    }

    base_nb["cells"].append(part_c_md_cell)
    base_nb["cells"].append(part_c_code_cell)

    # Verify all code cells compile cleanly
    for idx, cell in enumerate(base_nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Skip %%writefile magic lines when syntax checking with compile()
            lines = [l for l in src.splitlines(keepends=True) if not l.startswith("%%writefile")]
            compile("".join(lines), f"<cell_{idx}>", "exec")

    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(base_nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully built {APEX_NB} with {len(base_nb['cells'])} cells!")

    # Update kernel-metadata.json with the 6 dataset sources
    meta = {
        "id": "nswitzer/creaky-apex-cleanlab-blend",
        "title": "Creaky Apex Cleanlab Blend",
        "code_file": "creaky-apex-cleanlab-blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "rostigor/rsna-knee-r6-fe055406",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "nswitzer/creaky-n4-cleanlab-models",
            "nswitzer/creaky-c384-cleanlab-models",
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
    }
    with open(APEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {APEX_META} with all 6 dataset sources!")


if __name__ == "__main__":
    main()
