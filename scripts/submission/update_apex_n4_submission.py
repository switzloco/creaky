import json
import re
from pathlib import Path

def update_submission_notebook():
    nb_path = Path("notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # 1. Update Cell 1 (Imports & Constants & Checkpoint Filter)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "CHECKPOINT_FILTER" in "".join(cell["source"]):
            code = "".join(cell["source"])
            # Update Checkpoint filter to point to our newly uploaded N4 dataset
            code = re.sub(
                r'CHECKPOINT_FILTER:\s*List\[str\]\s*=\s*\[.*?\]',
                'CHECKPOINT_FILTER: List[str] = ["creaky-n4-cleanlab-models", "best_model_fold_0.pt", "best_model_fold_1.pt", "best_model_fold_2.pt"]',
                code
            )
            # Update Wide Slicing (2% to 98%)
            code = code.replace("start_idx = int(n * 0.12)", "start_idx = int(n * 0.02)")
            code = code.replace("int(n * 0.88)", "int(n * 0.98)")

            # Update series selection for fluid-sensitivity priority
            old_select = (
                '    if pipeline == "cache_v1":\n'
                '        match = study_series[values.str.capitalize() == plane]\n'
                '        if not match.empty:\n'
                '            return str(match.iloc[-1]["SeriesInstanceUID"])'
            )
            new_select = (
                '    if pipeline == "cache_v1":\n'
                '        match = study_series[values.str.capitalize() == plane]\n'
                '        if not match.empty:\n'
                '            if "Fluid_Sensitive" in match.columns or "Fat_Suppression" in match.columns:\n'
                '                fluid = match.get("Fluid_Sensitive", pd.Series(0, index=match.index)).fillna(0).astype(int)\n'
                '                fat = match.get("Fat_Suppression", pd.Series(0, index=match.index)).fillna(0).astype(int)\n'
                '                match = match.assign(_prio=(fluid * 2) + fat).sort_values("_prio", ascending=False)\n'
                '            return str(match.iloc[0]["SeriesInstanceUID"])'
            )
            if old_select in code:
                code = code.replace(old_select, new_select)
                print("Injected fluid-sensitive priority selection into select_series_uid!")

            cell["source"] = [line + "\n" for line in code.split("\n")[:-1]] + ([code.split("\n")[-1]] if code.split("\n")[-1] else [])
            print("Updated Cell 1 (Checkpoint filter, wide slicing, and sequence selection).")
            break

    # 2. Update Cell 11 (Class-Specific Ensembling Weights)
    for cell in nb["cells"]:
        code = "".join(cell["source"])
        if "OPTIMIZED_TARGET_WEIGHTS" in code:
            # Conservative synergistic weights anchored on 384px CoAtNet (0.949)
            # with targeted N4 boosts on Lateral Meniscus (0.25) and MCL (0.20)
            target_weights_block = (
                '        # Conservative Class-Specific Rank Blend Anchored on CoAtNet 0.949 Base\n'
                '        # Track N4 3-Fold Ensemble (ConvNeXt MoE + Fluid-Sensitive + 2%-98% Wide Slicing)\n'
                '        OPTIMIZED_TARGET_WEIGHTS = {\n'
                '            "ACL": 0.1500,\n'
                '            "MCL": 0.2000,              # Record Fold 1 Gold AUC: 0.9138\n'
                '            "Medial Meniscus": 0.1500,\n'
                '            "Lateral Meniscus": 0.2500,  # Solves peripheral clipping (+0.0286 ensemble lift)\n'
                '            "Medial OA": 0.1500,\n'
                '            "Lateral OA": 0.1500,\n'
                '            "PF OA": 0.1500,\n'
                '            "Effusion": 0.1200,\n'
                '            "Synovitis": 0.1000,\n'
                '            "Baker\'s": 0.1500,\n'
                '            "Contusion": 0.1500,\n'
                '            "Fracture": 0.1500,\n'
                '        }'
            )
            code = re.sub(
                r'# Target-Adaptive Class-Specific Rank Blend.*?\n\s*OPTIMIZED_TARGET_WEIGHTS\s*=\s*\{.*?\}',
                target_weights_block,
                code,
                flags=re.DOTALL
            )
            cell["source"] = [line + "\n" for line in code.split("\n")[:-1]] + ([code.split("\n")[-1]] if code.split("\n")[-1] else [])
            print("Updated Cell 11 with calibrated anchor-conservative N4 blend weights.")
            break

    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Saved updated notebook to {nb_path}")

    # 3. Update kernel-metadata.json
    meta_path = Path("notebooks/apex_cleanlab_blend/kernel-metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["dataset_sources"] = [
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "nswitzer/creaky-n4-cleanlab-models"
    ]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved updated kernel metadata to {meta_path}")

if __name__ == "__main__":
    update_submission_notebook()
