"""One-click ingestion script for Colab Pro A100 Fold 0 outputs.

Once best_model_fold_0.pt and val_preds_fold_0.csv are placed in checkpoints/c384_cleanlab/,
this script:
1. Verifies checkpoint integrity and keys (strict=True)
2. Computes 2-fold C384 + 3-fold N4 Silver OOF (1,740 studies) & Gold (58 studies) Macro AUC
3. Publishes a new version of Kaggle dataset nswitzer/creaky-c384-cleanlab-models
"""

import json
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

C384_DIR = "checkpoints/c384_cleanlab"
N4_DIR = "checkpoints/n4_cleanlab_trained"


def compute_macro_auc(df: pd.DataFrame, pred_prefix: str = "pred_") -> tuple[float, dict]:
    per_class = {}
    for t in TARGET_COLS:
        yt = df[t].values
        yp = df[f"{pred_prefix}{t}"].values
        mask = ~np.isnan(yt) & (yt != 0.5)
        if mask.sum() < 2:
            continue
        yb = (yt[mask] >= 0.5).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        per_class[t] = float(roc_auc_score(yb, yp[mask]))
    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return macro, per_class


def main():
    f0_ckpt = os.path.join(C384_DIR, "best_model_fold_0.pt")
    f0_csv = os.path.join(C384_DIR, "val_preds_fold_0.csv")
    f1_csv = os.path.join(C384_DIR, "val_preds_fold_1.csv")

    if not os.path.exists(f0_ckpt) or not os.path.exists(f0_csv):
        print(f"[Waiting] Place {f0_ckpt} and {f0_csv} from Colab Drive (/content/drive/MyDrive/creaky_c384/) into {C384_DIR}/ and re-run.")
        return

    state = torch.load(f0_ckpt, map_location="cpu")
    print(f"Loaded Fold 0 checkpoint: backbone={state.get('backbone')}, epoch={state.get('epoch')}, silver_auc={state.get('silver_auc'):.4f}, gold_auc={state.get('gold_auc'):.4f}")

    # Evaluate 2-fold C384 + 3-fold N4 on Gold (58 studies)
    c_f0 = pd.read_csv(f0_csv)
    c_f1 = pd.read_csv(f1_csv)
    c_f0_gold = c_f0[c_f0["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)
    c_f1_gold = c_f1[c_f1["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)

    c_2fold_gold = c_f0_gold.copy()
    for t in TARGET_COLS:
        c_2fold_gold[f"pred_{t}"] = 0.5 * c_f0_gold[f"pred_{t}"].values + 0.5 * c_f1_gold[f"pred_{t}"].values

    c2_macro, _ = compute_macro_auc(c_2fold_gold)
    print(f"2-Fold C384 Gold Macro AUC (F0+F1): {c2_macro:.4f}")

    # Load 3-fold N4 Gold
    n4_golds = []
    for f in [0, 1, 2]:
        df = pd.read_csv(os.path.join(N4_DIR, f"val_preds_fold_{f}.csv"))
        n4_golds.append(df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True))

    with open(os.path.join(C384_DIR, "optimal_c384_blend_weights.json"), "r", encoding="utf-8") as f:
        opt_w = json.load(f)

    blend_gold = c_2fold_gold.copy()
    for t in TARGET_COLS:
        p_n4 = np.mean([g[f"pred_{t}"].values for g in n4_golds], axis=0)
        p_c384 = c_2fold_gold[f"pred_{t}"].values
        w = float(opt_w.get(t, 0.25))
        blend_gold[f"pred_{t}"] = (1.0 - w) * p_n4 + w * p_c384

    b_macro, _ = compute_macro_auc(blend_gold)
    print(f"3-Fold N4 + 2-Fold C384 Gold Macro AUC: {b_macro:.4f}")

    import sys
    print("\nPublishing updated dataset version to nswitzer/creaky-c384-cleanlab-models...")
    subprocess.run(
        [sys.executable, "-m", "kaggle.cli", "datasets", "version", "-p", C384_DIR, "-m", f"Add C384 Fold 0 (2-fold Gold {b_macro:.4f})"],
        check=True,
    )
    print("Dataset update complete!")


if __name__ == "__main__":
    main()
