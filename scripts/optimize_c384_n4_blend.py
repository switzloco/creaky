"""Finds optimal per-target rank-blend weights between Track N4 (256px ConvNeXt) and E23 C384 (384px CoAtNet).

Uses Silver OOF (~870 studies, Fold 1) to fit per-target blend weights (via grid search in 0.05 increments)
and evaluates the out-of-sample generalization on the 58 Gold benchmark studies (plus 3-fold N4 + C384 Gold).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def calc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0.5) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt = (y_true[mask] >= 0.5).astype(int)
    yp = y_pred[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, yp))


def main():
    n4_dir = Path("checkpoints/n4_cleanlab_trained")
    c384_dir = Path("checkpoints/c384_cleanlab")

    f0_n4 = pd.read_csv(n4_dir / "val_preds_fold_0.csv").set_index("StudyInstanceUID")
    f1_n4 = pd.read_csv(n4_dir / "val_preds_fold_1.csv").set_index("StudyInstanceUID")
    f2_n4 = pd.read_csv(n4_dir / "val_preds_fold_2.csv").set_index("StudyInstanceUID")
    f1_c384 = pd.read_csv(c384_dir / "val_preds_fold_1.csv").set_index("StudyInstanceUID")

    common_f1 = f1_n4.index.intersection(f1_c384.index)
    f1_n4 = f1_n4.loc[common_f1]
    f1_c384 = f1_c384.loc[common_f1]

    silver_mask = ~f1_n4["is_gold"].astype(bool)
    gold_mask = f1_n4["is_gold"].astype(bool)

    print("=" * 95)
    print("  TARGET-ADAPTIVE RANK BLEND OPTIMIZATION: N4 (256px ConvNeXt) + C384 (384px CoAtNet)")
    print("=" * 95)

    optimal_weights = {}
    rows = []

    for t in TARGETS:
        pcol = f"pred_{t}"
        # Silver OOF
        ys = f1_n4.loc[silver_mask, t].to_numpy(dtype=np.float32)
        rn_s = rankdata(f1_n4.loc[silver_mask, pcol].to_numpy()) / silver_mask.sum()
        rc_s = rankdata(f1_c384.loc[silver_mask, pcol].to_numpy()) / silver_mask.sum()

        auc_n4_s = calc_auc(ys, rn_s)
        auc_c384_s = calc_auc(ys, rc_s)

        best_w = 0.0
        best_auc_s = auc_n4_s
        for w in np.linspace(0.0, 0.60, 25):
            blend_s = (1.0 - w) * rn_s + w * rc_s
            a = calc_auc(ys, blend_s)
            if a > best_auc_s:
                best_auc_s = a
                best_w = float(round(w, 3))

        optimal_weights[t] = best_w

        # Evaluate on Gold (3-Fold N4 ensemble vs 3-Fold N4 + C384 Fold 1)
        g_idx = f1_n4.index[gold_mask]
        yg = f1_n4.loc[g_idx, t].to_numpy(dtype=np.float32)

        r_n4_3f = (
            rankdata(f0_n4.loc[g_idx, pcol].to_numpy())
            + rankdata(f1_n4.loc[g_idx, pcol].to_numpy())
            + rankdata(f2_n4.loc[g_idx, pcol].to_numpy())
        ) / (3.0 * len(g_idx))
        r_c384_g = rankdata(f1_c384.loc[g_idx, pcol].to_numpy()) / len(g_idx)

        auc_n4_3f_g = calc_auc(yg, r_n4_3f)
        auc_c384_g = calc_auc(yg, r_c384_g)
        auc_adapt_g = calc_auc(yg, (1.0 - best_w) * r_n4_3f + best_w * r_c384_g)
        auc_flat25_g = calc_auc(yg, 0.75 * r_n4_3f + 0.25 * r_c384_g)

        rows.append({
            "Target": t,
            "Opt C384 Wt": best_w,
            "Silver N4": auc_n4_s,
            "Silver OptBlend": best_auc_s,
            "Silver Lift": best_auc_s - auc_n4_s,
            "Gold N4-3F": auc_n4_3f_g,
            "Gold C384": auc_c384_g,
            "Gold Flat25": auc_flat25_g,
            "Gold OptBlend": auc_adapt_g,
            "Gold Lift": auc_adapt_g - auc_n4_3f_g,
        })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 95)
    print(f"Silver OOF Macro AUC | N4 F1 Alone       : {df_res['Silver N4'].mean():.4f}")
    print(f"Silver OOF Macro AUC | Adaptive Blend    : {df_res['Silver OptBlend'].mean():.4f}  ({df_res['Silver OptBlend'].mean() - df_res['Silver N4'].mean():+.4f})")
    print("-" * 95)
    print(f"Gold 58 Macro AUC    | N4 3-Fold Alone   : {df_res['Gold N4-3F'].mean():.4f}")
    print(f"Gold 58 Macro AUC    | 75/25 Flat Blend  : {df_res['Gold Flat25'].mean():.4f}  ({df_res['Gold Flat25'].mean() - df_res['Gold N4-3F'].mean():+.4f})")
    print(f"Gold 58 Macro AUC    | Adaptive Blend    : {df_res['Gold OptBlend'].mean():.4f}  ({df_res['Gold OptBlend'].mean() - df_res['Gold N4-3F'].mean():+.4f})")
    print("=" * 95)

    out_path = c384_dir / "optimal_c384_blend_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(optimal_weights, f, indent=2)
    print(f"Saved optimal C384 blend weights to {out_path}")


if __name__ == "__main__":
    main()
