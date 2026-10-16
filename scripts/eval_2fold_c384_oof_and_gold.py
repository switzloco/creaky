"""Evaluates 2-Fold C384 (Fold 0 CoAtNet-2 + Fold 1 CoAtNet-1) + 3-Fold N4 across:
1. Fold 0 Silver OOF (870 studies)
2. Combined 2-Fold Silver OOF (1,740 studies: Fold 0 + Fold 1)
3. Out-of-Sample 58-Study Gold Benchmark
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

C384_DIR = "checkpoints/c384_cleanlab"
N4_DIR = "checkpoints/n4_cleanlab_trained"


def score_target(yt: np.ndarray, yp: np.ndarray) -> float:
    mask = ~np.isnan(yt) & (yt != 0.5)
    if mask.sum() < 2:
        return float("nan")
    yb = (yt[mask] >= 0.5).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return float(roc_auc_score(yb, yp[mask]))


def compute_macro_auc(df: pd.DataFrame, pred_prefix: str = "pred_") -> tuple[float, dict]:
    per_class = {}
    for t in TARGET_COLS:
        auc = score_target(df[t].values, df[f"{pred_prefix}{t}"].values)
        if not np.isnan(auc):
            per_class[t] = auc
    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return macro, per_class


def main():
    c_f0 = pd.read_csv(os.path.join(C384_DIR, "val_preds_fold_0.csv"))
    c_f1 = pd.read_csv(os.path.join(C384_DIR, "val_preds_fold_1.csv"))
    n_f0 = pd.read_csv(os.path.join(N4_DIR, "val_preds_fold_0.csv"))
    n_f1 = pd.read_csv(os.path.join(N4_DIR, "val_preds_fold_1.csv"))
    n_f2 = pd.read_csv(os.path.join(N4_DIR, "val_preds_fold_2.csv"))

    # 1. Fold 0 Silver OOF (870 studies)
    c0_s = c_f0[c_f0["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
    n0_s = n_f0[n_f0["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
    assert (c0_s["StudyInstanceUID"].values == n0_s["StudyInstanceUID"].values).all()

    n0_macro, n0_pc = compute_macro_auc(n0_s)
    c0_macro, c0_pc = compute_macro_auc(c0_s)

    b0_75_25 = n0_s.copy()
    for t in TARGET_COLS:
        b0_75_25[f"pred_{t}"] = 0.75 * n0_s[f"pred_{t}"].values + 0.25 * c0_s[f"pred_{t}"].values
    b0_macro, b0_pc = compute_macro_auc(b0_75_25)

    print("=================================================================")
    print("1. FOLD 0 SILVER OOF (870 Held-Out Studies)")
    print("=================================================================")
    print(f"  N4 Fold 0 (256px ConvNeXt-S)      : {n0_macro:.4f}")
    print(f"  C384 Fold 0 (384px CoAtNet-2)     : {c0_macro:.4f}")
    print(f"  75/25 Blend (N4 F0 + C384 F0)     : {b0_macro:.4f} ({b0_macro - n0_macro:+.4f})")

    # 2. Combined 2-Fold Silver OOF (1,740 studies: Fold 0 + Fold 1)
    c1_s = c_f1[c_f1["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
    n1_s = n_f1[n_f1["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
    assert (c1_s["StudyInstanceUID"].values == n1_s["StudyInstanceUID"].values).all()

    c_oof = pd.concat([c0_s, c1_s], ignore_index=True)
    n_oof = pd.concat([n0_s, n1_s], ignore_index=True)

    n_oof_macro, n_oof_pc = compute_macro_auc(n_oof)
    c_oof_macro, c_oof_pc = compute_macro_auc(c_oof)

    b_oof_75_25 = n_oof.copy()
    for t in TARGET_COLS:
        b_oof_75_25[f"pred_{t}"] = 0.75 * n_oof[f"pred_{t}"].values + 0.25 * c_oof[f"pred_{t}"].values
    b_oof_75_macro, _ = compute_macro_auc(b_oof_75_25)

    # Fit optimal per-pathology C384 weight on 1,740 Silver OOF studies (constrained in [0.10, 0.50] to prevent overfitting)
    opt_w_2fold = {}
    b_oof_opt = n_oof.copy()
    for t in TARGET_COLS:
        yt = n_oof[t].values
        pn = n_oof[f"pred_{t}"].values
        pc = c_oof[f"pred_{t}"].values
        best_w, best_a = 0.25, -1.0
        for w in np.linspace(0.10, 0.50, 41):
            a = score_target(yt, (1.0 - w) * pn + w * pc)
            if a > best_a:
                best_a = a
                best_w = float(round(w, 3))
        opt_w_2fold[t] = best_w
        b_oof_opt[f"pred_{t}"] = (1.0 - best_w) * pn + best_w * pc

    b_oof_opt_macro, b_oof_opt_pc = compute_macro_auc(b_oof_opt)

    print("\n=================================================================")
    print("2. COMBINED 2-FOLD SILVER OOF (1,740 Studies: Fold 0 + Fold 1)")
    print("=================================================================")
    print(f"  N4 2-Fold OOF (256px ConvNeXt-S)  : {n_oof_macro:.4f}")
    print(f"  C384 2-Fold OOF (384px CoAtNet)   : {c_oof_macro:.4f}")
    print(f"  75/25 Blend OOF                   : {b_oof_75_macro:.4f} ({b_oof_75_macro - n_oof_macro:+.4f})")
    print(f"  Adaptive Per-Pathology Blend OOF  : {b_oof_opt_macro:.4f} ({b_oof_opt_macro - n_oof_macro:+.4f})")
    print("\n  Per-Pathology Silver OOF Breakdown (1,740 studies):")
    for t in TARGET_COLS:
        print(f"    {t:18s}: N4={n_oof_pc[t]:.4f} | C384={c_oof_pc[t]:.4f} | Blend={b_oof_opt_pc[t]:.4f} ({b_oof_opt_pc[t] - n_oof_pc[t]:+.4f}) [w_c384={opt_w_2fold[t]:.2f}]")

    # 3. Out-of-Sample 58-Study Gold Benchmark
    c0_g = c_f0[c_f0["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)
    c1_g = c_f1[c_f1["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)
    n0_g = n_f0[n_f0["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)
    n1_g = n_f1[n_f1["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)
    n2_g = n_f2[n_f2["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True)

    n3_gold = n0_g.copy()
    c2_gold = c0_g.copy()
    for t in TARGET_COLS:
        n3_gold[f"pred_{t}"] = (n0_g[f"pred_{t}"].values + n1_g[f"pred_{t}"].values + n2_g[f"pred_{t}"].values) / 3.0
        c2_gold[f"pred_{t}"] = 0.5 * c0_g[f"pred_{t}"].values + 0.5 * c1_g[f"pred_{t}"].values

    n3_g_macro, n3_g_pc = compute_macro_auc(n3_gold)
    c0_g_macro, _ = compute_macro_auc(c0_g)
    c1_g_macro, _ = compute_macro_auc(c1_g)
    c2_g_macro, c2_g_pc = compute_macro_auc(c2_gold)

    blend_gold_2f = n3_gold.copy()
    for t in TARGET_COLS:
        w = opt_w_2fold[t]
        blend_gold_2f[f"pred_{t}"] = (1.0 - w) * n3_gold[f"pred_{t}"].values + w * c2_gold[f"pred_{t}"].values
    bg_macro, bg_pc = compute_macro_auc(blend_gold_2f)

    # Also test rank-averaging F0 and F1 before blending (since F0 is CoAtNet-2 5ep and F1 is CoAtNet-1 3ep)
    c2_rank_gold = c0_g.copy()
    blend_rank_gold = n3_gold.copy()
    for t in TARGET_COLS:
        r0 = pd.Series(c0_g[f"pred_{t}"].values).rank(pct=True).values
        r1 = pd.Series(c1_g[f"pred_{t}"].values).rank(pct=True).values
        rn = pd.Series(n3_gold[f"pred_{t}"].values).rank(pct=True).values
        rc = 0.5 * r0 + 0.5 * r1
        c2_rank_gold[f"pred_{t}"] = rc
        w = opt_w_2fold[t]
        blend_rank_gold[f"pred_{t}"] = (1.0 - w) * rn + w * rc
    bg_rank_macro, _ = compute_macro_auc(blend_rank_gold)

    print("\n=================================================================")
    print("3. OUT-OF-SAMPLE 58-STUDY GOLD BENCHMARK")
    print("=================================================================")
    print(f"  3-Fold N4 Solo (256px ConvNeXt-S)           : {n3_g_macro:.4f}")
    print(f"  C384 Fold 0 Solo (384px CoAtNet-2, 5ep)     : {c0_g_macro:.4f}")
    print(f"  C384 Fold 1 Solo (384px CoAtNet-1, 3ep)     : {c1_g_macro:.4f}")
    print(f"  2-Fold C384 Solo (F0 + F1)                  : {c2_g_macro:.4f}")
    print(f"  3-Fold N4 + 2-Fold C384 (Prob Blend)        : {bg_macro:.4f} ({bg_macro - n3_g_macro:+.4f})")
    print(f"  3-Fold N4 + 2-Fold C384 (Rank Blend)        : {bg_rank_macro:.4f} ({bg_rank_macro - n3_g_macro:+.4f})")

    with open(os.path.join(C384_DIR, "optimal_c384_blend_weights.json"), "w", encoding="utf-8") as f:
        json.dump(opt_w_2fold, f, indent=2)
    print(f"\nSaved updated 2-fold Silver-fitted weights to {C384_DIR}/optimal_c384_blend_weights.json")


if __name__ == "__main__":
    main()
