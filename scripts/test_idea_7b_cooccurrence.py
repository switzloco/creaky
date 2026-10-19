"""Tests Idea #7b: Anatomical Co-Occurrence & Bayesian / Log-Odds Post-Processing
strictly out-of-fold on 2,610 Silver OOF studies and out-of-sample on the 58 Gold studies.
"""

import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

C384_DIR = "checkpoints/c384_cleanlab"
N4_DIR = "checkpoints/n4_cleanlab_trained"

# Clinically established orthopedic co-occurrence graph (allowed predictors for each target)
ORTHOPEDIC_COUPLINGS = {
    "ACL": ["Contusion", "Effusion", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "MCL": ["ACL", "Contusion", "Effusion", "Fracture"],
    "Medial Meniscus": ["Medial OA", "ACL", "Effusion", "Baker's"],
    "Lateral Meniscus": ["Lateral OA", "ACL", "Contusion", "Effusion"],
    "Medial OA": ["Medial Meniscus", "PF OA", "Lateral OA", "Effusion", "Baker's"],
    "Lateral OA": ["Lateral Meniscus", "Medial OA", "PF OA", "Effusion"],
    "PF OA": ["Medial OA", "Lateral OA", "Effusion", "Synovitis"],
    "Effusion": ["Synovitis", "ACL", "Contusion", "Fracture", "Medial OA", "Baker's"],
    "Synovitis": ["Effusion", "Baker's", "PF OA", "Medial OA"],
    "Baker's": ["Effusion", "Medial Meniscus", "Medial OA", "Synovitis"],
    "Contusion": ["ACL", "Fracture", "MCL", "Effusion", "Lateral Meniscus"],
    "Fracture": ["Contusion", "Effusion", "MCL", "ACL"],
}


def score_target(yt: np.ndarray, yp: np.ndarray) -> float:
    mask = ~np.isnan(yt) & (yt != 0.5)
    if mask.sum() < 2:
        return float("nan")
    yb = (yt[mask] >= 0.5).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return float(roc_auc_score(yb, yp[mask]))


def to_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-5, 1.0 - 1e-5))


def load_calibrated_6model_data():
    with open(f"{C384_DIR}/optimal_c384_zlogit_calibration.json", "r", encoding="utf-8") as f:
        cal = json.load(f)
    opt_w = cal["weights"]
    stats = cal["zlogit_stats"]

    c_dfs = [pd.read_csv(f"{C384_DIR}/val_preds_fold_{f}.csv") for f in [0, 1, 2]]
    n_dfs = [pd.read_csv(f"{N4_DIR}/val_preds_fold_{f}.csv") for f in [0, 1, 2]]

    c_silvers = [df[df["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True) for df in c_dfs]
    n_silvers = [df[df["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True) for df in n_dfs]

    c_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in c_dfs]
    n_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in n_dfs]

    silver_folds_z = []
    silver_folds_y = []
    for f in [0, 1, 2]:
        Z = np.zeros((len(n_silvers[f]), 12), dtype=np.float64)
        Y = np.zeros((len(n_silvers[f]), 12), dtype=np.float64)
        for j, t in enumerate(TARGET_COLS):
            mu_n, sd_n = stats["n4"][str(f)][t]
            mu_c, sd_c = stats["c384"][str(f)][t]
            zn = (to_logit(n_silvers[f][f"pred_{t}"].values) - mu_n) / sd_n
            zc = (to_logit(c_silvers[f][f"pred_{t}"].values) - mu_c) / sd_c
            w = opt_w[t]
            Z[:, j] = (1.0 - w) * zn + w * zc
            Y[:, j] = n_silvers[f][t].values
        silver_folds_z.append(Z)
        silver_folds_y.append(Y)

    Z_gold = np.zeros((len(n_golds[0]), 12), dtype=np.float64)
    Y_gold = np.zeros((len(n_golds[0]), 12), dtype=np.float64)
    for j, t in enumerate(TARGET_COLS):
        zn_folds = [
            (to_logit(n_golds[f][f"pred_{t}"].values) - stats["n4"][str(f)][t][0]) / stats["n4"][str(f)][t][1]
            for f in [0, 1, 2]
        ]
        zc_folds = [
            (to_logit(c_golds[f][f"pred_{t}"].values) - stats["c384"][str(f)][t][0]) / stats["c384"][str(f)][t][1]
            for f in [0, 1, 2]
        ]
        w = opt_w[t]
        Z_gold[:, j] = (1.0 - w) * np.mean(zn_folds, axis=0) + w * np.mean(zc_folds, axis=0)
        Y_gold[:, j] = n_golds[0][t].values

    return silver_folds_z, silver_folds_y, Z_gold, Y_gold


def main():
    silver_folds_z, silver_folds_y, Z_gold, Y_gold = load_calibrated_6model_data()
    Z_silver = np.vstack(silver_folds_z)
    Y_silver = np.vstack(silver_folds_y)
    fold_ids = np.concatenate([np.full(len(silver_folds_z[f]), f) for f in [0, 1, 2]])

    # 1. Baseline scores
    base_silver_aucs = [score_target(Y_silver[:, j], Z_silver[:, j]) for j in range(12)]
    base_gold_aucs = [score_target(Y_gold[:, j], Z_gold[:, j]) for j in range(12)]
    print("================================================================================")
    print(f"BASELINE 6-MODEL Z-LOGIT | Silver OOF: {np.mean(base_silver_aucs):.5f} | Gold 58: {np.nanmean(base_gold_aucs):.5f}")
    print("================================================================================")

    # 2. Method A: Empirical Partial Log-Odds Co-Occurrence Matrix (Zero-Shot from Label Co-occurrence / Prediction Correlation)
    # Wait: in inference on a submission.csv, we have Rank-Space or Z-Logit predictions across 12 columns.
    # Let's test Cross-Validated Logistic / Ridge Stacking on the 12 Z-Logits (using ONLY clinically coupled neighbors!)
    for mode in ["all_neighbors", "orthopedic_graph"]:
        for alpha in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]:
            Z_silver_smoothed = np.copy(Z_silver)
            Z_gold_smoothed = np.copy(Z_gold)

            for j, t in enumerate(TARGET_COLS):
                if mode == "all_neighbors":
                    nbr_idx = [k for k in range(12) if k != j]
                else:
                    nbr_idx = [TARGET_COLS.index(nb) for nb in ORTHOPEDIC_COUPLINGS[t]]

                # Out-of-fold prediction for Silver
                oof_prior = np.zeros(len(Z_silver))
                for f in [0, 1, 2]:
                    tr_mask = (fold_ids != f) & (~np.isnan(Y_silver[:, j])) & (Y_silver[:, j] != 0.5)
                    va_mask = (fold_ids == f)
                    y_tr = (Y_silver[tr_mask, j] >= 0.5).astype(int)
                    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500)
                    clf.fit(Z_silver[tr_mask][:, nbr_idx], y_tr)
                    p_va = clf.predict_proba(Z_silver[va_mask][:, nbr_idx])[:, 1]
                    z_va = to_logit(p_va)
                    z_va = (z_va - z_va.mean()) / (z_va.std() + 1e-6)
                    oof_prior[va_mask] = z_va

                # Fit on all Silver to predict on Gold (100% out-of-sample for Gold!)
                full_mask = (~np.isnan(Y_silver[:, j])) & (Y_silver[:, j] != 0.5)
                y_full = (Y_silver[full_mask, j] >= 0.5).astype(int)
                clf_full = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500)
                clf_full.fit(Z_silver[full_mask][:, nbr_idx], y_full)
                p_gold = clf_full.predict_proba(Z_gold[:, nbr_idx])[:, 1]
                z_gold_prior = to_logit(p_gold)
                z_gold_prior = (z_gold_prior - z_gold_prior.mean()) / (z_gold_prior.std() + 1e-6)

                Z_silver_smoothed[:, j] = (1.0 - alpha) * Z_silver[:, j] + alpha * oof_prior
                Z_gold_smoothed[:, j] = (1.0 - alpha) * Z_gold[:, j] + alpha * z_gold_prior

            s_aucs = [score_target(Y_silver[:, j], Z_silver_smoothed[:, j]) for j in range(12)]
            g_aucs = [score_target(Y_gold[:, j], Z_gold_smoothed[:, j]) for j in range(12)]
            print(
                f"Mode={mode:16s} | alpha={alpha:.2f} | "
                f"Silver OOF: {np.mean(s_aucs):.5f} ({np.mean(s_aucs)-np.mean(base_silver_aucs):+.5f}) | "
                f"Gold 58: {np.nanmean(g_aucs):.5f} ({np.nanmean(g_aucs)-np.nanmean(base_gold_aucs):+.5f})"
            )

    # 3. Inspect per-target gains at alpha=0.08 (orthopedic_graph)
    print("\n--- Per-Target Breakdown (orthopedic_graph, alpha=0.08) ---")
    alpha = 0.08
    s_Selective = []
    g_Selective = []
    for j, t in enumerate(TARGET_COLS):
        nbr_idx = [TARGET_COLS.index(nb) for nb in ORTHOPEDIC_COUPLINGS[t]]
        oof_prior = np.zeros(len(Z_silver))
        for f in [0, 1, 2]:
            tr_mask = (fold_ids != f) & (~np.isnan(Y_silver[:, j])) & (Y_silver[:, j] != 0.5)
            va_mask = (fold_ids == f)
            y_tr = (Y_silver[tr_mask, j] >= 0.5).astype(int)
            clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=500)
            clf.fit(Z_silver[tr_mask][:, nbr_idx], y_tr)
            z_va = to_logit(clf.predict_proba(Z_silver[va_mask][:, nbr_idx])[:, 1])
            oof_prior[va_mask] = (z_va - z_va.mean()) / (z_va.std() + 1e-6)

        full_mask = (~np.isnan(Y_silver[:, j])) & (Y_silver[:, j] != 0.5)
        clf_full = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=500)
        clf_full.fit(Z_silver[full_mask][:, nbr_idx], (Y_silver[full_mask, j] >= 0.5).astype(int))
        z_gp = to_logit(clf_full.predict_proba(Z_gold[:, nbr_idx])[:, 1])
        z_gp = (z_gp - z_gp.mean()) / (z_gp.std() + 1e-6)

        s_new = score_target(Y_silver[:, j], (1.0 - alpha) * Z_silver[:, j] + alpha * oof_prior)
        g_new = score_target(Y_gold[:, j], (1.0 - alpha) * Z_gold[:, j] + alpha * z_gp)
        ds = s_new - base_silver_aucs[j]
        dg = g_new - base_gold_aucs[j]
        print(
            f"  {t:18s} | Silver: {base_silver_aucs[j]:.4f} -> {s_new:.4f} ({ds:+.4f}) | "
            f"Gold: {base_gold_aucs[j]:.4f} -> {g_new:.4f} ({dg:+.4f}) | coefs={np.round(clf_full.coef_[0], 3)}"
        )


if __name__ == "__main__":
    main()
