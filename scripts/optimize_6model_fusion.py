"""Computes and exports OOF-Anchored Per-Checkpoint Z-Logit Calibration constants and optimal blend weights
for all 6 models (3x N4 256px ConvNeXt-Small + 3x C384 384px CoAtNet).
"""

import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
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


def to_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-5, 1.0 - 1e-5))


def main():
    c_dfs = [pd.read_csv(f"{C384_DIR}/val_preds_fold_{f}.csv") for f in [0, 1, 2]]
    n_dfs = [pd.read_csv(f"{N4_DIR}/val_preds_fold_{f}.csv") for f in [0, 1, 2]]

    c_silvers = [df[df["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True) for df in c_dfs]
    n_silvers = [df[df["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True) for df in n_dfs]

    c_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in c_dfs]
    n_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in n_dfs]

    stats = {"n4": {}, "c384": {}}
    for f in [0, 1, 2]:
        stats["n4"][str(f)] = {}
        stats["c384"][str(f)] = {}
        for t in TARGET_COLS:
            ln = to_logit(n_silvers[f][f"pred_{t}"].values)
            lc = to_logit(c_silvers[f][f"pred_{t}"].values)
            stats["n4"][str(f)][t] = [round(float(ln.mean()), 5), round(float(ln.std() + 1e-6), 5)]
            stats["c384"][str(f)][t] = [round(float(lc.mean()), 5), round(float(lc.std() + 1e-6), 5)]

    c_silvers_cal, n_silvers_cal = [], []
    for f in [0, 1, 2]:
        cc = c_silvers[f].copy()
        nc = n_silvers[f].copy()
        for t in TARGET_COLS:
            mu_n, sd_n = stats["n4"][str(f)][t]
            mu_c, sd_c = stats["c384"][str(f)][t]
            nc[f"pred_{t}"] = (to_logit(nc[f"pred_{t}"].values) - mu_n) / sd_n
            cc[f"pred_{t}"] = (to_logit(cc[f"pred_{t}"].values) - mu_c) / sd_c
        c_silvers_cal.append(cc)
        n_silvers_cal.append(nc)

    c_oof_cal = pd.concat(c_silvers_cal, ignore_index=True)
    n_oof_cal = pd.concat(n_silvers_cal, ignore_index=True)

    opt_w = {}
    oof_aucs = []
    n_oof_aucs = []
    c_oof_aucs = []
    for t in TARGET_COLS:
        yt = n_oof_cal[t].values
        zn = n_oof_cal[f"pred_{t}"].values
        zc = c_oof_cal[f"pred_{t}"].values
        n_oof_aucs.append(score_target(yt, zn))
        c_oof_aucs.append(score_target(yt, zc))
        best_w, best_a = 0.25, -1.0
        for w in np.linspace(0.10, 0.50, 41):
            a = score_target(yt, (1.0 - w) * zn + w * zc)
            if a > best_a:
                best_a = a
                best_w = float(round(w, 3))
        opt_w[t] = best_w
        oof_aucs.append(best_a)

    print("==========================================================================")
    print(f"Z-LOGIT 3-FOLD SILVER OOF (2,610 Studies): N4={np.mean(n_oof_aucs):.4f} | C384={np.mean(c_oof_aucs):.4f} | Blend={np.mean(oof_aucs):.4f}")
    print("==========================================================================")

    gold_aucs = []
    for i, t in enumerate(TARGET_COLS):
        yt = n_golds[0][t].values
        zn_folds = [(to_logit(n_golds[f][f"pred_{t}"].values) - stats["n4"][str(f)][t][0]) / stats["n4"][str(f)][t][1] for f in [0, 1, 2]]
        zc_folds = [(to_logit(c_golds[f][f"pred_{t}"].values) - stats["c384"][str(f)][t][0]) / stats["c384"][str(f)][t][1] for f in [0, 1, 2]]
        zn_avg = np.mean(zn_folds, axis=0)
        zc_avg = np.mean(zc_folds, axis=0)
        w = opt_w[t]
        pb = expit((1.0 - w) * zn_avg + w * zc_avg)
        g_auc = score_target(yt, pb)
        if not np.isnan(g_auc):
            gold_aucs.append(g_auc)
        print(f"  {t:18s}: OOF {n_oof_aucs[i]:.4f} -> {oof_aucs[i]:.4f} ({oof_aucs[i]-n_oof_aucs[i]:+.4f}) | Gold={g_auc:.4f} | w_c384={w:.3f}")

    print(f"\nGold 58-Study Macro AUC (OOF-Anchored Z-Logit): {np.mean(gold_aucs):.4f}")

    out_payload = {
        "weights": opt_w,
        "zlogit_stats": stats,
    }
    with open(f"{C384_DIR}/optimal_c384_zlogit_calibration.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Saved calibration payload to {C384_DIR}/optimal_c384_zlogit_calibration.json")


if __name__ == "__main__":
    main()
