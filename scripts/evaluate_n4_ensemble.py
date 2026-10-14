import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

def evaluate_3fold_ensemble():
    base_dir = Path("checkpoints/n4_cleanlab_trained")
    f0 = pd.read_csv(base_dir / "val_preds_fold_0.csv")
    f1 = pd.read_csv(base_dir / "val_preds_fold_1.csv")
    f2 = pd.read_csv(base_dir / "val_preds_fold_2.csv")

    print(f"Loaded validation files: F0 ({len(f0)}), F1 ({len(f1)}), F2 ({len(f2)})")

    # Evaluate on Gold Studies (present in all folds as benchmark)
    g0 = f0[f0["is_gold"] == True].set_index("StudyInstanceUID")
    g1 = f1[f1["is_gold"] == True].set_index("StudyInstanceUID")
    g2 = f2[f2["is_gold"] == True].set_index("StudyInstanceUID")

    common_gold = g0.index.intersection(g1.index).intersection(g2.index)
    print(f"Common Gold Studies across all 3 folds: {len(common_gold)}")

    results = []
    for t in TARGETS:
        pcol = f"pred_{t}"
        y_true = g0.loc[common_gold, t].values
        mask = (y_true != 0.5)
        y_bin = (y_true[mask] >= 0.5).astype(int)

        # Individual fold Gold AUCs
        auc0 = roc_auc_score(y_bin, g0.loc[common_gold, pcol].values[mask])
        auc1 = roc_auc_score(y_bin, g1.loc[common_gold, pcol].values[mask])
        auc2 = roc_auc_score(y_bin, g2.loc[common_gold, pcol].values[mask])

        # 3-Fold Ensemble Average
        ens_pred = (g0.loc[common_gold, pcol].values + g1.loc[common_gold, pcol].values + g2.loc[common_gold, pcol].values) / 3.0
        auc_ens = roc_auc_score(y_bin, ens_pred[mask])

        results.append({
            "Target": t,
            "Fold 0": auc0,
            "Fold 1": auc1,
            "Fold 2": auc2,
            "3-Fold Ens": auc_ens,
            "Lift vs F0": auc_ens - auc0
        })

    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("  TRACK N4 (SMART-PLANE & WIDE-SLICING) 3-FOLD GOLD ENSEMBLE EVALUATION")
    print("="*80)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 80)
    print(f"Mean Gold AUC - Fold 0    : {df_res['Fold 0'].mean():.4f}")
    print(f"Mean Gold AUC - Fold 1    : {df_res['Fold 1'].mean():.4f}")
    print(f"Mean Gold AUC - Fold 2    : {df_res['Fold 2'].mean():.4f}")
    print(f"Mean Gold AUC - 3-FOLD ENS: {df_res['3-Fold Ens'].mean():.4f}  (OVERALL LIFT: {df_res['3-Fold Ens'].mean() - df_res['Fold 0'].mean():+.4f})")
    lm_row = df_res[df_res["Target"] == "Lateral Meniscus"]
    print(f"\n>>> LATERAL MENISCUS 3-FOLD ENSEMBLE GOLD AUC: {lm_row['3-Fold Ens'].values[0]:.4f} <<<")
    print("="*80)

if __name__ == "__main__":
    evaluate_3fold_ensemble()
