import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

def evaluate_fold0(csv_path="checkpoints/n4_cleanlab_trained/val_preds_fold_0.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist yet.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded validation predictions: {len(df)} studies.")
    
    is_gold = df.get("is_gold", pd.Series([False]*len(df)))
    gold_df = df[is_gold == True] if is_gold.any() else df
    
    print("\n" + "="*70)
    print("  TRACK N4 (SMART-PLANE & WIDE-SLICING) FOLD 0 VALIDATION RESULTS")
    print("="*70)
    
    results = []
    for t in TARGETS:
        pred_col = f"pred_{t}"
        if pred_col not in df.columns or t not in df.columns:
            continue
            
        full_auc = roc_auc_score(df[t], df[pred_col]) if len(np.unique(df[t])) > 1 else np.nan
        gold_auc = roc_auc_score(gold_df[t], gold_df[pred_col]) if len(np.unique(gold_df[t])) > 1 else np.nan
        brier = brier_score_loss(gold_df[t], gold_df[pred_col])
        pos_count = int(gold_df[t].sum())
        total_count = len(gold_df)
        
        results.append({
            "Target": t,
            "Full Val AUC": full_auc,
            "Gold AUC": gold_auc,
            "Gold Brier": brier,
            "Gold Pos/Total": f"{pos_count}/{total_count}"
        })
        
    res_df = pd.DataFrame(results)
    
    print(res_df.to_string(index=False))
    print("-"*70)
    mean_full_auc = res_df["Full Val AUC"].mean()
    mean_gold_auc = res_df["Gold AUC"].mean()
    print(f"Mean Full Validation AUC : {mean_full_auc:.4f}")
    print(f"Mean Gold Validation AUC : {mean_gold_auc:.4f}")
    
    lm_row = res_df[res_df["Target"] == "Lateral Meniscus"]
    if not lm_row.empty:
        lm_gold_auc = lm_row["Gold AUC"].values[0]
        print(f"\n>>> LATERAL MENISCUS GOLD AUC: {lm_gold_auc:.4f} <<<")
        print("Previous Baseline: 0.6923  |  Cleanlab E18: 0.7240")
        diff = lm_gold_auc - 0.6923
        print(f"Lift over baseline: {'+' if diff >= 0 else ''}{diff:.4f}")
    print("="*70)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/n4_cleanlab_trained/val_preds_fold_0.csv"
    evaluate_fold0(path)
