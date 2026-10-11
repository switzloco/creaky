import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def optimize_weights_between_models(p1_csv: str, p2_csv: str, name1: str = "Model 1", name2: str = "Model 2", gold_only: bool = True):
    split_name = "GOLD ONLY (N=58)" if gold_only else "ALL DEFINITIVE (Gold + Silver binary)"
    print("="*95)
    print(f"CLASS-SPECIFIC BLEND OPTIMIZATION: [{name1}] vs [{name2}] on {split_name}")
    print("="*95)
    
    df1 = pd.read_csv(p1_csv)
    df2 = pd.read_csv(p2_csv)
    
    common = pd.merge(df1, df2, on="StudyInstanceUID", suffixes=("_m1", "_m2"))
    if gold_only:
        common = common[common["is_gold_m1"] == True].copy()
    
    print(f"Aligned {len(common)} evaluation studies.")
    
    optimal_weights = {}
    improvements = []
    
    print(f"\n{'Target':<18} | {name1+' AUC':<12} | {name2+' AUC':<12} | {'Optimal w('+name1+')':<18} | {'Blended AUC':<12} | {'Delta':<8}")
    print("-" * 95)
    
    for t in TARGET_COLS:
        col_true = f"{t}_m1"
        subset = common[common[col_true].isin([0.0, 1.0])].copy()
        
        y_true = subset[col_true].values.astype(int)
        p1 = subset[f"pred_{t}_m1"].values
        p2 = subset[f"pred_{t}_m2"].values
        
        # Check if both classes present
        if len(np.unique(y_true)) < 2:
            continue
            
        auc1 = roc_auc_score(y_true, p1)
        auc2 = roc_auc_score(y_true, p2)
        
        # Rank-normalized versions
        r1 = pd.Series(p1).rank(pct=True).values
        r2 = pd.Series(p2).rank(pct=True).values
        
        def obj(w):
            blended = w * r1 + (1.0 - w) * r2
            return -roc_auc_score(y_true, blended)
        
        res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
        opt_w = float(res.x)
        blended_opt = opt_w * r1 + (1.0 - opt_w) * r2
        best_auc = roc_auc_score(y_true, blended_opt)
        
        baseline_best = max(auc1, auc2)
        delta = best_auc - baseline_best
        optimal_weights[t] = round(opt_w, 4)
        improvements.append((auc1, auc2, best_auc))
        
        print(f"{t:<18} | {auc1:<12.4f} | {auc2:<12.4f} | {opt_w:<18.4f} | {best_auc:<12.4f} | +{delta:<.4f}")
        
    m1_mean = np.mean([x[0] for x in improvements])
    m2_mean = np.mean([x[1] for x in improvements])
    blend_mean = np.mean([x[2] for x in improvements])
    
    print("-" * 95)
    print(f"{'MACRO AUC':<18} | {m1_mean:<12.4f} | {m2_mean:<12.4f} | {'Target-Adaptive':<18} | {blend_mean:<12.4f} | +{blend_mean - max(m1_mean, m2_mean):<.4f}")
    print("=" * 95)
    
    print("\nOPTIMIZED_TARGET_WEIGHTS = {")
    for k, v in optimal_weights.items():
        print(f'    "{k}": {v},')
    print("}")
    return optimal_weights

if __name__ == "__main__":
    p1 = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
    p2 = "checkpoints/e04_trained/val_preds_fold_0.csv"
    if os.path.exists(p1) and os.path.exists(p2):
        print("\n--- 1. GOLD STUDY EVALUATION ---")
        optimize_weights_between_models(p1, p2, name1="E18_Cleanlab", name2="E04_ConvNeXt", gold_only=True)
        print("\n--- 2. ALL DEFINITIVE EVALUATION ---")
        optimize_weights_between_models(p1, p2, name1="E18_Cleanlab", name2="E04_ConvNeXt", gold_only=False)
