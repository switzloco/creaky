import pandas as pd
import numpy as np
import os
import argparse
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def load_oof_predictions(csv_paths):
    """Load OOF predictions from a list of CSV paths.
    Assume each CSV contains StudyInstanceUID, TARGET_COLS, and {target}_true.
    Returns: list of DataFrames (one per model), aligned by StudyInstanceUID.
    """
    dfs = [pd.read_csv(p).set_index("StudyInstanceUID") for p in csv_paths]
    
    # Keep only studies present in all models
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
        
    dfs = [df.loc[common_idx] for df in dfs]
    return dfs

def macro_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
        except ValueError:
            pass
    return np.mean(aucs) if aucs else 0.0

def objective(weights, preds_list, y_true):
    """Minimize negative macro AUC."""
    # Normalize weights
    weights = np.array(weights)
    weights /= (np.sum(weights) + 1e-6)
    
    # Blend predictions
    blended = np.zeros_like(preds_list[0])
    for w, preds in zip(weights, preds_list):
        blended += w * preds
        
    return -macro_auc(y_true, blended)

def main():
    parser = argparse.ArgumentParser(description="Fit ensemble blending weights using Nelder-Mead.")
    parser.add_argument("csv_paths", nargs="+", help="Paths to OOF prediction CSV files")
    args = parser.parse_args()
    
    print(f"Loading OOF predictions from {len(args.csv_paths)} files...")
    dfs = load_oof_predictions(args.csv_paths)
    print(f"Found {len(dfs[0])} common studies.")
    
    # Extract truth and predictions
    # Use the first df for truth
    y_true_cols = [f"{t}_true" for t in TARGET_COLS]
    y_true = dfs[0][y_true_cols].values
    
    # Binarize targets at 0.5 (competition metric)
    # Mask out exactly 0.5
    mask = y_true != 0.5
    
    # We need to compute metric considering the mask.
    # To keep it simple, we'll flatten everything where mask is True?
    # No, ROC AUC must be computed per-column.
    
    preds_list = [df[TARGET_COLS].values for df in dfs]
    
    def masked_macro_auc(y_t, y_p):
        aucs = []
        for i in range(y_t.shape[1]):
            col_mask = y_t[:, i] != 0.5
            if not np.any(col_mask):
                continue
            yt_clean = (y_t[col_mask, i] >= 0.5).astype(int)
            yp_clean = y_p[col_mask, i]
            if len(np.unique(yt_clean)) > 1:
                aucs.append(roc_auc_score(yt_clean, yp_clean))
        return np.mean(aucs) if aucs else 0.0

    def masked_objective(weights):
        w = np.array(weights)
        w /= (np.sum(w) + 1e-6)
        blended = np.zeros_like(preds_list[0])
        for weight, preds in zip(w, preds_list):
            blended += weight * preds
        return -masked_macro_auc(y_true, blended)

    # Initial guess: equal weights
    init_weights = [1.0 / len(preds_list)] * len(preds_list)
    
    print(f"\nInitial baseline (equal weights) AUC: {-masked_objective(init_weights):.4f}")
    
    if len(preds_list) > 1:
        print("\nOptimizing weights (Nelder-Mead)...")
        bounds = [(0, 1) for _ in preds_list]
        res = minimize(masked_objective, init_weights, method='Nelder-Mead', bounds=bounds)
        
        best_weights = res.x / np.sum(res.x)
        print(f"\nOptimal weights:")
        for path, w in zip(args.csv_paths, best_weights):
            print(f"  {os.path.basename(path)}: {w:.4f}")
        print(f"Optimized AUC: {-res.fun:.4f}")
    else:
        print("Only one model provided. Skipping optimization.")

if __name__ == "__main__":
    main()
