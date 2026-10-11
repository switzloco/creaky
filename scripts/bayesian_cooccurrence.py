import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def build_cooccurrence_matrix(labels_df: pd.DataFrame) -> np.ndarray:
    """Calculates conditional probability matrix P(target_i = 1 | target_j = 1) [12 x 12]."""
    Y = labels_df[TARGET_COLS].values  # [N, 12]
    # Filter to binary rows (exclude 0.5 soft labels for precision)
    # Binary mask: row is valid if all values are 0 or 1, or treat >= 0.5 as 1
    Y_bin = (Y >= 0.5).astype(float)
    N, T = Y_bin.shape
    
    cond_prob = np.zeros((T, T), dtype=float)
    for j in range(T):
        pos_j = Y_bin[:, j] == 1.0
        n_pos_j = pos_j.sum()
        if n_pos_j > 0:
            for i in range(T):
                if i != j:
                    cond_prob[i, j] = (Y_bin[pos_j, i] == 1.0).sum() / n_pos_j
                else:
                    cond_prob[i, j] = 1.0
        else:
            cond_prob[:, j] = Y_bin.mean(axis=0)
            
    return cond_prob

def apply_bayesian_recalibration(probs: np.ndarray, cond_prob: np.ndarray, beta: float = 0.05) -> np.ndarray:
    """Calibrates predictions using anatomical co-occurrence evidence."""
    # Clip probs to avoid inf in logit
    p_clipped = np.clip(probs, 1e-4, 1.0 - 1e-4)
    raw_logits = logit(p_clipped)
    
    N, T = probs.shape
    # Compute cross-target evidence e_i = sum_{j != i} cond_prob[i, j] * p_j / (T - 1)
    # Mask out diagonal so target does not reinforce itself
    off_diag_cond = cond_prob.copy()
    np.fill_diagonal(off_diag_cond, 0.0)
    
    # Weight evidence by other targets' predicted probabilities
    evidence = np.dot(p_clipped, off_diag_cond.T) / (off_diag_cond.sum(axis=1) + 1e-6)
    evidence = np.clip(evidence, 1e-4, 1.0 - 1e-4)
    evidence_logits = logit(evidence)
    
    # Soft Bayesian updating
    calibrated_logits = raw_logits + beta * evidence_logits
    return expit(calibrated_logits)

def evaluate_calibration():
    # 1. Load train labels
    train_csv = "data/raw/train.csv"
    if not os.path.exists(train_csv):
        print(f"File {train_csv} not found.")
        return
        
    train_df = pd.read_csv(train_csv).dropna(subset=TARGET_COLS)
    print(f"Loaded {len(train_df)} training studies for co-occurrence mining.")
    
    cond_prob = build_cooccurrence_matrix(train_df)
    
    print("\n" + "="*80)
    print("KEY ANATOMICAL CO-OCCURRENCE PROBABILITIES P(Row=1 | Col=1):")
    print("="*80)
    co_df = pd.DataFrame(cond_prob, index=TARGET_COLS, columns=TARGET_COLS)
    print(co_df.round(3)[["ACL", "Medial OA", "Synovitis", "Contusion"]])
    
    # 2. Test calibration on Fold 0 validation predictions
    val_csv = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
    if not os.path.exists(val_csv):
        print(f"File {val_csv} not found.")
        return
        
    val_df = pd.read_csv(val_csv)
    gold_df = val_df[val_df["is_gold"] == True].copy()
    print(f"\nLoaded {len(gold_df)} Gold studies for evaluation.")
    
    raw_preds = gold_df[[f"pred_{t}" for t in TARGET_COLS]].values
    y_true = gold_df[TARGET_COLS].values
    
    raw_aucs = [roc_auc_score(y_true[:, i], raw_preds[:, i]) for i in range(len(TARGET_COLS))]
    raw_macro = np.mean(raw_aucs)
    print(f"\nBaseline Gold Macro AUC: {raw_macro:.4f}")
    
    best_beta = 0.0
    best_macro = raw_macro
    best_calibrated = raw_preds
    
    print("\nSweeping Bayesian calibration strength beta:")
    for beta in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        cal_preds = apply_bayesian_recalibration(raw_preds, cond_prob, beta=beta)
        aucs = [roc_auc_score(y_true[:, i], cal_preds[:, i]) for i in range(len(TARGET_COLS))]
        macro = np.mean(aucs)
        print(f"  beta = {beta:<4.2f} -> Gold Macro AUC: {macro:.4f} (Delta: {macro - raw_macro:+.4f})")
        if macro > best_macro:
            best_macro = macro
            best_beta = beta
            best_calibrated = cal_preds
            
    print(f"\n--> Optimal beta: {best_beta:.2f} | Best Gold Macro AUC: {best_macro:.4f} (+{best_macro - raw_macro:.4f})")
    
    # Save co-occurrence matrix for inference pipeline
    np.save("checkpoints/bayesian_cooccurrence_matrix.npy", cond_prob)
    print("Saved co-occurrence prior to checkpoints/bayesian_cooccurrence_matrix.npy")

if __name__ == "__main__":
    evaluate_calibration()
