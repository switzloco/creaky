"""Label Validator: Evaluates extracted labels against the 58 expert gold-standard labels.

Calculates:
- Per-target Accuracy
- Sensitivity (Recall) & Specificity
- F1 Score
- Non-empty extraction coverage (%)
- Macro-averaged AUC ROC
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

# Ensure scripts/label_mining is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from regex_labeler import RegexLabeler, TARGET_COLS



def evaluate_labels(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    """Evaluate predicted labels against true gold-standard labels."""
    # Ensure aligned on StudyInstanceUID
    merged = pd.merge(true_df, pred_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    
    results = {}
    auc_scores = []

    print("=" * 80)
    print(f"{'Finding':<20} | {'Acc':<6} | {'Sens':<6} | {'Spec':<6} | {'F1':<6} | {'Cover':<6} | {'AUC':<6}")
    print("-" * 80)

    for col in TARGET_COLS:
        y_true = merged[f"{col}_true"].values.astype(float)
        y_pred_raw = merged[f"{col}_pred"].values
        
        # Calculate coverage (how many were not UNK/NaN)
        valid_mask = ~pd.isna(y_pred_raw)
        coverage = valid_mask.mean() * 100.0

        # Fill UNK with 0.5 (uncertainty baseline) for AUC computation
        y_pred_filled = np.where(valid_mask, y_pred_raw.astype(float), 0.5)

        # For discrete metrics, default UNK to 0 (negative) or check only valid
        y_pred_binary = np.where(valid_mask, (y_pred_raw.astype(float) >= 0.5).astype(int), 0)

        acc = accuracy_score(y_true, y_pred_binary)
        
        # Sensitivity (positives) and Specificity (negatives)
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)

        sens = (y_pred_binary[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0.0
        spec = (y_pred_binary[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0.0
        
        # F1
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average="binary", zero_division=0)

        # AUC
        try:
            auc = roc_auc_score(y_true, y_pred_filled)
            auc_scores.append(auc)
        except Exception:
            auc = 0.5
            auc_scores.append(auc)

        results[col] = {
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "f1": f1,
            "coverage": coverage,
            "auc": auc
        }

        print(f"{col:<20} | {acc*100:5.1f}% | {sens*100:5.1f}% | {spec*100:5.1f}% | {f1*100:5.1f}% | {coverage:5.1f}% | {auc:6.3f}")

    macro_auc = np.mean(auc_scores)
    print("=" * 80)
    print(f"{'MACRO AVERAGE AUC ROC':<20} | {'':<6} | {'':<6} | {'':<6} | {'':<6} | {'':<6} | {macro_auc:6.3f}")
    print("=" * 80)

    return {"per_target": results, "macro_auc": macro_auc}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


    train_path = "data/raw/train.csv"
    train_df = pd.read_csv(train_path)

    # Filter to only the 58 gold-standard studies
    gold_df = train_df.dropna(subset=["ACL"]).copy()
    print(f"Loaded {len(gold_df)} gold-standard studies for validation.\n")

    # Run regex labeler
    labeler = RegexLabeler()
    pred_df = labeler.label_dataframe(gold_df)

    # Evaluate
    evaluate_labels(pred_df, gold_df)
