"""Competition evaluation metrics for RSNA Knee Abnormality Detection.

Metric: Macro-averaged Area Under the ROC Curve (AUC ROC) across 12 target findings.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


def compute_competition_metric(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_cols: List[str] = None
) -> Tuple[float, Dict[str, float]]:
    """Compute the macro-averaged AUC ROC across all 12 targets.

    Parameters:
    - y_true: Ground truth binary targets (N, 12) or DataFrame with target columns
    - y_pred: Predicted confidence scores / probabilities in [0, 1] (N, 12) or DataFrame
    - target_cols: List of target names (default: 12 competition findings)

    Returns:
    - (macro_auc, per_class_auc_dict)
    """
    if target_cols is None:
        target_cols = DEFAULT_TARGETS

    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[target_cols].values.astype(float)
    else:
        y_true_arr = np.array(y_true, dtype=float)

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred[target_cols].values.astype(float)
    else:
        y_pred_arr = np.array(y_pred, dtype=float)

    per_class_auc = {}
    valid_aucs = []

    for i, col in enumerate(target_cols):
        col_true = y_true_arr[:, i]
        col_pred = y_pred_arr[:, i]

        # Check if both classes (0 and 1) are present in ground truth
        unique_classes = np.unique(col_true[~np.isnan(col_true)])
        if len(unique_classes) < 2:
            # Cannot calculate AUC if only one class exists in batch/fold
            auc = 0.5
        else:
            try:
                # Handle NaNs in prediction by replacing with 0.5 baseline
                col_pred_clean = np.where(np.isnan(col_pred), 0.5, col_pred)
                auc = roc_auc_score(col_true, col_pred_clean)
            except Exception:
                auc = 0.5

        per_class_auc[col] = float(auc)
        valid_aucs.append(auc)

    macro_auc = float(np.mean(valid_aucs))
    return macro_auc, per_class_auc


if __name__ == "__main__":
    # Quick sanity test
    np.random.seed(42)
    dummy_true = np.random.randint(0, 2, size=(100, 12))
    dummy_pred = np.random.uniform(0, 1, size=(100, 12))
    macro, per_class = compute_competition_metric(dummy_true, dummy_pred)
    print(f"Sanity Check Macro AUC: {macro:.4f}")
    for k, v in per_class.items():
        print(f"  {k:20s}: {v:.4f}")
