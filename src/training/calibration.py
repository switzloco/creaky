"""Post-processing Probability Calibration and Clinical Co-occurrence Adjuster."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class TemperatureScaler:
    """Calibrates prediction confidence using learned per-target temperature parameters."""

    def __init__(self, num_classes: int = 12):
        self.num_classes = num_classes
        self.temperatures = np.ones(num_classes, dtype=np.float32)

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "TemperatureScaler":
        """Fit temperature parameter T per class on validation logits."""
        for i in range(self.num_classes):
            z = logits[:, i]
            y = targets[:, i]
            # Ignore soft uncertainty labels (0.5) during temperature fitting
            valid_mask = (y == 0.0) | (y == 1.0)
            if valid_mask.sum() < 5:
                continue

            z_val = z[valid_mask]
            y_val = y[valid_mask]

            def nll_loss(t):
                temp = max(1e-3, float(t[0]))
                p = 1.0 / (1.0 + np.exp(-z_val / temp))
                p = np.clip(p, 1e-7, 1.0 - 1e-7)
                loss = -np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p))
                return loss

            res = minimize(nll_loss, x0=[1.0], method="Nelder-Mead", bounds=[(0.05, 5.0)])
            self.temperatures[i] = float(np.clip(res.x[0], 0.1, 5.0))

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperatures to scale logits into calibrated probabilities."""
        scaled_logits = logits / np.maximum(1e-3, self.temperatures.reshape(1, -1))
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return np.clip(probs, 0.001, 0.999)


def apply_clinical_cooccurrence_priors(
    probs: np.ndarray,
    target_cols: Optional[List[str]] = None
) -> np.ndarray:
    """Adjust probabilities based on biological joint co-occurrence patterns.

    Clinical Rules:
    1. Pivot-Shift ACL Tear -> elevates Lateral Bone Contusion & Medial Meniscus tear risk.
    2. Medial OA -> elevates Medial Meniscus degenerative tear risk.
    3. Severe Effusion -> co-occurs with Synovitis.
    """
    out_probs = probs.copy()
    num_samples = len(out_probs)

    # Finding indices:
    # 0: ACL, 1: MCL, 2: Medial Meniscus, 3: Lateral Meniscus, 4: Medial OA,
    # 5: Lateral OA, 6: PF OA, 7: Effusion, 8: Synovitis, 9: Baker's, 10: Contusion, 11: Fracture
    for idx in range(num_samples):
        acl_p = out_probs[idx, 0]
        moa_p = out_probs[idx, 4]
        eff_p = out_probs[idx, 7]

        # Rule 1: High ACL confidence boosts contusion & medial meniscus slightly
        if acl_p > 0.7:
            out_probs[idx, 10] = np.clip(out_probs[idx, 10] * 1.15, 0.0, 0.99)  # Contusion
            out_probs[idx, 2] = np.clip(out_probs[idx, 2] * 1.10, 0.0, 0.99)   # Medial Meniscus

        # Rule 2: High Medial OA boosts Medial Meniscus degenerative signal
        if moa_p > 0.7:
            out_probs[idx, 2] = np.clip(out_probs[idx, 2] * 1.12, 0.0, 0.99)

        # Rule 3: High Effusion co-occurs with Synovitis
        if eff_p > 0.75:
            out_probs[idx, 8] = np.clip(out_probs[idx, 8] * 1.10, 0.0, 0.99)

    return np.clip(out_probs, 0.001, 0.999)
