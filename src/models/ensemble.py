"""Multi-Model Ensembling and Probability Blending Engine."""

from typing import List, Optional
import numpy as np
from scipy.stats import rankdata


def mean_ensemble(prediction_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Compute weighted arithmetic mean across multiple model predictions."""
    if not prediction_list:
        raise ValueError("Prediction list is empty.")

    if weights is None:
        weights = [1.0 / len(prediction_list)] * len(prediction_list)

    norm_weights = np.array(weights, dtype=np.float32) / sum(weights)
    ensemble = np.zeros_like(prediction_list[0])

    for pred, w in zip(prediction_list, norm_weights):
        ensemble += pred * w

    return np.clip(ensemble, 0.001, 0.999)


def rank_average_ensemble(prediction_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Compute normalized rank-average ensemble across models (robust to probability scale shifts)."""
    if not prediction_list:
        raise ValueError("Prediction list is empty.")

    if weights is None:
        weights = [1.0 / len(prediction_list)] * len(prediction_list)

    norm_weights = np.array(weights, dtype=np.float32) / sum(weights)
    n_samples, n_classes = prediction_list[0].shape
    blended_ranks = np.zeros((n_samples, n_classes), dtype=np.float32)

    for pred, w in zip(prediction_list, norm_weights):
        for j in range(n_classes):
            ranks = rankdata(pred[:, j]) / float(n_samples)
            blended_ranks[:, j] += ranks * w

    return np.clip(blended_ranks, 0.001, 0.999)
