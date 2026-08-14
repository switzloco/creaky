"""Custom Loss Functions for Multilabel Knee Abnormality Detection."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE with Logits Loss that supports per-sample and per-target confidence weights.

    Down-weights uncertain/soft labels (like 0.5) while fully optimizing high-confidence
    gold and regex-extracted targets.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.05):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Parameters:

        - logits: (B, 12)
        - targets: (B, 12) in [0.0, 1.0]
        - weights: (B, 12) in [0.0, 1.0]
        """
        # Apply label smoothing to binary targets
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
        else:
            smoothed_targets = targets

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )

        if weights is not None:
            # Apply confidence weights
            loss = (bce_loss * weights).sum() / (weights.sum() + 1e-7)
        else:
            loss = bce_loss.mean()

        return loss
