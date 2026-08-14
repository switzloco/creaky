"""Training and Evaluation Loop for Creaky Knee Models."""

import os
import sys
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.utils.metrics import compute_competition_metric


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training Batch", leave=False):
        sag = batch["sagittal"].to(device)
        cor = batch["coronal"].to(device)
        ax = batch["axial"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weights"].to(device)

        optimizer.zero_grad()
        logits = model(sag, cor, ax)
        loss = criterion(logits, targets, weights=weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model and compute exact competition Macro AUC ROC."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        sag = batch["sagittal"].to(device)
        cor = batch["coronal"].to(device)
        ax = batch["axial"].to(device)
        targets = batch["targets"].cpu().numpy()

        logits = model(sag, cor, ax)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(probs)
        all_targets.append(targets)

    if not all_preds:
        return 0.5, {}

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    macro_auc, per_class = compute_competition_metric(y_true, y_pred)
    return macro_auc, per_class
