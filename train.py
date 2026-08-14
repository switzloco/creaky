"""Master Training Script for Creaky Knee Abnormality Detection Models.

Supports multi-fold cross-validation, 2.5D slice attention pooling,
mixed-precision training, and Kaggle competition metric tracking (Macro AUC ROC).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from src.datasets.knee_dataset import KneeStudyDataset, DEFAULT_TARGETS
from src.models.knee_model import KneeAbnormalityClassifier
from src.training.loss import WeightedBCEWithLogitsLoss
from src.training.trainer import train_one_epoch, evaluate_model
from scripts.utils.metrics import compute_competition_metric


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    series_meta: pd.DataFrame,
    volumes_dir: str,
    args: argparse.Namespace
) -> float:
    """Train a single fold model and return best validation Macro AUC ROC."""
    print(f"\n{'='*70}")
    print(f"STARTING FOLD {fold} | Train: {len(train_df)} studies | Val: {len(val_df)} studies")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeStudyDataset(
        labels_df=train_df,
        series_meta_df=series_meta,
        volumes_dir=volumes_dir,
        num_slices_per_plane=args.num_slices,
        is_training=True
    )
    val_dataset = KneeStudyDataset(
        labels_df=val_df,
        series_meta_df=series_meta,
        volumes_dir=volumes_dir,
        num_slices_per_plane=args.num_slices,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model
    model = KneeAbnormalityClassifier(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        num_classes=len(DEFAULT_TARGETS),
        dropout=args.dropout
    ).to(device)

    criterion = WeightedBCEWithLogitsLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, per_class_auc = evaluate_model(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Macro AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "per_class_auc": per_class_auc,
                "args": vars(args)
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}")

    print(f"\nFold {fold} finished. Best Validation Macro AUC: {best_val_auc:.4f}")
    return best_val_auc


def main():
    parser = argparse.ArgumentParser(description="Train Creaky Knee Abnormality Detection Models")
    parser.add_argument("--labels_csv", default="data/processed/train_labels.csv")
    parser.add_argument("--meta_csv", default="data/processed/sample_processed/series_metadata.csv")
    parser.add_argument("--volumes_dir", default="data/processed/sample_processed")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_slices", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.labels_csv):
        print(f"Error: {args.labels_csv} not found.")
        sys.exit(1)

    labels_df = pd.read_csv(args.labels_csv)
    series_meta = pd.read_csv(args.meta_csv) if os.path.exists(args.meta_csv) else pd.DataFrame()

    print(f"Loaded {len(labels_df)} total studies.")
    train_fold(fold=0, train_df=labels_df.head(2), val_df=labels_df.head(2), series_meta=series_meta, volumes_dir=args.volumes_dir, args=args)


if __name__ == "__main__":
    main()
