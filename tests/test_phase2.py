import unittest
import torch
import numpy as np
import pandas as pd
import random

from scripts.test_multilabel_kfold import iterative_multilabel_split
from scripts.test_augmentation import VolumeConsistentAugmenter

class TestPhase2(unittest.TestCase):
    def test_stratification_balance(self):
        # Create synthetic multilabel dataset
        np.random.seed(42)
        n = 500
        targets = [f"T{i}" for i in range(12)]
        data = {t: np.random.choice([0.0, 1.0], size=n, p=[0.8, 0.2]) for t in targets}
        df = pd.DataFrame(data)
        
        folds = iterative_multilabel_split(df, targets, n_splits=5, seed=42)
        df["fold"] = folds
        
        # Check fold sizes: each should be ~100 +/- 2
        counts = df["fold"].value_counts()
        self.assertEqual(len(counts), 5)
        for c in counts:
            self.assertTrue(98 <= c <= 102)

        # Check positive distribution: each fold should have positives for every target
        for t in targets:
            pos_per_fold = [df[df["fold"] == f][t].sum() for f in range(5)]
            total_pos = df[t].sum()
            expected = total_pos / 5.0
            for p in pos_per_fold:
                self.assertTrue(abs(p - expected) <= 3)

    def test_volume_consistent_augmenter_properties(self):
        aug = VolumeConsistentAugmenter(p=1.0)
        tensor = torch.randn(16, 3, 64, 64)
        out = aug(tensor)
        
        # Shape must be identical
        self.assertEqual(out.shape, tensor.shape)
        # Slices must be different from input (transformation occurred)
        self.assertFalse(torch.allclose(out, tensor))
        
        # Test p=0.0: should return exact tensor
        no_aug = VolumeConsistentAugmenter(p=0.0)
        out_no_aug = no_aug(tensor)
        self.assertTrue(torch.equal(out_no_aug, tensor))

    def test_volume_consistent_hflip(self):
        # When hflip is activated, horizontal flip is applied identically across all K slices in plane
        import torchvision.transforms.functional as TF
        aug = VolumeConsistentAugmenter(p=0.0, hflip=True)
        tensor = torch.randn(16, 3, 64, 64)
        
        # Test flip applied
        flipped = aug(tensor, flip_this_volume=True)
        expected = TF.hflip(tensor)
        self.assertTrue(torch.equal(flipped, expected))
        
        # Verify that all slices were flipped identically across width dimension
        for k in range(16):
            self.assertTrue(torch.equal(flipped[k], TF.hflip(tensor[k])))

    def test_metric_continuous_and_discrete_labels(self):
        from scripts.training.train_kaggle_notebook import compute_competition_metric, label_counts, TARGET_COLS
        
        # 1. Discrete labels (Gold format: 0.0, 1.0)
        np.random.seed(42)
        n = 100
        y_gold = np.random.choice([0.0, 1.0], size=(n, 12), p=[0.7, 0.3])
        # Model predictions: correlated with truth
        y_pred = np.clip(y_gold + np.random.normal(0, 0.2, size=(n, 12)), 0.01, 0.99)
        
        macro_gold, per_class_gold = compute_competition_metric(y_gold, y_pred)
        self.assertFalse(np.isnan(macro_gold))
        self.assertGreater(macro_gold, 0.8)
        
        counts_gold = label_counts(y_gold)
        for t in TARGET_COLS:
            pos, neg = counts_gold[t]
            self.assertEqual(pos + neg, n)

        # 2. Continuous soft labels (Jev Silver format: e.g. 0.02, 0.98)
        y_silver = np.random.choice([0.02, 0.98], size=(n, 12), p=[0.7, 0.3])
        y_pred_silver = np.clip(y_silver + np.random.normal(0, 0.1, size=(n, 12)), 0.01, 0.99)
        macro_silver, per_class_silver = compute_competition_metric(y_silver, y_pred_silver)
        self.assertFalse(np.isnan(macro_silver))
        self.assertGreater(macro_silver, 0.8)
        
        counts_silver = label_counts(y_silver)
        for t in TARGET_COLS:
            pos, neg = counts_silver[t]
            self.assertEqual(pos + neg, n)

        # 3. Soft unannotated 0.5 labels are excluded from evaluation
        y_with_half = np.copy(y_gold)
        y_with_half[:10, :] = 0.5  # 10 studies with 0.5
        macro_half, _ = compute_competition_metric(y_with_half, y_pred)
        self.assertFalse(np.isnan(macro_half))
        counts_half = label_counts(y_with_half)
        for t in TARGET_COLS:
            pos, neg = counts_half[t]
            self.assertEqual(pos + neg, n - 10)  # exactly 10 excluded

if __name__ == "__main__":
    unittest.main()
