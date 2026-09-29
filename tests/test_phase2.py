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

if __name__ == "__main__":
    unittest.main()
