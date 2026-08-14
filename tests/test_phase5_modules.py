"""Unit tests for Phase 5: Augmentations, Calibration, and Ensembling modules."""

import os
import sys
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.datasets.augmentations import LateralityAwareAugmenter, swap_laterality_labels
from src.training.calibration import TemperatureScaler, apply_clinical_cooccurrence_priors
from src.models.ensemble import mean_ensemble, rank_average_ensemble


class TestPhase5Modules(unittest.TestCase):

    def test_laterality_label_swap(self):
        """Test swapping Medial <-> Lateral findings on horizontal flip."""
        # Index 2: Medial Meniscus (1.0), Index 3: Lateral Meniscus (0.0)
        # Index 4: Medial OA (1.0), Index 5: Lateral OA (0.0)
        targets = np.zeros(12, dtype=np.float32)
        targets[2] = 1.0  # Medial Meniscus
        targets[4] = 1.0  # Medial OA

        swapped = swap_laterality_labels(targets)
        self.assertEqual(swapped[2], 0.0)  # Medial Meniscus should now be 0.0
        self.assertEqual(swapped[3], 1.0)  # Lateral Meniscus should now be 1.0
        self.assertEqual(swapped[4], 0.0)  # Medial OA should now be 0.0
        self.assertEqual(swapped[5], 1.0)  # Lateral OA should now be 1.0

    def test_temperature_scaler(self):
        """Test temperature scaling transforms logits to calibrated probabilities."""
        scaler = TemperatureScaler(num_classes=12)
        logits = np.random.randn(50, 12).astype(np.float32)
        targets = np.random.randint(0, 2, size=(50, 12)).astype(np.float32)

        scaler.fit(logits, targets)
        calibrated_probs = scaler.transform(logits)

        self.assertEqual(calibrated_probs.shape, (50, 12))
        self.assertTrue((calibrated_probs >= 0.001).all())
        self.assertTrue((calibrated_probs <= 0.999).all())

    def test_clinical_cooccurrence_prior(self):
        """Test that high ACL confidence adjusts contusion/meniscus probabilities."""
        probs = np.full((1, 12), 0.5, dtype=np.float32)
        probs[0, 0] = 0.9  # High ACL tear confidence
        adjusted = apply_clinical_cooccurrence_priors(probs)

        # Contusion (index 10) and Medial Meniscus (index 2) should be boosted
        self.assertGreater(adjusted[0, 10], 0.5)
        self.assertGreater(adjusted[0, 2], 0.5)

    def test_ensembling_methods(self):
        """Test mean and rank-average ensemble aggregations."""
        p1 = np.array([[0.8, 0.2], [0.3, 0.7]])
        p2 = np.array([[0.6, 0.4], [0.5, 0.5]])

        mean_res = mean_ensemble([p1, p2])
        self.assertAlmostEqual(mean_res[0, 0], 0.7, places=3)

        rank_res = rank_average_ensemble([p1, p2])
        self.assertEqual(rank_res.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
