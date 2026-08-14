"""Unit tests for the metrics module."""
import os
import sys
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.abspath(os.path.join(current_dir, "..", "scripts", "utils"))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from metrics import compute_competition_metric, DEFAULT_TARGETS


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        """Perfect predictions should yield macro AUC of 1.0."""
        y_true = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        ])
        y_pred = np.array([
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        ])
        macro_auc, per_class = compute_competition_metric(y_true, y_pred)
        self.assertAlmostEqual(macro_auc, 1.0, places=4)

    def test_random_guess_predictions(self):
        """Uniform predictions (0.5) should yield macro AUC of 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=(200, 12))
        y_pred = np.full((200, 12), 0.5)
        macro_auc, _ = compute_competition_metric(y_true, y_pred)
        self.assertAlmostEqual(macro_auc, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
