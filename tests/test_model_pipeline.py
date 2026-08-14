"""Unit tests for Phase 3 Model Architecture, Dataset, and Loss modules."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add root and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import torch
    from src.datasets.knee_dataset import sample_2_5d_slabs, KneeStudyDataset
    from src.models.knee_model import GatedAttentionPool, KneeAbnormalityClassifier
    from src.training.loss import WeightedBCEWithLogitsLoss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not yet installed")
class TestModelPipeline(unittest.TestCase):

    def test_2_5d_slab_sampling_shape(self):
        """Test sampling K 2.5D slabs from a volume (N, H, W)."""
        vol = np.random.randint(0, 255, size=(30, 128, 128), dtype=np.uint8)
        slabs = sample_2_5d_slabs(vol, num_slices=8)
        self.assertEqual(slabs.shape, (8, 3, 128, 128))
        self.assertEqual(slabs.dtype, np.float32)

    def test_gated_attention_pooling(self):
        """Test Gated Attention Pool aggregates (B, K, D) to (B, D)."""
        pool = GatedAttentionPool(in_features=64, hidden_dim=32)
        x = torch.randn(2, 10, 64)  # B=2, K=10 slices, D=64
        pooled, weights = pool(x)
        self.assertEqual(pooled.shape, (2, 64))
        self.assertEqual(weights.shape, (2, 10, 1))
        # Attention weights along slice dimension must sum to 1.0
        sums = weights.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_model_forward_pass(self):
        """Test full model forward pass across all 3 anatomical planes."""
        model = KneeAbnormalityClassifier(backbone_name="resnet34", pretrained=False, num_classes=12)
        model.eval()

        b, k, c, h, w = 2, 4, 3, 64, 64
        sag = torch.randn(b, k, c, h, w)
        cor = torch.randn(b, k, c, h, w)
        ax = torch.randn(b, k, c, h, w)

        with torch.no_grad():
            logits = model(sag, cor, ax)

        self.assertEqual(logits.shape, (2, 12))

    def test_weighted_bce_loss_and_gradients(self):
        """Test loss computation and backward gradient flow."""
        criterion = WeightedBCEWithLogitsLoss(label_smoothing=0.05)
        logits = torch.randn(4, 12, requires_grad=True)
        targets = torch.tensor([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        weights = torch.tensor([[1.0]*12, [1.0]*12, [0.0]*12, [1.0]*12], dtype=torch.float32)

        loss = criterion(logits, targets, weights=weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

        # Backward pass
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (4, 12))


if __name__ == "__main__":
    unittest.main()
