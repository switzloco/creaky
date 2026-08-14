"""Advanced Medical Image Augmentations for Multi-Planar Knee MRI.

Includes clinically consistent transforms and Laterality-Aware Horizontal Flipping.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch


# Target index mappings
TARGET_INDICES = {
    "ACL": 0, "MCL": 1, "Medial Meniscus": 2, "Lateral Meniscus": 3,
    "Medial OA": 4, "Lateral OA": 5, "PF OA": 6, "Effusion": 7,
    "Synovitis": 8, "Baker's": 9, "Contusion": 10, "Fracture": 11
}


def swap_laterality_labels(targets: np.ndarray) -> np.ndarray:
    """Swap Medial and Lateral finding labels when an image is horizontally mirrored.

    Medial Meniscus (index 2) <--> Lateral Meniscus (index 3)
    Medial OA (index 4)       <--> Lateral OA (index 5)
    """
    new_targets = targets.copy()
    # Swap Meniscus
    new_targets[2], new_targets[3] = targets[3], targets[2]
    # Swap OA
    new_targets[4], new_targets[5] = targets[5], targets[4]
    return new_targets


class LateralityAwareAugmenter:
    """Medical MRI volume augmenter with laterality consistency."""

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_contrast: float = 0.5,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: Tuple[float, float] = (-0.1, 0.1),
        p_noise: float = 0.3
    ):
        self.p_hflip = p_hflip
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.p_noise = p_noise

    def __call__(
        self,
        slabs: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transforms to slabs of shape (K, 3, H, W) and update targets.

        Returns:
        - augmented_slabs: (K, 3, H, W)
        - augmented_targets: (12,)
        """
        out_slabs = slabs.copy()
        out_targets = targets.copy()

        # 1. Laterality-Aware Horizontal Flip (applied to Coronal & Axial slices)
        if np.random.rand() < self.p_hflip:
            # Flip along horizontal axis (W = axis 3)
            out_slabs = np.flip(out_slabs, axis=3).copy()
            out_targets = swap_laterality_labels(out_targets)

        # 2. Brightness & Contrast Scaling (MRI intensity shift)
        if np.random.rand() < self.p_contrast:
            alpha = np.random.uniform(*self.contrast_range)
            beta = np.random.uniform(*self.brightness_range)
            out_slabs = out_slabs * alpha + beta

        # 3. Additive Gaussian noise (simulates coil noise)
        if np.random.rand() < self.p_noise:
            noise = np.random.normal(0.0, 0.02, size=out_slabs.shape).astype(np.float32)
            out_slabs = out_slabs + noise

        return out_slabs, out_targets
