import torch
import torchvision.transforms.functional as TF
import random

class VolumeConsistentAugmenter:
    """Applies affine & photometric augmentations consistently across all slices in a plane.
    
    CRITICAL: Never applies horizontal flip, to preserve Medial vs. Lateral anatomy!
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor shape: (K, 3, H, W) - K slices, 3 slab channels, H x W pixels."""
        if random.random() > self.p:
            return tensor

        # Pick random parameters ONCE for the entire series
        angle = random.uniform(-7.0, 7.0)
        max_dx = 0.05 * tensor.shape[-1]
        max_dy = 0.05 * tensor.shape[-2]
        translations = (int(random.uniform(-max_dx, max_dx)), int(random.uniform(-max_dy, max_dy)))
        scale = random.uniform(0.95, 1.05)
        
        # Photometric adjustments
        contrast_factor = random.uniform(0.90, 1.10)
        brightness_factor = random.uniform(-0.08, 0.08)

        # Apply to all K slices identically
        K, C, H, W = tensor.shape
        # Reshape to (K*C, 1, H, W) or process per slice
        out = tensor.clone()
        for k in range(K):
            # Affine on the 3-channel slab
            slab = TF.affine(out[k], angle=angle, translate=translations, scale=scale, shear=0)
            # Photometric
            slab = slab * contrast_factor + brightness_factor
            out[k] = slab

        return out

# Test with dummy tensor
dummy = torch.randn(16, 3, 256, 256)
aug = VolumeConsistentAugmenter(p=1.0)
augmented = aug(dummy)

print("Original shape:", dummy.shape, "min:", dummy.min().item(), "max:", dummy.max().item())
print("Augmented shape:", augmented.shape, "min:", augmented.min().item(), "max:", augmented.max().item())
print("Mean absolute difference:", torch.abs(augmented - dummy).mean().item())
assert augmented.shape == dummy.shape
print("Augmentation test PASSED successfully!")
