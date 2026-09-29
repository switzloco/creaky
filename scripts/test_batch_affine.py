import torch
import torchvision.transforms.functional as TF

tensor = torch.randn(16, 3, 256, 256)
out = TF.affine(tensor, angle=5.0, translate=[2, -3], scale=1.02, shear=[0.0, 0.0])
print("Batch affine output shape:", out.shape)
assert out.shape == tensor.shape
print("Vectorized TF.affine works directly on (K, 3, H, W)!")
