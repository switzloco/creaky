import torchvision.models as tv
import torch

print("Checking Vision Transformer backbones in torchvision:")
swin_models = [m for m in dir(tv) if "swin" in m.lower() and not m.startswith("_")]
print("  Available Swin variants:", swin_models)

vit_models = [m for m in dir(tv) if "vit" in m.lower() and not m.startswith("_")]
print("  Available ViT variants :", vit_models[:10])

# Test Swin-Tiny feature extraction
print("\nTesting swin_t feature dimension:")
try:
    swin = tv.swin_t(weights=None)
    in_feat = swin.head.in_features
    print(f"  swin_t head.in_features: {in_feat}")
except Exception as e:
    print(f"  swin_t test failed: {e}")

# Test Swin-Small feature extraction
print("\nTesting swin_s feature dimension:")
try:
    swin_s = tv.swin_s(weights=None)
    in_feat_s = swin_s.head.in_features
    print(f"  swin_s head.in_features: {in_feat_s}")
except Exception as e:
    print(f"  swin_s test failed: {e}")
