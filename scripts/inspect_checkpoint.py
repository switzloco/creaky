import torch

ckpt_path = "checkpoints/convnext_trained/best_model_fold_0.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Checkpoint Metadata:")
print("  Architecture:", ckpt.get("architecture"))
print("  Backbone    :", ckpt.get("backbone"))
print("  Best Epoch  :", ckpt.get("epoch"))
print("  Best Val AUC:", ckpt.get("val_auc"))
print("\nPer-Class Validation AUC Breakdown:")
per_class = ckpt.get("per_class_auc", {})
for col, auc in per_class.items():
    print(f"  - {col:18s}: {auc:.4f}")

sd = ckpt.get("model_state_dict", {})
print(f"\nModel State Dict contains {len(sd)} weight tensors.")
