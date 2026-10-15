import json
import torch
from kaggle.api.kaggle_api_extended import KaggleApi

ckpt = torch.load("checkpoints/c384_cleanlab/best_model_fold_1.pt", map_location="cpu", weights_only=False)
print("=== CHECKPOINT METADATA ===")
for k in ["epoch", "architecture", "backbone", "fold", "preprocessing", "val_auc", "gold_auc", "silver_auc", "config"]:
    print(f"  {k}: {ckpt.get(k)}")

api = KaggleApi()
api.authenticate()
raw_log = api.kernels_logs("nswitzer/training-book-c384-cleanlab-gpu-fold-1")
entries = json.loads(raw_log)
print("\n=== EPOCH SUMMARIES FROM KAGGLE LOG ===")
for e in entries:
    data = e.get("data", "")
    for line in data.splitlines():
        if any(w in line for w in ["--- Epoch", "Train Loss", "Train cls loss", "Gold   AUC:", "Silver AUC:", "Selection (", ">>> Saved", "Training Complete", "[Fast Mode]"]):
            print(" ", line)
