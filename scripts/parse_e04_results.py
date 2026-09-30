import json
import torch
import pandas as pd
import numpy as np

# 1. Parse log
print("======================================================================")
print("PARSING E04 EXECUTION LOG")
print("======================================================================")
with open("checkpoints/e04_trained/training-book.log", "r", encoding="utf-8") as f:
    text = f.read()

events = json.loads(text)
full_text = "".join([e.get("data", "") for e in events])

# Extract epoch summaries
import re
epochs = re.findall(r"--- Epoch (\d+)/\d+ Summary \(Fold \d+\) ---(.*?)(?=--- Epoch|\Z)", full_text, re.DOTALL)
for ep_num, ep_body in epochs:
    print(f"\n--- Epoch {ep_num} ---")
    for line in ep_body.strip().splitlines():
        if any(k in line for k in ["Train Loss", "Gold   AUC", "Silver AUC", "Selection", "Saved NEW BEST"]):
            print(" ", line.strip())

# 2. Inspect Checkpoint
print("\n======================================================================")
print("INSPECTING CHECKPOINT: checkpoints/e04_trained/best_model_fold_0.pt")
print("======================================================================")
ckpt = torch.load("checkpoints/e04_trained/best_model_fold_0.pt", map_location="cpu")
print("Saved Epoch         :", ckpt.get("epoch"))
print("Architecture        :", ckpt.get("architecture"))
print("Backbone            :", ckpt.get("backbone"))
print("Preprocessing Tag   :", ckpt.get("preprocessing"))
print("Fold                :", ckpt.get("fold"))
print("Best Selection AUC  :", ckpt.get("val_auc"))
print("Selection Metric    :", ckpt.get("select_metric"))
print("Gold AUC at Best    :", ckpt.get("gold_auc"))
print("Silver AUC at Best  :", ckpt.get("silver_auc"))
print("\nPer-Class Silver AUC at Best:")
for k, v in ckpt.get("silver_per_class_auc", {}).items():
    print(f"  {k:18s}: {v:.4f}")

# 3. Inspect Validation Predictions
print("\n======================================================================")
print("INSPECTING VALIDATION PREDICTIONS: checkpoints/e04_trained/val_preds_fold_0.csv")
print("======================================================================")
preds_df = pd.read_csv("checkpoints/e04_trained/val_preds_fold_0.csv")
print(f"Rows: {len(preds_df)}, Columns: {list(preds_df.columns)}")
print(f"Gold studies: {preds_df['is_gold'].sum()}, Silver OOF studies: {(~preds_df['is_gold']).sum()}")
print("Sample prediction probabilities:")
pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
print(preds_df[pred_cols[:4]].describe().round(4))
