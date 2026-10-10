import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 1. Load E18 validation predictions
val_e18_path = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
if not os.path.exists(val_e18_path):
    print(f"Error: {val_e18_path} not found.")
    exit(1)

df_e18 = pd.read_csv(val_e18_path)
gold_e18 = df_e18[df_e18["is_gold"] == True].copy()

# Load train.csv and train_series.csv
train_csv = pd.read_csv("data/raw/train.csv")
train_gold = train_csv[train_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

series_csv = pd.read_csv("data/raw/train_series.csv")
gold_series = series_csv[series_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

# Aggregate series info per study
series_summary = gold_series.groupby("StudyInstanceUID").agg(
    series_count=("SeriesInstanceUID", "count"),
    planes=("Anatomical_Plane", lambda x: list(x.dropna().unique())),
).reset_index()


merged = pd.merge(gold_e18, train_gold[["StudyInstanceUID", "Report"]], on="StudyInstanceUID", how="left")
merged = pd.merge(merged, series_summary, on="StudyInstanceUID", how="left")

y_true = merged["Lateral Meniscus"].values
y_pred = merged["pred_Lateral Meniscus"].values

auc = roc_auc_score(y_true, y_pred)
print(f"\n=======================================================")
print(f"LATERAL MENISCUS GOLD PERFORMANCE (Fold 0, N={len(merged)})")
print(f"AUC: {auc:.4f}")
print(f"True Positives in Gold: {int(y_true.sum())} / {len(y_true)} ({y_true.mean()*100:.1f}%)")
print(f"=======================================================\n")

merged["LM_pred_binary"] = (merged["pred_Lateral Meniscus"] >= 0.5).astype(float)
merged["error_type"] = "TN"
merged.loc[(merged["Lateral Meniscus"] == 1.0) & (merged["LM_pred_binary"] == 1.0), "error_type"] = "TP"
merged.loc[(merged["Lateral Meniscus"] == 1.0) & (merged["LM_pred_binary"] == 0.0), "error_type"] = "FN"
merged.loc[(merged["Lateral Meniscus"] == 0.0) & (merged["LM_pred_binary"] == 1.0), "error_type"] = "FP"
merged.loc[(merged["Lateral Meniscus"] == 0.0) & (merged["LM_pred_binary"] == 0.0), "error_type"] = "TN"

print("Confusion Matrix Breakdown (Threshold 0.5):")
print(merged["error_type"].value_counts())


print("\n" + "="*80)
print("ALL POSITIVE GOLD CASES (Lateral Meniscus = 1.0):")
print("="*80)
pos_cases = merged[merged["Lateral Meniscus"] == 1.0].sort_values("pred_Lateral Meniscus")
for idx, row in pos_cases.iterrows():
    print(f"\nUID: {row['StudyInstanceUID']}")
    print(f"Pred LM: {row['pred_Lateral Meniscus']:.4f} | Medial Meniscus: {row['Medial Meniscus']} (Pred: {row['pred_Medial Meniscus']:.4f})")
    print(f"ACL: {row['ACL']} (Pred: {row['pred_ACL']:.4f}) | Lat OA: {row['Lateral OA']} (Pred: {row['pred_Lateral OA']:.4f})")
    print(f"Report: {row['Report']}")

print("\n" + "="*80)
print("FALSE POSITIVE CASES (Lateral Meniscus = 0.0, Pred >= 0.5):")
print("="*80)
fp_cases = merged[(merged["Lateral Meniscus"] == 0.0) & (merged["pred_Lateral Meniscus"] >= 0.5)].sort_values("pred_Lateral Meniscus", ascending=False)
for idx, row in fp_cases.iterrows():
    print(f"\nUID: {row['StudyInstanceUID']}")
    print(f"Pred LM: {row['pred_Lateral Meniscus']:.4f} | Medial Meniscus: {row['Medial Meniscus']} (Pred: {row['pred_Medial Meniscus']:.4f})")
    print(f"Report: {row['Report']}")
