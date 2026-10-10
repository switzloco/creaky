import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

val_e18_path = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
df_e18 = pd.read_csv(val_e18_path)
gold_e18 = df_e18[df_e18["is_gold"] == True].copy()

train_csv = pd.read_csv("data/raw/train.csv")
train_gold = train_csv[train_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

series_csv = pd.read_csv("data/raw/train_series.csv")
gold_series = series_csv[series_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

# Count sagittal series and total series
sag_counts = gold_series[gold_series["Anatomical_Plane"] == "Sagittal"].groupby("StudyInstanceUID")["SeriesInstanceUID"].count()
cor_counts = gold_series[gold_series["Anatomical_Plane"] == "Coronal"].groupby("StudyInstanceUID")["SeriesInstanceUID"].count()
ax_counts = gold_series[gold_series["Anatomical_Plane"] == "Axial"].groupby("StudyInstanceUID")["SeriesInstanceUID"].count()

merged = pd.merge(gold_e18, train_gold[["StudyInstanceUID", "Report"]], on="StudyInstanceUID", how="left")
merged["sag_series_count"] = merged["StudyInstanceUID"].map(sag_counts).fillna(0).astype(int)
merged["cor_series_count"] = merged["StudyInstanceUID"].map(cor_counts).fillna(0).astype(int)
merged["ax_series_count"] = merged["StudyInstanceUID"].map(ax_counts).fillna(0).astype(int)

# Target comparison
print("================================================================================")
print("COMPREHENSIVE AUDIT: 23 GOLD POSITIVE LATERAL MENISCUS STUDIES")
print("================================================================================")

pos = merged[merged["Lateral Meniscus"] == 1.0].sort_values("pred_Lateral Meniscus")
for i, (_, row) in enumerate(pos.iterrows(), 1):
    status = "TP" if row["pred_Lateral Meniscus"] >= 0.5 else "FN"
    print(f"\n[{i:02d}] Status: {status} | Pred LM: {row['pred_Lateral Meniscus']:.4f} | Pred MM: {row['pred_Medial Meniscus']:.4f} (True MM: {row['Medial Meniscus']})")
    print(f"     Planes: Sag={row['sag_series_count']}, Cor={row['cor_series_count']}, Ax={row['ax_series_count']}")
    print(f"     UID: {row['StudyInstanceUID']}")
    # Print short snippet of report
    rep = row['Report'].replace('\n', ' ').strip()
    if len(rep) > 300:
        rep = rep[:300] + "..."
    print(f"     Report: {rep}")

print("\n" + "="*80)
print("AUDIT: 8 FALSE POSITIVE LATERAL MENISCUS STUDIES (True LM = 0.0, Pred LM >= 0.5)")
print("="*80)
fp = merged[(merged["Lateral Meniscus"] == 0.0) & (merged["pred_Lateral Meniscus"] >= 0.5)].sort_values("pred_Lateral Meniscus", ascending=False)
for i, (_, row) in enumerate(fp.iterrows(), 1):
    print(f"\n[{i:02d}] FP | Pred LM: {row['pred_Lateral Meniscus']:.4f} | Pred MM: {row['pred_Medial Meniscus']:.4f} (True MM: {row['Medial Meniscus']})")
    print(f"     Planes: Sag={row['sag_series_count']}, Cor={row['cor_series_count']}, Ax={row['ax_series_count']}")
    print(f"     UID: {row['StudyInstanceUID']}")
    rep = row['Report'].replace('\n', ' ').strip()
    if len(rep) > 300:
        rep = rep[:300] + "..."
    print(f"     Report: {rep}")
