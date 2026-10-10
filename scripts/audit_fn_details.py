import os
import sys
import pandas as pd
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

val_e18_path = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
df_e18 = pd.read_csv(val_e18_path)
gold_e18 = df_e18[df_e18["is_gold"] == True].copy()

train_csv = pd.read_csv("data/raw/train.csv")
train_gold = train_csv[train_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

series_csv = pd.read_csv("data/raw/train_series.csv")
gold_series = series_csv[series_csv["StudyInstanceUID"].isin(gold_e18["StudyInstanceUID"])].copy()

sag_series = gold_series[gold_series["Anatomical_Plane"] == "Sagittal"]

merged = pd.merge(gold_e18, train_gold[["StudyInstanceUID", "Report"]], on="StudyInstanceUID", how="left")

fn = merged[(merged["Lateral Meniscus"] == 1.0) & (merged["pred_Lateral Meniscus"] < 0.5)].sort_values("pred_Lateral Meniscus")

print("="*90)
print(f"DEEP DIVE: ALL {len(fn)} FALSE NEGATIVE LATERAL MENISCUS STUDIES IN GOLD")
print("="*90)

for idx, (_, row) in enumerate(fn.iterrows(), 1):
    uid = row["StudyInstanceUID"]
    s_rows = gold_series[gold_series["StudyInstanceUID"] == uid]
    sag_rows = s_rows[s_rows["Anatomical_Plane"] == "Sagittal"]
    cor_rows = s_rows[s_rows["Anatomical_Plane"] == "Coronal"]
    ax_rows = s_rows[s_rows["Anatomical_Plane"] == "Axial"]
    
    print(f"\n--- [FN {idx:02d}] UID: {uid} ---")
    print(f"Pred LM: {row['pred_Lateral Meniscus']:.4f} | True LM: 1.0")
    print(f"Pred MM: {row['pred_Medial Meniscus']:.4f} | True MM: {row['Medial Meniscus']}")
    print(f"Pred ACL: {row['pred_ACL']:.4f} | True ACL: {row['ACL']}")
    print(f"Pred Lat OA: {row['pred_Lateral OA']:.4f} | True Lat OA: {row['Lateral OA']}")
    print(f"Series Count: Total={len(s_rows)} | Sag={len(sag_rows)} | Cor={len(cor_rows)} | Ax={len(ax_rows)}")
    for _, s in s_rows.iterrows():
        print(f"  SeriesUID: {s['SeriesInstanceUID'][:30]}... | Plane: {s['Anatomical_Plane']} | Fluid: {s['Fluid_Sensitive']} | FatSup: {s['Fat_Suppression']}")
    
    print(f"Report:\n{row['Report'].strip()}")
