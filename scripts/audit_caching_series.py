import os
import sys
import pandas as pd
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

series_df = pd.read_csv("data/raw/train_series.csv")
p_col = "Anatomical_Plane" if "Anatomical_Plane" in series_df.columns else "Plane"
series_df["norm_plane"] = series_df[p_col].astype(str).str.capitalize()

# Replicate the caching script's series selection logic
cached_series = {}
for _, row in series_df.iterrows():
    suid = str(row["StudyInstanceUID"])
    if suid not in cached_series:
        cached_series[suid] = {}
    plane = row["norm_plane"]
    if plane in ["Sagittal", "Coronal", "Axial"]:
        # Overwrite with last seen
        cached_series[suid][plane] = {
            "SeriesInstanceUID": str(row["SeriesInstanceUID"]),
            "Fluid_Sensitive": row.get("Fluid_Sensitive", None),
            "Fat_Suppression": row.get("Fat_Suppression", None),
        }

# Check for the 58 Gold studies
val_e18_path = "checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv"
df_e18 = pd.read_csv(val_e18_path)
gold_e18 = df_e18[df_e18["is_gold"] == True].copy()

# Audit selected sagittal series for all 58 Gold studies
selected_info = []
for _, row in gold_e18.iterrows():
    suid = row["StudyInstanceUID"]
    info = cached_series.get(suid, {}).get("Sagittal", {})
    all_sags = series_df[(series_df["StudyInstanceUID"] == suid) & (series_df["norm_plane"] == "Sagittal")]
    selected_info.append({
        "StudyInstanceUID": suid,
        "is_gold": True,
        "true_LM": row["Lateral Meniscus"],
        "pred_LM": row["pred_Lateral Meniscus"],
        "true_MM": row["Medial Meniscus"],
        "pred_MM": row["pred_Medial Meniscus"],
        "sag_count": len(all_sags),
        "selected_fluid": info.get("Fluid_Sensitive"),
        "selected_fat": info.get("Fat_Suppression"),
        "has_fluid_sag": (all_sags["Fluid_Sensitive"] == 1).any(),
        "has_fat_sag": (all_sags["Fat_Suppression"] == 1).any(),
    })

audit_df = pd.DataFrame(selected_info)

print("="*80)
print("AUDIT OF SERIES SELECTION IN CACHED DATASET (GOLD STUDIES, N=58)")
print("="*80)
print(f"Studies with multiple Sagittal series: {(audit_df['sag_count'] > 1).sum()} / {len(audit_df)}")
print(f"Studies where selected Sagittal is NOT fluid sensitive: {(audit_df['selected_fluid'] == 0).sum()} / {len(audit_df)}")
print(f"Studies with fluid sensitive sagittal available but NON-fluid was selected: {((audit_df['selected_fluid'] == 0) & audit_df['has_fluid_sag']).sum()}")

# Check False Negatives for Lateral Meniscus
fn_audit = audit_df[(audit_df["true_LM"] == 1.0) & (audit_df["pred_LM"] < 0.5)]
print(f"\nOf the {len(fn_audit)} False Negatives for Lateral Meniscus:")
print(f"  Selected Sagittal is NOT fluid-sensitive: {(fn_audit['selected_fluid'] == 0).sum()} / {len(fn_audit)}")
print(f"  Fluid-sensitive was available but overwritten by non-fluid: {((fn_audit['selected_fluid'] == 0) & fn_audit['has_fluid_sag']).sum()}")

# Print table of the 13 FNs
print("\nFalse Negative Breakdown:")
for _, r in fn_audit.iterrows():
    print(f"UID: {r['StudyInstanceUID'][:25]}... | Pred LM: {r['pred_LM']:.4f} | Total Sags: {r['sag_count']} | Selected Fluid: {r['selected_fluid']} | Fluid Available: {r['has_fluid_sag']}")
