import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/train_series.csv")
p_col = "Anatomical_Plane" if "Anatomical_Plane" in df.columns else "Plane"
df["norm_plane"] = df[p_col].astype(str).str.capitalize()

# Compute priority score
# Fluid=1 & Fat=1 is gold standard for cartilage, meniscus, and bone marrow edema!
fluid = df.get("Fluid_Sensitive", pd.Series(0, index=df.index)).fillna(0).astype(int)
fat = df.get("Fat_Suppression", pd.Series(0, index=df.index)).fillna(0).astype(int)

df["priority_score"] = (fluid * 2) + fat

# Group by StudyInstanceUID and norm_plane
# Old naive approach: took last row in csv
old_selection = {}
for _, row in df.iterrows():
    suid = str(row["StudyInstanceUID"])
    if suid not in old_selection:
        old_selection[suid] = {}
    plane = row["norm_plane"]
    if plane in ["Sagittal", "Coronal", "Axial"]:
        old_selection[suid][plane] = row

# New smart approach: sort by priority_score descending, pick highest score
new_selection = {}
sorted_df = df.sort_values(["StudyInstanceUID", "norm_plane", "priority_score"], ascending=[True, True, False])
for _, row in sorted_df.iterrows():
    suid = str(row["StudyInstanceUID"])
    if suid not in new_selection:
        new_selection[suid] = {}
    plane = row["norm_plane"]
    if plane in ["Sagittal", "Coronal", "Axial"] and plane not in new_selection[suid]:
        new_selection[suid][plane] = row

print(f"Total Unique Studies: {len(new_selection)}")

# Compare Sagittal selections
old_fluid_sags = sum(1 for s in old_selection.values() if "Sagittal" in s and s["Sagittal"]["Fluid_Sensitive"] == 1)
new_fluid_sags = sum(1 for s in new_selection.values() if "Sagittal" in s and s["Sagittal"]["Fluid_Sensitive"] == 1)

old_best_sags = sum(1 for s in old_selection.values() if "Sagittal" in s and s["Sagittal"]["priority_score"] == 3)
new_best_sags = sum(1 for s in new_selection.values() if "Sagittal" in s and s["Sagittal"]["priority_score"] == 3)

print("\n" + "="*80)
print("SAGITTAL SEQUENCE SELECTION COMPARISON (N=4,407 Studies):")
print("="*80)
print(f"Old Naive Selection with Fluid_Sensitive=1: {old_fluid_sags} ({old_fluid_sags/len(new_selection)*100:.1f}%)")
print(f"New Smart Selection with Fluid_Sensitive=1: {new_fluid_sags} ({new_fluid_sags/len(new_selection)*100:.1f}%)")
print(f"Improvement: +{new_fluid_sags - old_fluid_sags} studies upgraded to fluid-sensitive!")
print()
print(f"Old Selection with BOTH Fluid=1 and FatSup=1: {old_best_sags} ({old_best_sags/len(new_selection)*100:.1f}%)")
print(f"New Selection with BOTH Fluid=1 and FatSup=1: {new_best_sags} ({new_best_sags/len(new_selection)*100:.1f}%)")
print(f"Improvement: +{new_best_sags - old_best_sags} studies upgraded to Gold-Standard T2/PD FS!")

# Coronal comparison
old_fluid_cor = sum(1 for s in old_selection.values() if "Coronal" in s and s["Coronal"]["priority_score"] == 3)
new_fluid_cor = sum(1 for s in new_selection.values() if "Coronal" in s and s["Coronal"]["priority_score"] == 3)
print("\n" + "="*80)
print("CORONAL SEQUENCE SELECTION COMPARISON (MCL & Collateral Quality):")
print("="*80)
print(f"Old Selection with Both Fluid=1 & FatSup=1: {old_fluid_cor} ({old_fluid_cor/len(new_selection)*100:.1f}%)")
print(f"New Selection with Both Fluid=1 & FatSup=1: {new_fluid_cor} ({new_fluid_cor/len(new_selection)*100:.1f}%)")
print(f"Improvement: +{new_fluid_cor - old_fluid_cor} studies upgraded!")
