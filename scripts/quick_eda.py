"""Quick EDA script to understand the competition data structure."""
import pandas as pd

# Load data
train = pd.read_csv("data/raw/train.csv")
train_series = pd.read_csv("data/raw/train_series.csv")
test = pd.read_csv("data/raw/test.csv")
test_series = pd.read_csv("data/raw/test_series.csv")
sample_sub = pd.read_csv("data/raw/sample_submission.csv")

TARGET_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
               "Medial OA", "Lateral OA", "PF OA", "Effusion",
               "Synovitis", "Baker's", "Contusion", "Fracture"]

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Training studies: {len(train)}")
print(f"Test studies: {len(test)}")
print(f"Train series (total): {len(train_series)}")
print(f"Test series (total): {len(test_series)}")

# Gold-labeled studies
labeled = train.dropna(subset=["ACL"])
unlabeled = train[train["ACL"].isna()]
print(f"\nGold-labeled studies: {len(labeled)}")
print(f"Unlabeled studies (report only): {len(unlabeled)}")

print("\n" + "=" * 60)
print("TRAIN.CSV COLUMNS")
print("=" * 60)
print(train.columns.tolist())

print("\n" + "=" * 60)
print("TRAIN_SERIES.CSV COLUMNS")
print("=" * 60)
print(train_series.columns.tolist())
print(f"\nAnatomical planes: {train_series['Anatomical_Plane'].value_counts().to_dict()}")
print(f"Fluid sensitive: {train_series['Fluid_Sensitive'].value_counts().to_dict()}")
print(f"Fat suppression: {train_series['Fat_Suppression'].value_counts().to_dict()}")

# Series per study
series_per_study = train_series.groupby("StudyInstanceUID").size()
print(f"\nSeries per study — min: {series_per_study.min()}, max: {series_per_study.max()}, "
      f"median: {series_per_study.median()}, mean: {series_per_study.mean():.1f}")

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION (58 GOLD-LABELED STUDIES)")
print("=" * 60)
for col in TARGET_COLS:
    pos = int(labeled[col].sum())
    neg = int((labeled[col] == 0).sum())
    print(f"  {col:20s} — positive: {pos:3d}, negative: {neg:3d}, prevalence: {pos/len(labeled)*100:.1f}%")

print("\n" + "=" * 60)
print("SAMPLE GOLD-LABELED ROW")
print("=" * 60)
row = labeled.iloc[0]
print(f"StudyUID: {row['StudyInstanceUID']}")
print(f"Report (first 300 chars): {str(row['Report'])[:300]}")
print(f"Labels: {row[TARGET_COLS].to_dict()}")

print("\n" + "=" * 60)
print("SAMPLE REPORTS (MULTILINGUAL)")
print("=" * 60)
for i in range(min(5, len(unlabeled))):
    row = unlabeled.iloc[i]
    report = str(row["Report"])[:200]
    print(f"\n--- Study {i+1} ---")
    print(f"UID: {row['StudyInstanceUID']}")
    print(f"Report: {report}...")

print("\n" + "=" * 60)
print("TEST DATA")
print("=" * 60)
print(f"Test columns: {test.columns.tolist()}")
print(f"Sample submission columns: {sample_sub.columns.tolist()}")
print(f"Test studies: {len(test)}")
print(f"NOTE: test.csv has NO 'Report' column — model must work from images only at inference!")
