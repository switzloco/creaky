"""Inspect the 58 gold-labeled reports to understand terminology across languages."""
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
labeled = train.dropna(subset=["ACL"])

print(f"Total labeled: {len(labeled)}")

target_cols = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

for idx, (_, row) in enumerate(labeled.head(10).iterrows()):
    print(f"\n{'='*70}")
    print(f"Sample {idx+1} | Study UID: {row['StudyInstanceUID']}")
    positives = [col for col in target_cols if row[col] == 1.0]
    negatives = [col for col in target_cols if row[col] == 0.0]
    print(f"POSITIVES: {positives}")
    print(f"NEGATIVES: {negatives}")
    print("-" * 70)
    print(row['Report'][:600])
