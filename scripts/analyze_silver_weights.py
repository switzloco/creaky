import io, gzip, base64, re
import pandas as pd
import numpy as np

with open("scripts/training/train_kaggle_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

m = re.search(r'EMBEDDED_JEV_LABELS_B64\s*=\s*"([^"]+)"', content)
b64 = m.group(1)
raw = gzip.decompress(base64.b64decode(b64))
df = pd.read_csv(io.BytesIO(raw))

silver_df = df[df["label_source"] == "jev"]
print(f"Total Silver Studies: {len(silver_df)}")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

print("\n--- Weight breakdown for Silver ---")
for t in TARGETS:
    w = silver_df[f"{t}_weight"]
    low_w = (w < 0.5).sum()
    med_w = ((w >= 0.5) & (w < 0.8)).sum()
    high_w = (w >= 0.8).sum()
    print(f"  {t:18s}: <0.5 (noisy): {low_w:4d} | 0.5-0.8 (hedged): {med_w:4d} | >=0.8 (confident): {high_w:4d}")

# Check study-level average weights
mean_study_weight = silver_df[[f"{t}_weight" for t in TARGETS]].mean(axis=1)
print(f"\nStudies with avg weight < 0.6: {(mean_study_weight < 0.6).sum()}")
print(f"Studies with avg weight < 0.7: {(mean_study_weight < 0.7).sum()}")
print(f"Studies with avg weight < 0.8: {(mean_study_weight < 0.8).sum()}")
print(f"Lowest average study weight: {mean_study_weight.min():.3f}")
