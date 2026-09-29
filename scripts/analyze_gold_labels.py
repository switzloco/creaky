import io, gzip, base64, re
import pandas as pd
import numpy as np

with open("scripts/training/train_kaggle_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

m = re.search(r'EMBEDDED_JEV_LABELS_B64\s*=\s*"([^"]+)"', content)
b64 = m.group(1)
raw = gzip.decompress(base64.b64decode(b64))
df = pd.read_csv(io.BytesIO(raw))

gold_df = df[df["label_source"] == "gold"]
print(f"Total Gold Studies: {len(gold_df)}")

# Also let's check fallback regex extractor vs gold on these 58 studies if reports are available
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

print("\nGold target positive counts (out of 58):")
for t in TARGETS:
    pos = (gold_df[t] == 1.0).sum()
    neg = (gold_df[t] == 0.0).sum()
    other = len(gold_df) - pos - neg
    print(f"  {t:18s}: {pos:2d} positive, {neg:2d} negative ({other} other)")
