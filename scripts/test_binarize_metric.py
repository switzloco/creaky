import io, gzip, base64, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

with open("scripts/training/train_kaggle_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

m = re.search(r'EMBEDDED_JEV_LABELS_B64\s*=\s*"([^"]+)"', content)
b64 = m.group(1)
raw = gzip.decompress(base64.b64decode(b64))
df = pd.read_csv(io.BytesIO(raw))

TARGET_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Test on 434 silver validation studies (first 10%)
silver_df = df[df["label_source"] == "jev"].reset_index(drop=True)
val_silver = silver_df.iloc[:434]
y_true = val_silver[TARGET_COLS].to_numpy(dtype=np.float32)

print("--- With yt >= 0.5 Binarization ---")
for i, col in enumerate(TARGET_COLS):
    yt = y_true[:, i]
    yt_bin = (yt >= 0.5).astype(int)
    pos = (yt_bin == 1).sum()
    neg = (yt_bin == 0).sum()
    print(f"  {col:18s}: pos {pos:3d} / neg {neg:3d} (Classes: {np.unique(yt_bin)})")
