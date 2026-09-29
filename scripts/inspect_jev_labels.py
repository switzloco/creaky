import re
import gzip
import base64
import io
import pandas as pd

with open("scripts/training/train_kaggle_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

m = re.search(r'EMBEDDED_JEV_LABELS_B64\s*=\s*"([^"]+)"', content)
if m:
    b64 = m.group(1)
    raw = gzip.decompress(base64.b64decode(b64))
    df = pd.read_csv(io.BytesIO(raw))
    print(f"Jev labels shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nLabel sources:", df["label_source"].value_counts().to_dict() if "label_source" in df.columns else "no label_source")
    
    # check weight columns
    weight_cols = [c for c in df.columns if "weight" in c]
    print(f"Weight cols found: {len(weight_cols)}")
    if weight_cols:
        print(df[weight_cols].describe().T[["min", "mean", "max"]])
    
    # check missing values
    targets = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    print("\nNull counts across targets:")
    print(df[targets].isna().sum())
    
    # check how many are 0.5 or unmentioned
    print("\nValue distributions (0.0 vs 1.0 vs others) for ACL:")
    print(df["ACL"].value_counts(dropna=False).head(5))
else:
    print("EMBEDDED_JEV_LABELS_B64 not found via regex")
