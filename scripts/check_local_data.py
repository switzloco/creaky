import os
import pandas as pd

raw_train_p = "data/raw/train.csv"
if os.path.exists(raw_train_p):
    df = pd.read_csv(raw_train_p)
    print("Found data/raw/train.csv with shape:", df.shape)
    print("Columns:", list(df.columns))
    # Check how many have reports
    if "Report" in df.columns:
        print("Non-null reports:", df["Report"].notna().sum())
    # Check how many have gold labels (e.g. non-null ACL)
    gold_mask = df["ACL"].notna() if "ACL" in df.columns else False
    print("Gold studies in train.csv:", gold_mask.sum())
    if "Report" in df.columns:
        gold_reports = df[gold_mask]["Report"].notna().sum()
        print(f"Gold studies with reports: {gold_reports}/{gold_mask.sum()}")
else:
    print("data/raw/train.csv does not exist locally.")
