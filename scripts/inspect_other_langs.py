"""Inspect other language samples."""
import pandas as pd

train = pd.read_csv("data/raw/train.csv")

for i in range(10, 30):
    report = str(train.iloc[i]["Report"])
    print(f"\n--- Index {i} ---")
    print(report[:250])
