import pandas as pd
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for path in [
    "data/processed/cleanlab_flagged_e11_fold0.csv",
    "data/processed/cleanlab_flagged_multifold.csv",
    "data/processed/jev_vs_gold_audit.csv"
]:
    try:
        df = pd.read_csv(path)
        print(f"=== {path} (shape: {df.shape}) ===")
        print(df.head(3))
        print("Columns:", list(df.columns))
    except Exception as e:
        print(f"Error {path}: {e}")
