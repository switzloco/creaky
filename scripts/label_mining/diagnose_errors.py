"""Diagnostic script to inspect regex mistakes on the 58 gold-standard studies."""
import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from regex_labeler import RegexLabeler, TARGET_COLS

sys.stdout.reconfigure(encoding="utf-8")

train_df = pd.read_csv("data/raw/train.csv")
gold_df = train_df.dropna(subset=["ACL"]).copy()

labeler = RegexLabeler()
pred_df = labeler.label_dataframe(gold_df)

merged = pd.merge(gold_df, pred_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))

for target in ["Medial Meniscus", "PF OA", "Effusion", "Synovitis", "Contusion", "Fracture"]:
    print(f"\n{'='*80}")
    print(f"ANALYZING FALSE NEGATIVES & MISSES FOR: {target}")
    print("=" * 80)
    
    # False negatives (True=1, Pred != 1)
    fn_rows = merged[(merged[f"{target}_true"] == 1.0) & (merged[f"{target}_pred"] != 1.0)]
    print(f"Total True Positives missed by regex: {len(fn_rows)}")
    
    for idx, (_, row) in enumerate(fn_rows.head(3).iterrows()):
        print(f"\n--- Study {idx+1} (Pred: {row[f'{target}_pred']}) ---")
        report = str(row['Report'])
        # Print relevant sentences
        sentences = [s.strip() for s in report.split('\n') if len(s.strip()) > 5]
        print("\n".join(sentences[:8]))
