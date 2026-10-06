"""Task T8: Diagnostic extraction of Lateral Meniscus gold vs Jev disagreements.

Extracts all gold studies where:
1. Gold = 1 (Torn), but Jev < 0.5 (False Negative / Missed in text)
2. Gold = 0 (Intact), but Jev >= 0.5 (False Alarm)
Prints clinical report excerpts and checks laterality / phrasing.
"""
import sys
import io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def main():
    train_csv = "data/raw/train.csv"
    jev_csv = "data/processed/jev_encoded_features.csv"
    
    df_train = pd.read_csv(train_csv)
    df_jev = pd.read_csv(jev_csv)
    
    # Gold studies are rows in train.csv with non-null labels in the main label columns
    gold_mask = df_train["Lateral Meniscus"].notna()
    gold_df = df_train[gold_mask].copy()
    print(f"Total Gold studies: {len(gold_df)}")
    
    # Merge with Jev predictions
    merged = pd.merge(
        gold_df[["StudyInstanceUID", "Report", "Lateral Meniscus", "Medial Meniscus"]],
        df_jev[["StudyInstanceUID", "Lateral Meniscus_jev_prob", "Medial Meniscus_jev_prob"]],
        on="StudyInstanceUID"
    )
    
    merged["gold_lat"] = merged["Lateral Meniscus"].astype(float)
    merged["jev_lat"] = merged["Lateral Meniscus_jev_prob"].astype(float)
    merged["jev_pred"] = (merged["jev_lat"] >= 0.5).astype(float)
    
    # False Negatives (Gold=1, Jev=0)
    fn = merged[(merged["gold_lat"] == 1.0) & (merged["jev_pred"] == 0.0)]
    print(f"\n=======================================================")
    print(f"LATERAL MENISCUS FALSE NEGATIVES (Gold=1, Jev=0): {len(fn)}")
    print(f"=======================================================")
    for idx, row in fn.iterrows():
        print(f"\nStudyUID: {row['StudyInstanceUID']}")
        print(f"Gold Lat: {row['gold_lat']} | Jev Lat Prob: {row['jev_lat']:.3f} | Medial Gold: {row['Medial Meniscus']}")
        print(f"Report Excerpt:\n{str(row['Report'])[:400]}...\n")
        
    # False Alarms (Gold=0, Jev=1)
    fp = merged[(merged["gold_lat"] == 0.0) & (merged["jev_pred"] == 1.0)]
    print(f"\n=======================================================")
    print(f"LATERAL MENISCUS FALSE ALARMS (Gold=0, Jev=1): {len(fp)}")
    print(f"=======================================================")
    for idx, row in fp.iterrows():
        print(f"\nStudyUID: {row['StudyInstanceUID']}")
        print(f"Gold Lat: {row['gold_lat']} | Jev Lat Prob: {row['jev_lat']:.3f} | Medial Gold: {row['Medial Meniscus']}")
        print(f"Report Excerpt:\n{str(row['Report'])[:400]}...\n")

if __name__ == "__main__":
    main()
