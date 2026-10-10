import pandas as pd
import numpy as np
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Check e18 val preds and e11 val preds
for val_path, name in [
    ("checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv", "E18 Cleanlab"),
    ("e11_output/val_preds_fold_0.csv", "E11 Report-Aux")
]:
    print(f"\n=================== {name} ({val_path}) ===================")
    df = pd.read_csv(val_path)
    print(f"Total rows: {len(df)}, Gold rows: {(df['is_gold'] == True).sum()}, Silver rows: {(df['is_gold'] == False).sum()}")
    
    # Analyze Gold errors:
    gold = df[df["is_gold"] == True].copy()
    gold_errors = []
    for t in TARGET_COLS:
        y_true = gold[t].values
        y_pred = gold[f"pred_{t}"].values
        diff = np.abs(y_true - y_pred)
        for uid, yt, yp, d in zip(gold["StudyInstanceUID"], y_true, y_pred, diff):
            if d >= 0.5: # Major error
                err_type = "False Positive" if yp > yt else "False Negative"
                gold_errors.append({
                    "UID": uid,
                    "Target": t,
                    "True": yt,
                    "Pred": yp,
                    "Diff": d,
                    "Type": err_type,
                    "Split": "Gold"
                })
    err_df = pd.DataFrame(gold_errors).sort_values("Diff", ascending=False)
    print(f"Total Gold Major Discrepancies (|y - pred| >= 0.5): {len(err_df)}")
    print("Gold major errors by target:")
    print(err_df["Target"].value_counts().to_string())
    print("\nTop 15 worst Gold errors:")
    print(err_df.head(15)[["Target", "Type", "True", "Pred", "Diff", "UID"]].to_string())

