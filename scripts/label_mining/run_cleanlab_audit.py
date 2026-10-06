"""Run Cleanlab Confident Learning across all 12 targets on E11 Fold 0 validation predictions.

Outputs:
1. Summary table of estimated label errors per target.
2. Filtered list of highest-confidence label errors (NLP mistakes vs true image pathology).
3. Saves flagged study IDs to data/processed/cleanlab_flagged_e11_fold0.csv
"""
import sys
import io
import os
import pandas as pd
import numpy as np
from cleanlab.filter import find_label_issues

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def main():
    val_preds_path = "e11_output/val_preds_fold_0.csv"
    train_csv_path = "data/raw/train.csv"
    out_csv = "data/processed/cleanlab_flagged_e11_fold0.csv"
    
    if not os.path.exists(val_preds_path):
        print(f"File not found: {val_preds_path}")
        return
        
    df = pd.read_csv(val_preds_path)
    train_df = pd.read_csv(train_csv_path)[["StudyInstanceUID", "Report"]].dropna()
    df = pd.merge(df, train_df, on="StudyInstanceUID", how="left")
    
    print(f"Loaded {len(df)} validation studies from E11 Fold 0.")
    print(f"Gold studies: {df['is_gold'].sum()} | Silver studies: {(~df['is_gold']).sum()}")

    targets = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    
    summary_records = []
    all_issues = []

    for t in targets:
        y_jev = df[t].values
        y_pred = df[f"pred_{t}"].values
        
        # Binarize silver at 0.5 threshold
        y_bin = (y_jev >= 0.5).astype(int)
        
        # Cleanlab 2D prediction probabilities: [P(class=0), P(class=1)]
        pred_probs = np.vstack([1.0 - y_pred, y_pred]).T
        
        # Confident learning to find label issues
        issues_idx = find_label_issues(
            labels=y_bin,
            pred_probs=pred_probs,
            filter_by="prune_by_noise_rate",
            return_indices_ranked_by="self_confidence",
            n_jobs=1
        )
        
        summary_records.append({
            "Target": t,
            "Total Studies": len(df),
            "Estimated Label Issues": len(issues_idx),
            "Issue Rate (%)": round(len(issues_idx) / len(df) * 100, 2)
        })
        
        for rank, idx in enumerate(issues_idx):
            row = df.iloc[idx]
            all_issues.append({
                "Target": t,
                "Rank": rank + 1,
                "StudyInstanceUID": row["StudyInstanceUID"],
                "is_gold": row["is_gold"],
                "Report_Label": row[t],
                "Model_Pred": round(row[f"pred_{t}"], 4),
                "Abs_Diff": round(abs(row[t] - row[f"pred_{t}"]), 4),
                "Report_Excerpt": str(row.get("Report", ""))[:180]
            })

    summary_df = pd.DataFrame(summary_records).sort_values(by="Estimated Label Issues", ascending=False)
    
    print("\n" + "="*70)
    print("CLEANLAB ESTIMATED LABEL ERRORS PER TARGET (E11 Fold 0 OOF)")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    issues_df = pd.DataFrame(all_issues)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    issues_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(issues_df)} flagged label issues to {out_csv}")
    
    # Showcase top 3 highest-confidence label flips for Lateral Meniscus, Synovitis, Effusion
    for focus in ["Lateral Meniscus", "Synovitis", "Effusion"]:
        f_df = issues_df[issues_df["Target"] == focus].head(3)
        print(f"\nTop Glaring Issues for {focus}:")
        for _, r in f_df.iterrows():
            print(f"  Study: {r['StudyInstanceUID'][:25]}... (Gold={r['is_gold']})")
            print(f"  Given Label: {r['Report_Label']} vs Model Pred: {r['Model_Pred']}")
            print(f"  Text: {r['Report_Excerpt']}...")

if __name__ == "__main__":
    main()
