import sys
import io
import os
import pandas as pd
import numpy as np
from cleanlab.filter import find_label_issues

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def run_cleanlab_on_preds(csv_paths, out_csv="data/processed/cleanlab_flagged_multifold.csv"):
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"Loaded {len(df)} rows from {p}")
            dfs.append(df)
        else:
            print(f"Warning: {p} not found")
    
    if not dfs:
        return
        
    combined_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate in case any study appears in multiple (e.g. gold studies appear in all validation sets)
    combined_df = combined_df.drop_duplicates(subset=["StudyInstanceUID"]).reset_index(drop=True)
    print(f"\nTotal unique studies with out-of-fold predictions: {len(combined_df)}")
    print(f"Gold: {combined_df['is_gold'].sum()} | Silver: {(~combined_df['is_gold']).sum()}")
    
    targets = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    
    all_issues = []
    
    for t in targets:
        y_jev = combined_df[t].values
        y_pred = combined_df[f"pred_{t}"].values
        
        # Binarize silver at 0.5 threshold
        y_bin = (y_jev >= 0.5).astype(int)
        
        # Cleanlab 2D prediction probabilities
        pred_probs = np.vstack([1.0 - y_pred, y_pred]).T
        
        issues_idx = find_label_issues(
            labels=y_bin,
            pred_probs=pred_probs,
            filter_by="prune_by_noise_rate",
            return_indices_ranked_by="self_confidence",
            n_jobs=1
        )
        
        for rank, idx in enumerate(issues_idx):
            row = combined_df.iloc[idx]
            diff = abs(y_jev[idx] - y_pred[idx])
            all_issues.append({
                "Target": t,
                "Rank": rank + 1,
                "StudyInstanceUID": row["StudyInstanceUID"],
                "is_gold": bool(row["is_gold"]),
                "Report_Label": float(y_jev[idx]),
                "Model_Pred": float(y_pred[idx]),
                "Abs_Diff": float(diff)
            })
            
    issues_df = pd.DataFrame(all_issues)
    print(f"\nTotal Cleanlab issues found across multi-fold OOF: {len(issues_df)}")
    silver_issues = issues_df[issues_df["is_gold"] == False]
    print(f"Silver issues: {len(silver_issues)}")
    high_conf = silver_issues[silver_issues["Abs_Diff"] >= 0.5]
    print(f"High-confidence Silver issues (Abs_Diff >= 0.5): {len(high_conf)} across {high_conf['StudyInstanceUID'].nunique()} studies")
    print("\nIssues by target (Abs_Diff >= 0.5):")
    print(high_conf["Target"].value_counts())
    
    issues_df.to_csv(out_csv, index=False)
    print(f"\nSaved issues to: {out_csv}")
    return issues_df

if __name__ == "__main__":
    csv_paths = [
        "checkpoints/e04_trained/val_preds_fold_0.csv",
        "checkpoints/e13_trained/val_preds_fold_1.csv",
        "checkpoints/e14_trained/val_preds_fold_2.csv"
    ]
    run_cleanlab_on_preds(csv_paths)
