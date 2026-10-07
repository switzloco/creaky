import io
import sys
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def apply_cleanlab_filtering(
    labels_path="data/processed/train_labels.csv",
    issues_path="data/processed/cleanlab_flagged_multifold.csv",
    out_path="data/processed/train_labels_cleanlab.csv",
    diff_thresh=0.5
):
    labels_df = pd.read_csv(labels_path)
    issues_df = pd.read_csv(issues_path)
    
    print(f"Loaded master labels: {len(labels_df)} studies")
    print(f"Loaded Cleanlab issues: {len(issues_df)} issues")
    
    # Strictly filter for Silver issues with high confidence (Abs_Diff >= diff_thresh)
    silver_issues = issues_df[(issues_df["is_gold"] == False) & (issues_df["Abs_Diff"] >= diff_thresh)]
    print(f"Applying zero-weighting to {len(silver_issues)} high-confidence Silver issues (Abs_Diff >= {diff_thresh})...")
    
    # Create lookup set: (StudyInstanceUID, Target)
    flagged_pairs = set(zip(silver_issues["StudyInstanceUID"].astype(str), silver_issues["Target"]))
    
    zeroed_counts = {t: 0 for t in issues_df["Target"].unique()}
    
    for idx, row in labels_df.iterrows():
        uid = str(row["StudyInstanceUID"])
        is_gold = row.get("label_source") == "gold"
        if is_gold:
            continue  # NEVER touch Gold expert labels
            
        for t in zeroed_counts.keys():
            if (uid, t) in flagged_pairs:
                labels_df.at[idx, f"{t}_weight"] = 0.0
                zeroed_counts[t] += 1
                
    print("\nZero-weighted targets count:")
    for t, cnt in sorted(zeroed_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<20}: {cnt} studies zero-weighted")
        
    total_zeroed = sum(zeroed_counts.values())
    print(f"\nTotal zeroed labels: {total_zeroed}")
    
    # Verify gold labels intact
    gold_mask = labels_df["label_source"] == "gold"
    for t in zeroed_counts.keys():
        gold_weights = labels_df.loc[gold_mask, f"{t}_weight"].values
        assert (gold_weights == 1.0).all(), f"Error: Gold weights corrupted for {t}!"
    print("Verification PASSED: All 58 Gold studies retained strict 1.0 weight.")
    
    labels_df.to_csv(out_path, index=False)
    print(f"Successfully saved cleanlab-filtered labels to: {out_path}")
    return labels_df

if __name__ == "__main__":
    apply_cleanlab_filtering()
