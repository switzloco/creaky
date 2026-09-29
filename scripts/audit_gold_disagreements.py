import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, ".")

# Import MultilingualRegexLabeler from training script
from scripts.training.train_kaggle_notebook import MultilingualRegexLabeler, TARGET_COLS

df = pd.read_csv("data/raw/train.csv")
gold_df = df[df["ACL"].notna()].copy()
print(f"Total Gold Studies: {len(gold_df)}")

labeler = MultilingualRegexLabeler()

# Run labeler on all 58 gold reports
results = []
for idx, row in gold_df.iterrows():
    report = row["Report"]
    preds = labeler.extract_from_report(report)
    res_row = {"StudyInstanceUID": row["StudyInstanceUID"], "Report": report}
    for t in TARGET_COLS:
        res_row[f"gold_{t}"] = float(row[t])
        res_row[f"pred_{t}"] = preds.get(t)  # float or None
    results.append(res_row)

audit_df = pd.DataFrame(results)

print("\n" + "="*80)
print("PHASE 3 LABEL AUDIT: REGEX EXTRACTION VS 58 EXPERT GOLD LABELS")
print("="*80)

summary_rows = []
for t in TARGET_COLS:
    g = audit_df[f"gold_{t}"].to_numpy()
    p = audit_df[f"pred_{t}"].to_numpy()
    
    # Coverage: how many had a definitive 0.0 or 1.0 prediction vs None
    covered = ~pd.isna(p)
    cov_pct = covered.sum() / len(g) * 100
    
    # For covered samples, calculate accuracy and agreement
    g_cov = g[covered]
    p_cov = p[covered].astype(float)
    correct = (g_cov == p_cov).sum()
    acc_pct = (correct / len(g_cov) * 100) if len(g_cov) > 0 else 0.0
    
    # False positives and False negatives
    fp = ((p_cov == 1.0) & (g_cov == 0.0)).sum()
    fn = ((p_cov == 0.0) & (g_cov == 1.0)).sum()
    
    # Unmentioned (None) that were positive in Gold
    missed_pos = ((pd.isna(p)) & (g == 1.0)).sum()
    missed_neg = ((pd.isna(p)) & (g == 0.0)).sum()
    
    # AUC on binary predictions (using 0.5 for unmentioned)
    p_filled = np.where(pd.isna(p), 0.5, p.astype(float))
    try:
        auc = roc_auc_score(g, p_filled)
    except Exception:
        auc = float("nan")
        
    summary_rows.append({
        "Target": t,
        "Gold Pos": int((g == 1.0).sum()),
        "Coverage": f"{cov_pct:.1f}%",
        "Agreement": f"{acc_pct:.1f}%",
        "FP (Spurious)": fp,
        "FN (Missed)": fn,
        "None (was Pos)": missed_pos,
        "AUC": f"{auc:.4f}"
    })

sum_df = pd.DataFrame(summary_rows)
print(sum_df.to_string(index=False))

# Show top 5 most striking disagreements
print("\n" + "="*80)
print("INSPECTING SPECIFIC DISAGREEMENTS FOR WORST TARGETS")
print("="*80)

for t in ["Synovitis", "PF OA", "MCL", "Contusion"]:
    print(f"\n--- Disagreements on {t} ---")
    count = 0
    for idx, row in audit_df.iterrows():
        g_val = row[f"gold_{t}"]
        p_val = row[f"pred_{t}"]
        if (p_val is not None and p_val != g_val) or (p_val is None and g_val == 1.0):
            count += 1
            reason = "False Positive" if p_val == 1.0 else ("False Negative" if p_val == 0.0 else "Unmentioned Positive")
            print(f"[{count}] Study: {row['StudyInstanceUID'][:15]}... | Gold: {g_val} | Pred: {p_val} ({reason})")
            # Print relevant sentences from report
            rep = str(row["Report"])
            lines = [l.strip() for l in rep.split("\n") if l.strip()]
            for l in lines[:4]:
                print(f"    Report line: {l}")
            if count >= 3:
                break
