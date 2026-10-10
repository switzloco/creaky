import pandas as pd
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

df = pd.read_csv("checkpoints/e18_cleanlab_trained/val_preds_fold_0.csv")

cases = [
    ("Case 1 (ACL)", "1.2.826.0.1.3680043.8.498.26790702379506190936834447203448882465"),
    ("Case 2 (Lateral Meniscus)", "1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269"),
    ("Case 3 (ACL + MM)", "1.2.826.0.1.3680043.8.498.11382021393803389951964005983002209238"),
    ("Case 4 (Contusion + Effusion)", "1.2.826.0.1.3680043.8.498.11548045715264151632153040089882701935"),
    ("Case 5 (PF OA)", "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149")
]

for label, uid in cases:
    row = df[df["StudyInstanceUID"] == uid]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n=== {label} (UID: {uid}) ===")
        print(f"is_gold: {r['is_gold']}")
        for t in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']:
            yt = r[t]
            yp = r[f"pred_{t}"]
            flag = " **MISMATCH**" if abs(yt - yp) >= 0.4 else ""
            print(f"  {t:18s} | True: {yt} | Pred: {yp:.4f}{flag}")
    else:
        print(f"UID {uid} not in val_preds_fold_0.csv")

