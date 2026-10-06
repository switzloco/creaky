import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = [
    'ACL Tear',
    'Anterior Cruciate Ligament Tear',
    'Full-Thickness Cartilage Defect',
    'Full-Thickness Tear',
    'Lateral Meniscus Tear',
    'Medial Meniscus Tear',
    'Meniscal Tear',
    'Partial-Thickness Cartilage Defect',
    'Partial-Thickness Tear',
    'Patellar Tendon Tear',
    'Posterior Cruciate Ligament Tear',
    'Quadriceps Tendon Tear'
]

def eval_fold(csv_path, fold_name):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print(f"\n=================== {fold_name} ({len(df)} rows) ===================")
    print("Columns:", df.columns.tolist()[:15])
    
    targets = [c for c in df.columns if not c.startswith('pred_') and c not in ['StudyInstanceUID', 'is_gold', 'fold']]
    print("Found targets:", targets)
    
    for split_name, sub_df in [('Overall (Silver+Gold)', df), ('Gold Only', df[df['is_gold'] == 1])]:
        print(f"\n--- {split_name} (N={len(sub_df)}) ---")
        aucs = []
        for t in targets:
            pred_col = f"pred_{t}"
            if pred_col in sub_df.columns:
                y_true = sub_df[t].values
                y_pred = sub_df[pred_col].values
                # Binarize if continuous
                y_bin = (y_true > 0.5).astype(int)
                if len(np.unique(y_bin)) > 1:
                    auc = roc_auc_score(y_bin, y_pred)
                    aucs.append(auc)
                    print(f"  {t:<25}: AUC = {auc:.4f} (pos={int(y_bin.sum())}/{len(y_bin)})")
        if aucs:
            print(f"  --> Macro AUC ({len(aucs)} targets): {np.mean(aucs):.4f}")

if __name__ == '__main__':
    eval_fold('checkpoints/e04_trained/val_preds_fold_0.csv', 'Fold 0 (E04 Baseline)')
    eval_fold('checkpoints/e06_trained/val_preds_fold_0.csv', 'Fold 0 (E11 Report-Aux)')
    eval_fold('checkpoints/e13_trained/val_preds_fold_1.csv', 'Fold 1 (E13 Report-Aux)')
    eval_fold('checkpoints/e14_trained/val_preds_fold_2.csv', 'Fold 2 (E14 Report-Aux)')
