import io
import sys
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

cl_df = pd.read_csv('data/processed/cleanlab_flagged_e11_fold0.csv')
print("Total Cleanlab flagged issues:", len(cl_df))
print("Unique studies in flagged issues:", cl_df['StudyInstanceUID'].nunique())
print("\nIs Gold breakdown:")
print(cl_df['is_gold'].value_counts())

print("\nIssue counts by target:")
print(cl_df['Target'].value_counts())

print("\nAbs_Diff summary statistics:")
print(cl_df['Abs_Diff'].describe())

print("\nTop 10 highest-discrepancy label errors:")
for idx, row in cl_df.sort_values(by='Abs_Diff', ascending=False).head(10).iterrows():
    print(f"Target: {row['Target']:<18} | UID: {row['StudyInstanceUID'][:20]}... | Report_Label: {row['Report_Label']} | Pred: {row['Model_Pred']:.4f} | Diff: {row['Abs_Diff']:.4f}")
