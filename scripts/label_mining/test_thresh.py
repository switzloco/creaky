import pandas as pd

cl_df = pd.read_csv('data/processed/cleanlab_flagged_e11_fold0.csv')
silver_issues = cl_df[cl_df['is_gold'] == False]

for thresh in [0.4, 0.5, 0.6, 0.7]:
    subset = silver_issues[silver_issues['Abs_Diff'] >= thresh]
    print(f"Threshold >= {thresh}: {len(subset)} issues across {subset['StudyInstanceUID'].nunique()} studies")
    print(subset['Target'].value_counts()[:5])
    print("-" * 40)
