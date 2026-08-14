"""Automated Submission Validator for RSNA Knee Abnormality Detection.

Validates that submission.csv strictly obeys all Kaggle submission rules.
"""

import os
import sys
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    "StudyInstanceUID", "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


def validate_submission_file(sub_csv_path: str, test_csv_path: str, sample_sub_path: str = None) -> bool:
    """Check submission.csv against all validation rules."""
    print("=" * 70)
    print(f"VALIDATING SUBMISSION FILE: {sub_csv_path}")
    print("=" * 70)

    # 1. Existence check
    if not os.path.exists(sub_csv_path):
        print(f"[FAIL] Submission file does not exist at: {sub_csv_path}")
        return False
    print("[PASS] File exists.")

    sub_df = pd.read_csv(sub_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # 2. Row count check
    if len(sub_df) != len(test_df):
        print(f"[FAIL] Row count mismatch! Expected {len(test_df)}, got {len(sub_df)}")
        return False
    print(f"[PASS] Row count matches test set ({len(sub_df)} rows).")

    # 3. Column names and ordering check
    if list(sub_df.columns) != EXPECTED_COLUMNS:
        print("[FAIL] Column names or ordering mismatch!")
        print(f"  Expected: {EXPECTED_COLUMNS}")
        print(f"  Actual:   {list(sub_df.columns)}")
        return False
    print("[PASS] Columns and ordering match exactly.")

    # 4. StudyInstanceUID matching
    test_uids = set(test_df["StudyInstanceUID"].astype(str))
    sub_uids = set(sub_df["StudyInstanceUID"].astype(str))
    if test_uids != sub_uids:
        print("[FAIL] StudyInstanceUID set does not match test.csv!")
        return False
    print("[PASS] StudyInstanceUIDs match test.csv 1-to-1.")

    # 5. Check for NaNs, Infs, and Null values
    if sub_df.isna().any().any():
        print("[FAIL] Submission contains NaN or Null values!")
        print(sub_df.isna().sum())
        return False
    print("[PASS] Zero NaN or Null values detected.")

    # 6. Check probability ranges [0.0, 1.0]
    for col in EXPECTED_COLUMNS[1:]:
        vals = sub_df[col].values
        if not np.issubdtype(vals.dtype, np.number):
            print(f"[FAIL] Column '{col}' contains non-numeric values!")
            return False
        if (vals < 0.0).any() or (vals > 1.0).any():
            print(f"[FAIL] Column '{col}' contains values outside [0.0, 1.0] range!")
            return False
    print("[PASS] All prediction probabilities are valid numbers in [0.0, 1.0].")

    print("\n" + "=" * 70)
    print("ALL SUBMISSION VALIDATION CHECKS PASSED! READY FOR LEADERBOARD.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sub_file = "submission.csv" if len(sys.argv) < 2 else sys.argv[1]
    test_file = "data/raw/test.csv"
    sample_sub = "data/raw/sample_submission.csv"
    success = validate_submission_file(sub_file, test_file, sample_sub)
    if not success:
        sys.exit(1)
