"""Unit tests for Phase 4 Kaggle Submission and Validation Pipeline."""

import os
import sys
import unittest
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.submission.validate_submission import validate_submission_file, EXPECTED_COLUMNS
from scripts.submission.submission_notebook import run_submission_inference


class TestSubmissionPipeline(unittest.TestCase):

    def test_expected_submission_columns(self):
        """Ensure expected columns match the 13 required competition fields."""
        self.assertEqual(len(EXPECTED_COLUMNS), 13)
        self.assertEqual(EXPECTED_COLUMNS[0], "StudyInstanceUID")
        self.assertIn("ACL", EXPECTED_COLUMNS)
        self.assertIn("Fracture", EXPECTED_COLUMNS)

    def test_submission_generator_and_validator(self):
        """Test generating a submission and verifying that it passes all Kaggle checks."""
        test_csv = os.path.join(root_dir, "data", "raw", "test.csv")
        if not os.path.exists(test_csv):
            self.skipTest("test.csv not found")

        temp_sub_path = os.path.join(root_dir, "tests", "test_submission.csv")
        sub_df = run_submission_inference(output_path=temp_sub_path)

        self.assertTrue(os.path.exists(temp_sub_path))
        is_valid = validate_submission_file(temp_sub_path, test_csv)
        self.assertTrue(is_valid)

        # Cleanup
        if os.path.exists(temp_sub_path):
            os.remove(temp_sub_path)


if __name__ == "__main__":
    unittest.main()
