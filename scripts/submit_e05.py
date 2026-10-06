import io
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

res = api.competition_submit_code(
    file_name="submission.csv",
    message="E05: Solo Phase 2 Fold 0 Ablation",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/submission-notebook",
    kernel_version=11
)

print("Submission response:", res)
