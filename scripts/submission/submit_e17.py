import io
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

print("Submitting Version 4 of creaky-0-94-ensemble (E17: 3-Fold ReportAux ConvNeXt + CoAtNet)...")
res = api.competition_submit_code(
    file_name="submission.csv",
    message="E17: 3-Fold ReportAux ConvNeXt (F0+F1+F2) + CoAtNet 0.943 Hybrid Blend",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-0-94-ensemble",
    kernel_version=4
)

print("Submission response:", res)
