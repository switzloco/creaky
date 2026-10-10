import io
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

kernel_slug = "nswitzer/creaky-0-94-ensemble"
print(f"Monitoring kernel: {kernel_slug} for completion...", flush=True)

while True:
    status_obj = api.kernels_status(kernel_slug)
    status_str = str(status_obj.status).upper()
    failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
    print(f"[{time.strftime('%H:%M:%S')}] Kernel status: {status_str} | Failure: {failure}", flush=True)
    
    if "COMPLETE" in status_str:
        print("\n=== Kernel Version 5 Run Complete! Submitting to RSNA Competition ===")
        res = api.competition_submit_code(
            file_name="submission.csv",
            message="Version 5: E18 Cleanlab-pruned Fold 0 (0.8597 Macro AUC record) + F1 + F2 + CoAtNet 0.943 Hybrid Blend",
            competition="rsna-knee-abnormality-detection",
            kernel=kernel_slug,
            kernel_version=5
        )
        print("Submission response:", res)
        break
    elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
        print(f"Kernel failed with status: {status_str}, failure: {failure}")
        sys.exit(1)
        
    time.sleep(30)

