import io
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

kernel_slug = "nswitzer/creaky-apex-cleanlab-blend"
print(f"Monitoring kernel: {kernel_slug} for completion...", flush=True)

while True:
    status_obj = api.kernels_status(kernel_slug)
    status_str = str(status_obj.status).upper()
    failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
    print(f"[{time.strftime('%H:%M:%S')}] Kernel status: {status_str} | Failure: {failure}", flush=True)
    
    if "COMPLETE" in status_str:
        print("\n=== Kernel Run Complete! Submitting E19 to RSNA Competition ===", flush=True)
        res = api.competition_submit_code(
            file_name="submission.csv",
            message="E19: Apex Grandmaster V5 (0.952+) + Creaky E18 Cleanlab MoE Blend (Record 0.8597 Macro AUC)",
            competition="rsna-knee-abnormality-detection",
            kernel=kernel_slug,
            kernel_version=1
        )
        print("Submission response:", res, flush=True)
        break
    elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
        print(f"Kernel failed with status: {status_str}, failure: {failure}", flush=True)
        sys.exit(1)
        
    time.sleep(45)
