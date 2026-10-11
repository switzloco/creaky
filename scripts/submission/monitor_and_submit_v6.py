import io
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

kernel_slug = "nswitzer/creaky-apex-cleanlab-blend"
print(f"=== MONITORING APEX GRANDMASTER V6 SUBMISSION PIPELINE ===", flush=True)
print(f"Monitoring kernel: {kernel_slug} (Target-Adaptive Class Weights Blend)...", flush=True)

consecutive_errors = 0
while True:
    try:
        status_obj = api.kernels_status(kernel_slug)
        status_str = str(status_obj.status).upper()
        failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
        print(f"[{time.strftime('%H:%M:%S')}] Kernel status: {status_str} | Failure: {failure}", flush=True)
        consecutive_errors = 0
        
        if "COMPLETE" in status_str:
            print("\n=== Kernel Run Complete! Submitting Apex Grandmaster V6 to RSNA Competition ===", flush=True)
            res = api.competition_submit_code(
                file_name="submission.csv",
                message="E21: Apex Grandmaster V6 True Synergy Blend (Target-Adaptive Class Weights + Cleanlab E18 Ensemble)",
                competition="rsna-knee-abnormality-detection",
                kernel=kernel_slug,
                kernel_version=3
            )
            print("Submission response:", res, flush=True)
            print("Successfully submitted Apex V6 to competition leaderboard!", flush=True)
            break
        elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
            print(f"Kernel failed with status: {status_str}, failure: {failure}", flush=True)
            sys.exit(1)
            
    except Exception as e:
        consecutive_errors += 1
        print(f"[{time.strftime('%H:%M:%S')}] Transient status check error ({consecutive_errors}/10): {e}", flush=True)
        if consecutive_errors >= 10:
            print("Too many consecutive network errors. Exiting.")
            sys.exit(1)
            
    time.sleep(45)
