import io
import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def monitor_and_submit():
    api = KaggleApi()
    api.authenticate()

    kernel_slug = "nswitzer/creaky-apex-cleanlab-blend"
    competition_slug = "rsna-knee-abnormality-detection"
    message = "E22: Apex Grandmaster N4 Synergy (384px CoAtNet Anchor + Full 3-Fold Track N4 Suite)"

    print("==================================================================", flush=True)
    print("  MONITORING APEX N4 GRANDMASTER SUBMISSION RUN", flush=True)
    print(f"  Kernel: {kernel_slug}", flush=True)
    print("==================================================================", flush=True)

    start_time = time.time()
    consecutive_errors = 0

    while True:
        try:
            status_obj = api.kernels_status(kernel_slug)
            status_str = str(status_obj.status).upper()
            failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
            elapsed_min = (time.time() - start_time) / 60.0

            print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] Kernel Status: {status_str} | Failure: {failure}", flush=True)
            consecutive_errors = 0

            if "COMPLETE" in status_str:
                print(f"\n--> Kernel run COMPLETE! Submitting to {competition_slug}...", flush=True)
                try:
                    res = api.competition_submit_code(
                        file_name="submission.csv",
                        message=message,
                        competition=competition_slug,
                        kernel=kernel_slug
                    )
                    print(f"Submission response: {res}", flush=True)
                    print("\nSuccessfully submitted E22 to competition leaderboard!", flush=True)
                except Exception as sub_err:
                    print(f"Submission API call error: {sub_err}", flush=True)
                break

            elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
                print(f"Kernel failed with status: {status_str}, error: {failure}", flush=True)
                sys.exit(1)

        except Exception as e:
            consecutive_errors += 1
            print(f"[{time.strftime('%H:%M:%S')}] Transient status check error ({consecutive_errors}/10): {e}", flush=True)
            if consecutive_errors >= 10:
                print("Too many consecutive network errors. Exiting.")
                sys.exit(1)

        time.sleep(45)

if __name__ == "__main__":
    monitor_and_submit()
