import io
import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def monitor_caching(kernel_slug="nswitzer/creaky-caching-dataset-n4-384", poll_sec=120):
    api = KaggleApi()
    api.authenticate()

    print("=" * 70, flush=True)
    print(f"  MONITORING CACHING KERNEL: {kernel_slug}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    consecutive_errs = 0

    while True:
        try:
            status_obj = api.kernels_status(kernel_slug)
            consecutive_errs = 0
            raw_status = getattr(status_obj, "status", None) or getattr(status_obj, "_status", "")
            status = str(raw_status).upper()
            failure_msg = getattr(status_obj, "failure_message", None) or getattr(status_obj, "_failure_message", None)
            elapsed_min = (time.time() - start_time) / 60.0

            print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] Status: {status} | FailureMsg: {failure_msg}", flush=True)

            if "COMPLETE" in status:
                print(f"\n🎉 CACHING COMPLETE in {elapsed_min:.1f} minutes!", flush=True)
                break
            elif any(x in status for x in ["ERROR", "CANCELLED", "FAILED"]):
                print(f"\n[!] Caching kernel halted with status {status}: {failure_msg}", flush=True)
                break
        except Exception as e:
            consecutive_errs += 1
            print(f"[{time.strftime('%H:%M:%S')}] Network/API Warning ({consecutive_errs}): {e}", flush=True)
            if consecutive_errs > 30:
                print("Too many consecutive network errors, exiting monitor.", flush=True)
                break

        time.sleep(poll_sec)

if __name__ == "__main__":
    monitor_caching()
