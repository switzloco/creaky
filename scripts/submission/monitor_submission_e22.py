import io
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def monitor_sub(sub_ref=57189444):
    api = KaggleApi()
    api.authenticate()

    print(f"==================================================================", flush=True)
    print(f"  MONITORING SUBMISSION {sub_ref} (E22 APEX N4 GRANDMASTER)", flush=True)
    print(f"==================================================================", flush=True)

    start_time = time.time()
    while True:
        try:
            subs = api.competition_submissions("rsna-knee-abnormality-detection")
            match = [s for s in subs if getattr(s, "ref", None) == sub_ref]
            if match:
                s = match[0]
                status_str = str(getattr(s, "status", "")).upper()
                score = getattr(s, "publicScore", None)
                elapsed_min = (time.time() - start_time) / 60.0
                print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] Status: {status_str} | Public Score: {score}", flush=True)

                if "COMPLETE" in status_str:
                    print(f"\n🎉 SUBMISSION EVALUATION COMPLETE! FINAL PUBLIC SCORE: {score}", flush=True)
                    break
                elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
                    print(f"\n[!] Submission failed with status: {status_str}", flush=True)
                    break
        except Exception as e:
            print(f"Warning: {e}", flush=True)

        time.sleep(30)

if __name__ == "__main__":
    monitor_sub()
