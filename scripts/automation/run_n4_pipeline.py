import io
import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

training_kernel = "nswitzer/training-book-n4-cleanlab-fold-0"
print(f"=== TRACK N4 AUTOMATED PIPELINE (PHASE 3: MONITOR & EVALUATE) ===", flush=True)
print(f"Monitoring Training Kernel [{training_kernel}]...", flush=True)

consecutive_errors = 0
while True:
    try:
        status_obj = api.kernels_status(training_kernel)
        status_str = str(status_obj.status).upper()
        failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
        print(f"[{time.strftime('%H:%M:%S')}] Training status: {status_str} | Failure: {failure}", flush=True)
        consecutive_errors = 0
        
        if "COMPLETE" in status_str:
            print("\n--> Track N4 Training COMPLETE! Downloading outputs...", flush=True)
            break
        elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
            print(f"Training kernel failed with status: {status_str}, failure: {failure}", flush=True)
            sys.exit(1)
            
    except Exception as e:
        consecutive_errors += 1
        print(f"[{time.strftime('%H:%M:%S')}] Transient error querying status ({consecutive_errors}/10): {e}", flush=True)
        if consecutive_errors >= 10:
            print("Too many consecutive network errors. Exiting.")
            sys.exit(1)
            
    time.sleep(60)

# 4. Download Outputs
out_dir = "checkpoints/n4_cleanlab_trained"
os.makedirs(out_dir, exist_ok=True)
print(f"\nPhase 4: Downloading training outputs to {out_dir}...", flush=True)
try:
    api.kernels_output(training_kernel, path=out_dir)
    print(f"Downloaded files: {os.listdir(out_dir)}", flush=True)
except Exception as e:
    print(f"Error downloading outputs: {e}")

# 5. Evaluate Performance
print("\nPhase 5: Evaluating Lateral Meniscus and Macro AUC on Gold Studies...", flush=True)
try:
    from scripts.evaluate_n4_fold0 import evaluate_fold0
    evaluate_fold0(os.path.join(out_dir, "val_preds_fold_0.csv"))
except Exception as e:
    print(f"Error during in-process evaluation: {e}")
    os.system(f'uv run python scripts/evaluate_n4_fold0.py "{out_dir}/val_preds_fold_0.csv"')
print("\n=== Track N4 Pipeline Execution Finished ===", flush=True)
