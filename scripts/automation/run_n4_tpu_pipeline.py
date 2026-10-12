import argparse
import io
import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def run_tpu_pipeline(fold: int, push: bool = True):
    api = KaggleApi()
    api.authenticate()

    kernel_dir = Path(f"notebooks/training-book-n4-tpu-f{fold}")
    kernel_slug = f"nswitzer/training-book-n4-cleanlab-tpu-fold-{fold}"
    
    print(f"\n=======================================================", flush=True)
    print(f"  TRACK N4 TPU PIPELINE: FOLD {fold}", flush=True)
    print(f"  Kernel Slug: {kernel_slug}", flush=True)
    print(f"=======================================================", flush=True)

    if push:
        print(f"\n[Phase 1] Pushing {kernel_dir} to Kaggle TPU...", flush=True)
        try:
            api.kernels_push(str(kernel_dir))
            print(f"Successfully pushed {kernel_slug} to Kaggle!", flush=True)
        except Exception as e:
            print(f"Push failed: {e}", flush=True)
            sys.exit(1)

    print(f"\n[Phase 2] Monitoring TPU Training on Kaggle...", flush=True)
    consecutive_errors = 0
    start_time = time.time()
    
    while True:
        try:
            status_obj = api.kernels_status(kernel_slug)
            status_str = str(status_obj.status).upper()
            failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] TPU Status: {status_str} | Info: {failure}", flush=True)
            consecutive_errors = 0
            
            if "COMPLETE" in status_str:
                print(f"\n--> Fold {fold} TPU Training COMPLETE! Downloading outputs...", flush=True)
                break
            elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
                print(f"Training failed with status: {status_str}, error: {failure}", flush=True)
                return False
                
        except Exception as e:
            consecutive_errors += 1
            print(f"[{time.strftime('%H:%M:%S')}] Status check warning ({consecutive_errors}/10): {e}", flush=True)
            if consecutive_errors >= 10:
                print("Too many consecutive errors. Aborting.")
                return False

        time.sleep(60)

    # Phase 3: Download outputs
    out_dir = Path("checkpoints/n4_cleanlab_trained")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Phase 3] Harvesting Fold {fold} weights and validation predictions...", flush=True)
    try:
        api.kernels_output(kernel_slug, path=str(out_dir))
        print(f"Downloaded files: {os.listdir(out_dir)}", flush=True)
    except Exception as e:
        print(f"Error downloading outputs: {e}", flush=True)
        return False

    # Phase 4: Evaluate
    preds_csv = out_dir / f"val_preds_fold_{fold}.csv"
    if preds_csv.exists():
        print(f"\n[Phase 4] Evaluating Fold {fold} Performance...", flush=True)
        try:
            from scripts.evaluate_n4_fold0 import evaluate_fold0
            evaluate_fold0(str(preds_csv))
        except Exception as e:
            print(f"Error during evaluation: {e}", flush=True)
    else:
        print(f"Predictions file not found at {preds_csv}", flush=True)

    print(f"\n=== Fold {fold} TPU Pipeline Finished Successfully ===", flush=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2], help="Fold index to launch (1 or 2)")
    parser.add_argument("--no-push", action="store_true", help="Monitor existing kernel without pushing")
    args = parser.parse_args()

    run_tpu_pipeline(fold=args.fold, push=not args.no_push)
