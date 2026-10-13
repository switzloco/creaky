import io
import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def run_dual_monitor():
    api = KaggleApi()
    api.authenticate()

    jobs = {
        "Fold 1 (GPU)": {
            "slug": "nswitzer/training-book-n4-cleanlab-fold-1",
            "fold": 1,
            "done": False,
        },
        "Fold 1 (TPU)": {
            "slug": "nswitzer/training-book-n4-cleanlab-tpu-fold-1",
            "fold": 1,
            "done": False,
        }
    }

    out_dir = Path("checkpoints/n4_cleanlab_trained")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================", flush=True)
    print("  DUAL TRAINING MONITOR: FOLD 1 (TPU) + FOLD 2 (GPU)", flush=True)
    print("==================================================================", flush=True)

    start_time = time.time()
    
    while not all(j["done"] for j in jobs.values()):
        elapsed_min = (time.time() - start_time) / 60.0
        status_line = []

        for name, job in jobs.items():
            if job["done"]:
                status_line.append(f"{name}: COMPLETE [HARVESTED]")
                continue

            try:
                status_obj = api.kernels_status(job["slug"])
                status_str = str(status_obj.status).upper()
                failure = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
                status_line.append(f"{name}: {status_str}")

                if "COMPLETE" in status_str:
                    print(f"\n--> {name} [{job['slug']}] COMPLETED! Harvesting outputs...", flush=True)
                    try:
                        api.kernels_output(job["slug"], path=str(out_dir))
                        print(f"Downloaded files into {out_dir}", flush=True)
                        preds_file = out_dir / f"val_preds_fold_{job['fold']}.csv"
                        if preds_file.exists():
                            from scripts.evaluate_n4_fold0 import evaluate_fold0
                            print(f"\nEvaluating performance for {name}:", flush=True)
                            evaluate_fold0(str(preds_file))
                    except Exception as harvest_err:
                        print(f"Harvest error for {name}: {harvest_err}", flush=True)
                    
                    job["done"] = True

                elif any(x in status_str for x in ["ERROR", "CANCELLED", "FAILED"]):
                    print(f"\n[!] {name} failed with status {status_str}: {failure}", flush=True)
                    job["done"] = True

            except Exception as e:
                status_line.append(f"{name}: ERR ({e})")

        print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] {' | '.join(status_line)}", flush=True)
        time.sleep(60)

    print("\n==================================================================", flush=True)
    print("  DUAL TRAINING COMPLETE: ALL ACTIVE FOLDS HARVESTED", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_dual_monitor()
