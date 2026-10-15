import io
import os
import sys
import time
import json
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def monitor_smokes():
    api = KaggleApi()
    api.authenticate()

    gpu_slug = "nswitzer/training-book-c384-cleanlab-gpu-fold-1"
    tpu_slug = "nswitzer/training-book-c384-cleanlab-tpu-fold-0"

    print("=" * 70, flush=True)
    print("  MONITORING PHASE 3 SMOKE TEST: GPU FOLD 1 (V2) & TPU FOLD 0", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    gpu_done = False

    while not gpu_done:
        elapsed_min = (time.time() - start_time) / 60.0
        try:
            s_gpu = str(api.kernels_status(gpu_slug).status).upper()
            s_tpu = str(api.kernels_status(tpu_slug).status).upper()
            print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] GPU Fold 1: {s_gpu} | TPU Fold 0: {s_tpu}", flush=True)

            if any(x in s_gpu for x in ["COMPLETE", "ERROR", "CANCELLED", "FAILED"]):
                gpu_done = True
                print(f"\n🎉 GPU Smoke Test concluded with status: {s_gpu}", flush=True)
                break
        except Exception as e:
            print(f"Warning: {e}", flush=True)

        time.sleep(20)

    # Extract GPU smoke metrics
    print("\n" + "=" * 70, flush=True)
    print(f"  EXTRACTING METRICS FOR GPU ({gpu_slug})", flush=True)
    print("=" * 70, flush=True)

    try:
        raw_log = api.kernels_logs(gpu_slug)
        entries = json.loads(raw_log)
        smoke_lines = []
        fast_mode = False
        for e in entries:
            data = e.get("data", "")
            if "[Fast Mode]" in data:
                fast_mode = True
            if "SMOKE" in data or "Error" in data or "Traceback" in data or "sec_per" in data:
                smoke_lines.append(data.strip())
        print(f"Fast Mode Detected: {fast_mode}", flush=True)
        print("Smoke Log Highlights:", flush=True)
        for sl in smoke_lines[-20:]:
            print("  " + sl, flush=True)
    except Exception as e:
        print(f"Error reading GPU log: {e}", flush=True)

if __name__ == "__main__":
    monitor_smokes()
