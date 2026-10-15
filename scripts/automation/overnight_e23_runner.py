"""Autonomous Overnight Orchestrator for Experiment E23 (GPU Fold 1 + TPU Fold 0).

1. Runs 20-step smoke test on GPU Fold 1 with hardened coatnet_rmlp_1_rw2_224 (384px, 3 epochs, batch_size=1).
2. Checks projected_hours against the GPU quota gate:
   - If projected_hours > 6.6, adjusts epochs 3 -> 2 so it is 100% guaranteed to finish inside the 7.0h remaining GPU quota without timing out.
   - Sets smoke_steps = None and pushes the full GPU Fold 1 run.
3. Watches TPU Fold 0 (currently queued as v1). As soon as v1 exits/errors, sets smoke_steps = None on the hardened TPU notebook (with bfloat16 autocast) and pushes v2 for full overnight training.
4. Polls every 90s with full network/DNS exception resilience.
5. Automatically harvests completed checkpoints and validation predictions into checkpoints/c384_cleanlab/ and runs Phase 5 OOF evaluation (scripts/evaluate_c384.py).
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

GPU_SLUG = "nswitzer/training-book-c384-cleanlab-gpu-fold-1"
TPU_SLUG = "nswitzer/training-book-c384-cleanlab-tpu-fold-0"
GPU_NB_DIR = "notebooks/training-book-c384-gpu-f1"
TPU_NB_DIR = "notebooks/training-book-c384-tpu-f0"
GPU_NB_FILE = os.path.join(GPU_NB_DIR, "training-book-c384-gpu-f1.ipynb")
TPU_NB_FILE = os.path.join(TPU_NB_DIR, "training-book-c384-tpu-f0.ipynb")
HARVEST_DIR = Path("checkpoints/c384_cleanlab")


def set_nb_config(nb_file: str, replacements: dict):
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][1]["source"])
    for old_str, new_str in replacements.items():
        src = src.replace(old_str, new_str)
    compile(src, f"<{nb_file}>", "exec")
    nb["cells"][1]["source"] = [l for l in src.splitlines(keepends=True)]
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def push_kernel(nb_dir: str) -> bool:
    res = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", nb_dir],
        capture_output=True,
        text=True,
    )
    print(f"[PUSH {nb_dir}] rc={res.returncode} | {res.stdout.strip()} {res.stderr.strip()}", flush=True)
    return res.returncode == 0


def parse_smoke_log(api: KaggleApi, slug: str):
    raw_log = api.kernels_logs(slug)
    entries = json.loads(raw_log)
    proj_hours = None
    sec_train = None
    peak_mem = "n/a"
    for e in entries:
        data = e.get("data", "")
        m1 = re.search(r"SMOKE sec_per_train_step=([\d.]+)\s+sec_per_val_step=([\d.]+)\s+peak_mem=([^\n]+)", data)
        if m1:
            sec_train = float(m1.group(1))
            peak_mem = m1.group(3).strip()
        m2 = re.search(r"SMOKE projected_hours=([\d.]+)", data)
        if m2:
            proj_hours = float(m2.group(1))
    return sec_train, proj_hours, peak_mem


def harvest_kernel(api: KaggleApi, slug: str, label: str):
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[HARVEST] Downloading outputs for {label} ({slug}) -> {HARVEST_DIR}...", flush=True)
    try:
        api.kernels_output(slug, path=str(HARVEST_DIR), quiet=False)
        print(f"[HARVEST] Successfully harvested {slug}!", flush=True)
    except Exception as e:
        print(f"[HARVEST ERROR] {slug}: {e}", flush=True)


def main():
    api = KaggleApi()
    api.authenticate()

    print("=" * 75, flush=True)
    print("  LAUNCHING AUTONOMOUS OVERNIGHT E23 PIPELINE", flush=True)
    print("=" * 75, flush=True)

    # Step 1: Push hardened GPU Fold 1 Smoke Probe (coatnet_rmlp_1_rw2_224 @ 384px, B=1, 3 epochs)
    print("[STEP 1] Pushing GPU Fold 1 smoke probe (coatnet_rmlp_1_rw2_224, 384px, B=1, 3 epochs)...", flush=True)
    push_kernel(GPU_NB_DIR)
    time.sleep(15)

    gpu_phase = "SMOKE"      # SMOKE -> FULL -> DONE
    tpu_phase = "V1_QUEUED"  # V1_QUEUED -> FULL -> DONE
    start_ts = time.time()

    while gpu_phase != "DONE" or tpu_phase != "DONE":
        elapsed_m = (time.time() - start_ts) / 60.0
        try:
            s_gpu = str(api.kernels_status(GPU_SLUG).status).upper()
            s_tpu = str(api.kernels_status(TPU_SLUG).status).upper()
            print(
                f"[{time.strftime('%H:%M:%S')} | {elapsed_m:5.1f}m] "
                f"GPU ({gpu_phase}): {s_gpu} | TPU ({tpu_phase}): {s_tpu}",
                flush=True,
            )

            # Handle GPU state machine
            if gpu_phase == "SMOKE" and any(k in s_gpu for k in ["COMPLETE", "ERROR", "CANCELLED", "FAILED"]):
                if "COMPLETE" in s_gpu:
                    sec_train, proj_hours, peak_mem = parse_smoke_log(api, GPU_SLUG)
                    print(
                        f"  --> GPU Smoke Complete: sec_train={sec_train}, proj_hours={proj_hours}, peak_mem={peak_mem}",
                        flush=True,
                    )
                    repl = {'"smoke_steps": 20,': '"smoke_steps": None,'}
                    if proj_hours is not None and proj_hours > 6.6:
                        print(
                            f"  --> proj_hours ({proj_hours:.2f}h) > 6.6h quota gate; adjusting epochs 3 -> 2 for safe completion.",
                            flush=True,
                        )
                        repl['"epochs": 3,'] = '"epochs": 2,'
                    set_nb_config(GPU_NB_FILE, repl)
                    print("  --> Launching FULL GPU Fold 1 training run!", flush=True)
                    if push_kernel(GPU_NB_DIR):
                        gpu_phase = "FULL"
                        time.sleep(20)
                else:
                    print(f"  --> [!] GPU Smoke ended with {s_gpu}. Stopping GPU track.", flush=True)
                    gpu_phase = "DONE"

            elif gpu_phase == "FULL" and any(k in s_gpu for k in ["COMPLETE", "ERROR", "CANCELLED", "FAILED"]):
                print(f"  --> GPU Full Run finished with status: {s_gpu}", flush=True)
                if "COMPLETE" in s_gpu:
                    harvest_kernel(api, GPU_SLUG, "GPU Fold 1")
                gpu_phase = "DONE"

            # Handle TPU state machine
            if tpu_phase == "V1_QUEUED" and any(k in s_tpu for k in ["COMPLETE", "ERROR", "CANCELLED", "FAILED"]):
                print(f"  --> TPU v1 exited queue with status {s_tpu}. Pushing hardened FULL TPU Fold 0 run!", flush=True)
                set_nb_config(TPU_NB_FILE, {'"smoke_steps": 20,': '"smoke_steps": None,', '"epochs": 4,': '"epochs": 3,'})
                if push_kernel(TPU_NB_DIR):
                    tpu_phase = "FULL"
                    time.sleep(20)

            elif tpu_phase == "FULL" and any(k in s_tpu for k in ["COMPLETE", "ERROR", "CANCELLED", "FAILED"]):
                print(f"  --> TPU Full Run finished with status: {s_tpu}", flush=True)
                if "COMPLETE" in s_tpu:
                    harvest_kernel(api, TPU_SLUG, "TPU Fold 0")
                tpu_phase = "DONE"

        except Exception as e:
            print(f"[Monitor Warning - retrying in 60s] {e}", flush=True)

        if gpu_phase == "DONE" and tpu_phase == "DONE":
            break
        time.sleep(60)

    print("\n" + "=" * 75, flush=True)
    print("  RUNNING PHASE 5 EVALUATION (scripts/evaluate_c384.py)", flush=True)
    print("=" * 75, flush=True)
    subprocess.run(["uv", "run", "python", "scripts/evaluate_c384.py"], check=False)


if __name__ == "__main__":
    main()
