"""Builds the turnkey Google Colab Pro (A100) notebook for C384 Fold 0 (coatnet_rmlp_2_rw_384, 5 epochs)."""

import json
import os

TPU_NB = "notebooks/training-book-c384-tpu-f0/training-book-c384-tpu-f0.ipynb"
COLAB_DIR = "notebooks/colab-c384-a100"
COLAB_NB = os.path.join(COLAB_DIR, "colab-c384-a100-fold0.ipynb")

MARKDOWN_HEADER = [
    "# 🚀 Creaky E23 — Native 384×384 CoAtNet-2 MoE on Google Colab Pro (A100)\n",
    "\n",
    "**Instructions (Takes ~60 seconds to start):**\n",
    "1. In Colab menu: **Runtime → Change runtime type → Hardware accelerator: `A100 GPU`** (or `L4 GPU`).\n",
    "2. Run **Cell 1** below:\n",
    "   - Mounts your Google Drive (`/content/drive/MyDrive/creaky_c384`) so checkpoints are auto-saved after every epoch.\n",
    "   - Automatically handles both legacy `kaggle.json` (`username` + `key`) and new Kaggle v2 API tokens (`token` / `KAGGLE_API_TOKEN`), and backs up `kaggle.json` to Google Drive so future runs are 100% hands-free.\n",
    "   - Pulls `train.csv` and all **4,407 pre-cached 384×384 `.npz` volumes** (`15.15 GB`) in ~60–90 seconds using a 32-thread parallel GCS downloader.\n",
    "3. Run **Cell 2** to train **Fold 0** with the full **73.3M-param `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`** (`5 epochs`, `bfloat16` Tensor Cores, Cleanlab zero-weighting, 64-D report supervision).\n",
]

SETUP_CELL_CODE = """# === CELL 1: 1-CLICK COLAB A100 ENVIRONMENT & 32-THREAD FAST CACHE DOWNLOADER ===
import os
import json
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

subprocess.run(["pip", "install", "-q", "-U", "kaggle", "kagglesdk", "timm", "scikit-learn", "tqdm"], check=True)

import requests
import torch
from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB | BF16: {torch.cuda.is_bf16_supported()}")

# 1. Mount Google Drive for persistent checkpoint auto-backup
DRIVE_SAVE_DIR = "/content/drive/MyDrive/creaky_c384"
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    print(f"--> Google Drive mounted! Checkpoints will auto-save to: {DRIVE_SAVE_DIR}")
except Exception as e:
    print(f"Note: Google Drive not mounted ({e}). Checkpoints will save to /kaggle/working.")

# 2. Authenticate Kaggle (handles both legacy username/key and v2 token formats)
kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
drive_kaggle_json = Path(DRIVE_SAVE_DIR) / "kaggle.json"
content_kaggle_json = Path("/content/kaggle.json")

if not kaggle_json_path.exists():
    if drive_kaggle_json.exists():
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_kaggle_json, kaggle_json_path)
        print(f"--> Loaded kaggle.json from Google Drive ({drive_kaggle_json})!")
    elif content_kaggle_json.exists():
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(content_kaggle_json, kaggle_json_path)
        print("--> Found existing /content/kaggle.json!")

if not kaggle_json_path.exists():
    try:
        from google.colab import userdata
        ku = userdata.get("KAGGLE_USERNAME") or "nswitzer"
        kk = userdata.get("KAGGLE_KEY") or userdata.get("KAGGLE_API_TOKEN")
        if kk:
            kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
            kaggle_json_path.write_text(json.dumps({"username": ku, "key": kk}))
            print("--> Loaded Kaggle credentials from Colab Secrets.")
    except Exception:
        pass

if not kaggle_json_path.exists():
    print("Please upload your kaggle.json file (located at C:\\\\Users\\\\nswitzer\\\\.kaggle\\\\kaggle.json):")
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        fname = next(iter(uploaded.keys()))
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json_path.write_bytes(uploaded[fname])

# Normalize kaggle.json so both legacy (username + key) and v2 (token / access_token) work
if kaggle_json_path.exists():
    raw_text = kaggle_json_path.read_text().strip()
    try:
        raw_cfg = json.loads(raw_text)
    except Exception:
        raw_cfg = {"key": raw_text}
    ku = raw_cfg.get("username") or raw_cfg.get("user") or "nswitzer"
    kk = (
        raw_cfg.get("key")
        or raw_cfg.get("token")
        or raw_cfg.get("access_token")
        or next((str(v) for k, v in raw_cfg.items() if k not in ("username", "user")), "")
    )
    norm_payload = json.dumps({"username": ku, "key": kk})
    for target_p in [
        kaggle_json_path,
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]:
        target_p.parent.mkdir(parents=True, exist_ok=True)
        target_p.write_text(norm_payload)
        os.chmod(target_p, 0o600)
    if kk.startswith("KGAT_"):
        acc_p = kaggle_json_path.parent / "access_token"
        acc_p.write_text(kk)
        os.chmod(acc_p, 0o600)
        os.environ["KAGGLE_API_TOKEN"] = kk
    os.environ["KAGGLE_USERNAME"] = ku
    os.environ["KAGGLE_KEY"] = kk
    if os.path.exists(DRIVE_SAVE_DIR):
        drive_kaggle_json.write_text(norm_payload)
        print(f"--> Normalized credentials (username='{ku}') and backed up to {drive_kaggle_json}!")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
print("--> Kaggle API authenticated successfully!")

# 3. Download train.csv
COMP_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
COMP_DIR.mkdir(parents=True, exist_ok=True)
if not (COMP_DIR / "train.csv").exists():
    print("Downloading train.csv...")
    api.competition_download_file("rsna-knee-abnormality-detection", "train.csv", path=str(COMP_DIR), quiet=False)
    zip_p = COMP_DIR / "train.csv.zip"
    if zip_p.exists():
        shutil.unpack_archive(str(zip_p), str(COMP_DIR))
        zip_p.unlink()
print(f"--> train.csv ready ({ (COMP_DIR / 'train.csv').stat().st_size / 1e6:.2f} MB)")

# 4. 32-Thread Parallel Download of 384px Slice Cache (4,407 studies, 15.15 GB)
CACHE_ROOT = Path("/kaggle/input/creaky-caching-dataset-n4-384")
CACHE_SLICE_DIR = CACHE_ROOT / "cached_slices"
CACHE_SLICE_DIR.mkdir(parents=True, exist_ok=True)

existing_npz = list(CACHE_SLICE_DIR.glob("*.npz"))
if len(existing_npz) >= 4400:
    print(f"--> 384px cache already present ({len(existing_npz)} .npz files)! Skipping download.")
else:
    print("Listing signed URLs for nswitzer/creaky-caching-dataset-n4-384...")
    file_items = []
    page_token = None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "nswitzer"
            req.kernel_slug = "creaky-caching-dataset-n4-384"
            req.page_size = 100
            if page_token:
                req.page_token = page_token
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in (resp.files or []):
                if f.file_name.endswith(".npz"):
                    file_items.append((f.file_name, f.url))
            page_token = resp.next_page_token
            if not page_token:
                break
    print(f"Found {len(file_items)} cached .npz volumes. Downloading with 32 threads...")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=3)
    session.mount("https://", adapter)

    def _dl_one(item):
        rel_name, url = item
        dest = CACHE_ROOT / rel_name
        if dest.exists() and dest.stat().st_size > 0:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = session.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True

    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(_dl_one, it) for it in file_items]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading 384px Cache"):
            pass

    final_count = len(list(CACHE_SLICE_DIR.glob("*.npz")))
    print(f"--> Cache download complete! {final_count} / {len(file_items)} .npz volumes on local NVMe.")
"""


def main():
    os.makedirs(COLAB_DIR, exist_ok=True)
    with open(TPU_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    train_cell_idx = len(nb["cells"]) - 1
    src = "".join(nb["cells"][train_cell_idx]["source"])

    # Configure for A100: coatnet_rmlp_2_rw_384, Fold 0, 5 epochs, batch_size=2, accum_steps=2, smoke_steps=None
    src = src.replace('"epochs": 4,', '"epochs": 5,')
    src = src.replace('"epochs": 3,', '"epochs": 5,')
    src = src.replace('"fold": 2,', '"fold": 0,')
    src = src.replace('"batch_size": 1,', '"batch_size": 2,')
    src = src.replace('"accum_steps": 4,', '"accum_steps": 2,')
    src = src.replace('"smoke_steps": 20,', '"smoke_steps": None,')

    old_cuda_amp = """    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast"""

    new_cuda_amp = """    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            print("[AMP] Using native CUDA bfloat16 autocast (A100/L4 Tensor Cores)")
            scaler = None
            autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = torch.cuda.amp.autocast"""

    if old_cuda_amp in src:
        src = src.replace(old_cuda_amp, new_cuda_amp)

    old_save_log = '            print(f"  >>> Saved validation predictions to: {val_preds_path}")'
    new_save_log = """            print(f"  >>> Saved validation predictions to: {val_preds_path}")
            if os.path.exists("/content/drive/MyDrive/creaky_c384"):
                import shutil as _shutil
                _shutil.copy2(best_ckpt_path, f"/content/drive/MyDrive/creaky_c384/best_model_fold_{target_fold}.pt")
                _shutil.copy2(val_preds_path, f"/content/drive/MyDrive/creaky_c384/val_preds_fold_{target_fold}.csv")
                print(f"  >>> Auto-backed up checkpoint & OOF predictions to Google Drive (/content/drive/MyDrive/creaky_c384/)!")"""

    if old_save_log in src and "Auto-backed up checkpoint" not in src:
        src = src.replace(old_save_log, new_save_log)

    compile(SETUP_CELL_CODE, "<colab_cell_1>", "exec")
    compile(src, "<colab_cell_2>", "exec")

    colab_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": MARKDOWN_HEADER,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l for l in SETUP_CELL_CODE.splitlines(keepends=True)],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l for l in src.splitlines(keepends=True)],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "machine_shape": "hm",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(COLAB_NB, "w", encoding="utf-8") as f:
        json.dump(colab_nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully built turnkey Colab A100 notebook at {COLAB_NB}!")


if __name__ == "__main__":
    main()
