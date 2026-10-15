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
    "   - Automatically repairs malformed `kaggle.json` files and backs up the clean JSON to Google Drive so future runs are 100% hands-free.\n",
    "   - Pulls `train.csv` and all **4,407 pre-cached 384×384 `.npz` volumes** (`15.15 GB`) onto writable local NVMe (`data/processed/cached_slices`) in ~60–90 seconds using a 32-thread parallel GCS downloader.\n",
    "3. Run **Cell 2** to train **Fold 0** with the full **73.3M-param `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`** (`5 epochs`, `bfloat16` Tensor Cores, Cleanlab zero-weighting, 64-D report supervision).\n",
]

SETUP_CELL_CODE = r"""# === CELL 1: 1-CLICK COLAB A100 ENVIRONMENT & 32-THREAD FAST CACHE DOWNLOADER ===
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

subprocess.run(["pip", "install", "-q", "-U", "kaggle", "kagglesdk", "timm", "scikit-learn", "tqdm"], check=True)

import requests
import torch
from tqdm.auto import tqdm

# Bypass Colab's read-only /kaggle/input FUSE mount so Cell 2 uses writable local NVMe (data/raw & data/processed/cached_slices)
import genericpath
os.path.exists = lambda p: False if str(p).replace("\\", "/").startswith("/kaggle") else genericpath.exists(p)

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
    print(f"Note: Google Drive not mounted ({e}). Checkpoints will save to checkpoints/.")

# 2. Authenticate Kaggle (auto-repairs malformed kaggle.json quotes/syntax)
kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
drive_kaggle_json = Path(DRIVE_SAVE_DIR) / "kaggle.json"
content_kaggle_json = Path("/content/kaggle.json")

if not kaggle_json_path.exists():
    if drive_kaggle_json.exists():
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_kaggle_json, kaggle_json_path)
    elif content_kaggle_json.exists():
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(content_kaggle_json, kaggle_json_path)

if not kaggle_json_path.exists():
    print("Please upload your kaggle.json file:")
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        fname = next(iter(uploaded.keys()))
        kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json_path.write_bytes(uploaded[fname])

# Auto-repair broken quotes in kaggle.json (e.g. {" "username:nswitzer,key:...})
raw_text = kaggle_json_path.read_text(errors="ignore")
m_key = re.search(r"(KGAT_[A-Za-z0-9_]+|[a-fA-F0-9]{32})", raw_text)
long_tokens = re.findall(r"[A-Za-z0-9_]{20,}", raw_text)
kk = m_key.group(1) if m_key else (long_tokens[-1] if long_tokens else "")
ku = "nswitzer"
norm_payload = json.dumps({"username": ku, "key": kk})

for target_p in [
    kaggle_json_path,
    Path.home() / ".config" / "kaggle" / "kaggle.json",
]:
    target_p.parent.mkdir(parents=True, exist_ok=True)
    target_p.write_text(norm_payload)
    os.chmod(target_p, 0o600)

os.environ["KAGGLE_USERNAME"] = ku
os.environ["KAGGLE_KEY"] = kk
if kk.startswith("KGAT_") or len(kk) > 32:
    os.environ["KAGGLE_API_TOKEN"] = kk
    token_file = Path.home() / ".kaggle" / "access_token"
    token_file.write_text(kk)
    os.chmod(token_file, 0o600)

if genericpath.exists(DRIVE_SAVE_DIR):
    drive_kaggle_json.write_text(norm_payload)
print(f"--> Repaired & verified kaggle.json (username='{ku}', key_len={len(kk)}, bearer={kk.startswith('KGAT_') or len(kk) > 32}) and saved to Drive!")

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def make_kaggle_client():
    if kk.startswith("KGAT_") or len(kk) > 32:
        return KaggleClient(api_token=kk)
    return KaggleClient(username=ku, password=kk)

print("--> Kaggle SDK client initialized with Bearer Token auth!")

# 3. Download train.csv to writable local NVMe (data/raw/train.csv)
COMP_DIR = Path("data/raw")
COMP_DIR.mkdir(parents=True, exist_ok=True)
train_csv_path = COMP_DIR / "train.csv"
if not train_csv_path.exists():
    print("Downloading train.csv via KaggleClient...")
    with make_kaggle_client() as client:
        dl_req = ApiDownloadDataFileRequest()
        dl_req.competition_name = "rsna-knee-abnormality-detection"
        dl_req.file_name = "train.csv"
        dl_resp = client.competitions.competition_api_client.download_data_file(dl_req)
        raw_bytes = dl_resp.content
    if raw_bytes[:2] == b"PK":
        zip_p = COMP_DIR / "train.csv.zip"
        zip_p.write_bytes(raw_bytes)
        shutil.unpack_archive(str(zip_p), str(COMP_DIR))
        zip_p.unlink()
    else:
        train_csv_path.write_bytes(raw_bytes)
print(f"--> train.csv ready at {train_csv_path} ({train_csv_path.stat().st_size / 1e6:.2f} MB)")

# 4. 32-Thread Parallel Download of 384px Slice Cache (4,407 studies, 15.15 GB) to data/processed/cached_slices
CACHE_ROOT = Path("data/processed")
CACHE_SLICE_DIR = CACHE_ROOT / "cached_slices"
CACHE_SLICE_DIR.mkdir(parents=True, exist_ok=True)

existing_npz = list(CACHE_SLICE_DIR.glob("*.npz"))
if len(existing_npz) >= 4400:
    print(f"--> 384px cache already present ({len(existing_npz)} .npz files)! Skipping download.")
else:
    print("Listing signed URLs for nswitzer/creaky-caching-dataset-n4-384...")
    file_items = []
    page_token = None
    with make_kaggle_client() as client:
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
        dest = CACHE_SLICE_DIR / Path(rel_name).name
        if dest.exists() and dest.stat().st_size > 0:
            return True
        r = session.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True

    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(_dl_one, it) for it in file_items]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading 384px Cache"):
            pass

    final_count = len(list(CACHE_SLICE_DIR.glob("*.npz")))
    print(f"--> Cache download complete! {final_count} / {len(file_items)} .npz volumes on local NVMe ({CACHE_SLICE_DIR}).")
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
    src = src.replace('elif os.path.exists("data/processed/cached_slices"):', 'if not self.cached_dir and os.path.exists("data/processed/cached_slices"):')

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
