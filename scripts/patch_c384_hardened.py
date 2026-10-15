"""Hardens both C384 notebooks (GPU Fold 1 & TPU Fold 0) and creates Colab A100 notebook."""

import json
import os
import shutil

GPU_NB = "notebooks/training-book-c384-gpu-f1/training-book-c384-gpu-f1.ipynb"
TPU_NB = "notebooks/training-book-c384-tpu-f0/training-book-c384-tpu-f0.ipynb"
COLAB_DIR = "notebooks/colab-c384-a100"
COLAB_NB = os.path.join(COLAB_DIR, "colab-c384-a100-fold0.ipynb")


def harden_source(src: str, mode: str) -> str:
    # 1. Fix silent try/except around shape assert and raw_v1 fallback target_size
    old_cache_block = """        if cached_p:
            try:
                npz = np.load(cached_p)
                for plane in ["Sagittal", "Coronal", "Axial"]:
                    arr = npz[plane.lower()]  # (K, H, W) uint8
                    slabs = cached_plane_to_slabs(arr)
                    assert slabs.shape == (self.num_slices, 3, CONFIG["target_size"][0], CONFIG["target_size"][1]), f"Slice shape mismatch: got {slabs.shape}, expected ({self.num_slices}, 3, {CONFIG['target_size']})"
                    plane_tensors[plane] = torch.tensor(slabs, dtype=torch.float32)
            except Exception:
                plane_tensors = {}"""

    new_cache_block = """        if cached_p:
            try:
                npz = np.load(cached_p)
                raw_arrays = {plane: npz[plane.lower()] for plane in ["Sagittal", "Coronal", "Axial"]}
            except Exception as e:
                raise RuntimeError(f"Failed to read cached .npz at {cached_p}: {e}") from e
            for plane, arr in raw_arrays.items():
                slabs = cached_plane_to_slabs(arr)
                assert slabs.shape == (self.num_slices, 3, CONFIG["target_size"][0], CONFIG["target_size"][1]), (
                    f"Slice shape mismatch in {cached_p} ({plane}): got {slabs.shape}, expected ({self.num_slices}, 3, {CONFIG['target_size']})"
                )
                plane_tensors[plane] = torch.tensor(slabs, dtype=torch.float32)"""

    if old_cache_block in src:
        src = src.replace(old_cache_block, new_cache_block)

    src = src.replace(
        "load_and_preprocess_series(series_files, target_size=(256, 256), num_slices=self.num_slices)",
        "load_and_preprocess_series(series_files, target_size=CONFIG[\"target_size\"], num_slices=self.num_slices)",
    )

    # 2. Add Layer-Wise LR (backbone lr, MoE/report heads lr * 4.0)
    old_opt = """    params = list(model.parameters()) + (list(report_head.parameters()) if use_report_aux else [])
    optimizer = torch.optim.AdamW(params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])"""

    new_opt = """    backbone_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")] + (
        list(report_head.parameters()) if use_report_aux else []
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG["lr"]},
            {"params": head_params, "lr": CONFIG["lr"] * 4.0},
        ],
        weight_decay=CONFIG["weight_decay"],
    )"""

    if old_opt in src:
        src = src.replace(old_opt, new_opt)

    # 3. Enable bfloat16 autocast on TPU and A100
    old_amp = """    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast_ctx = contextlib.nullcontext"""

    new_amp = """    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast
    elif is_tpu:
        scaler = None
        autocast_ctx = lambda: torch.autocast("xla", dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_ctx = contextlib.nullcontext"""

    if old_amp in src:
        src = src.replace(old_amp, new_amp)

    # 4. Mode-specific CONFIG adjustments
    if mode == "gpu_f1":
        # Plan 3.4 fallback: coatnet_rmlp_1_rw2_224 at 384px, 3 epochs, batch_size=1, accum_steps=4
        src = src.replace(
            '"backbone": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",',
            '"backbone": "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k",',
        )
        src = src.replace('"timm_img_size": None,', '"timm_img_size": 384,')
        src = src.replace('"epochs": 4,', '"epochs": 3,')
        src = src.replace('"batch_size": 2,', '"batch_size": 1,')
        src = src.replace('"accum_steps": 2,', '"accum_steps": 4,')
        src = src.replace('"smoke_steps": None,', '"smoke_steps": 20,')
    elif mode == "tpu_f0":
        src = src.replace('"batch_size": 2,', '"batch_size": 1,')
        src = src.replace('"accum_steps": 2,', '"accum_steps": 4,')
        src = src.replace('"fold": 0,', '"fold": 2,')
        src = src.replace('"smoke_steps": 20,', '"smoke_steps": None,')

    compile(src, f"<{mode}>", "exec")
    return src


def update_notebook(path: str, mode: str):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][1]["source"])
    new_src = harden_source(src, mode)
    nb["cells"][1]["source"] = [l for l in new_src.splitlines(keepends=True)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated and verified {path} ({mode})")


def create_colab_notebook():
    os.makedirs(COLAB_DIR, exist_ok=True)
    with open(TPU_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === COLAB PRO (A100 / L4) 1-CLICK SETUP ===\n",
            "# 1. Upload your kaggle.json or set KAGGLE_USERNAME / KAGGLE_KEY in Colab Secrets\n",
            "# 2. Pulls the 15 GB 384px cache in ~2 minutes onto local NVMe (/kaggle/input/...)\n",
            "import os, subprocess\n",
            "try:\n",
            "    from google.colab import userdata, drive\n",
            "    drive.mount('/content/drive', force_remount=False)\n",
            "    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n",
            "    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')\n",
            "except Exception as e:\n",
            "    print('Note: Set KAGGLE_USERNAME and KAGGLE_KEY or ~/.kaggle/kaggle.json:', e)\n",
            "\n",
            "os.makedirs('/kaggle/input/rsna-knee-abnormality-detection', exist_ok=True)\n",
            "os.makedirs('/kaggle/input/creaky-caching-dataset-n4-384', exist_ok=True)\n",
            "os.makedirs('/kaggle/working', exist_ok=True)\n",
            "if not os.path.exists('/kaggle/input/creaky-caching-dataset-n4-384/cached_slices'):\n",
            "    subprocess.run(['pip', 'install', '-q', 'kaggle', 'timm'], check=True)\n",
            "    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'rsna-knee-abnormality-detection', '-f', 'train.csv', '-p', '/kaggle/input/rsna-knee-abnormality-detection'], check=False)\n",
            "    subprocess.run(['unzip', '-o', '/kaggle/input/rsna-knee-abnormality-detection/train.csv.zip', '-d', '/kaggle/input/rsna-knee-abnormality-detection'], check=False)\n",
            "    subprocess.run(['kaggle', 'kernels', 'output', 'nswitzer/creaky-caching-dataset-n4-384', '-p', '/kaggle/input/creaky-caching-dataset-n4-384'], check=True)\n",
        ],
    }

    src = "".join(nb["cells"][1]["source"])
    # Configure Colab A100 for Fold 0, batch_size=2, accum_steps=2, smoke_steps=None
    src = src.replace('"batch_size": 1,', '"batch_size": 2,')
    src = src.replace('"accum_steps": 4,', '"accum_steps": 2,')
    src = src.replace('"smoke_steps": 20,', '"smoke_steps": None,')
    compile(src, "<colab_a100>", "exec")
    nb["cells"][1]["source"] = [l for l in src.splitlines(keepends=True)]
    nb["cells"].insert(1, setup_cell)

    with open(COLAB_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created Colab Pro A100 notebook at {COLAB_NB}")


if __name__ == "__main__":
    update_notebook(TPU_NB, "tpu_f0")
