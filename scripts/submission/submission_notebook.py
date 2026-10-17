"""Standalone Kaggle Submission Script / Inference Notebook Generator.

This script is self-contained and designed to run inside a Kaggle Code Competition
environment with Internet Disabled.

Pipeline:
1. Auto-detects Kaggle vs. Local paths.
2. Loads test.csv and test_series.csv.
3. Reads DICOM series directly from disk on-the-fly.
4. Preprocesses slices exactly like the training cache (see the preprocessing section).
5. Feeds 2.5D multi-planar slabs into the KneeAbnormalityClassifier.
6. Exports submission.csv with exactly matching StudyInstanceUIDs and target probabilities.
"""

import os
import sys
import glob
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ==============================================================================

# CONFIGURATION & CONSTANTS
# ==============================================================================

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Which attached checkpoints to use. Empty = every .pt/.pth found under /kaggle/input.
# Otherwise only checkpoints whose full path contains one of these substrings are loaded,
# e.g. ["creaky-e04-convnext-fold0"] for a solo run. Attached kernel outputs (like
# training-book) are picked up automatically, so set this for any ablation submission.
CHECKPOINT_FILTER: List[str] = []

# Default class priors if model weights are not found
DEFAULT_PRIORS = {
    "ACL": 0.414, "MCL": 0.155, "Medial Meniscus": 0.448, "Lateral Meniscus": 0.397,
    "Medial OA": 0.259, "Lateral OA": 0.190, "PF OA": 0.362, "Effusion": 0.603,
    "Synovitis": 0.466, "Baker's": 0.207, "Contusion": 0.328, "Fracture": 0.310
}


def get_data_paths() -> Tuple[str, str, str, List[str]]:
    """Auto-detect competition paths and model checkpoints across Kaggle and Local environments.

    Uses deterministic path checks (NOT os.walk) to avoid 4+ minute scans of 27K+ DICOMs.
    Handles code competition layout where test data is only present during hidden submission.
    """
    test_csv = None
    test_series_csv = None
    test_series_dir = None
    ckpt_files = []

    # 1. Check if running inside Kaggle environment
    if os.path.exists("/kaggle/input"):
        print("Detected Kaggle environment. Checking known competition paths...")

        # Known Kaggle competition data paths (deterministic, no os.walk needed)
        COMP_SLUG = "rsna-knee-abnormality-detection"
        candidate_roots = [
            f"/kaggle/input/competitions/{COMP_SLUG}",   # competition data mount
            f"/kaggle/input/{COMP_SLUG}",                 # dataset attachment
        ]

        # Also scan for any top-level directory that contains test.csv
        try:
            for entry in os.listdir("/kaggle/input"):
                entry_path = os.path.join("/kaggle/input", entry)
                if os.path.isdir(entry_path) and entry_path not in candidate_roots:
                    candidate_roots.append(entry_path)
            # Check /kaggle/input/competitions/ subfolders too
            comp_base = "/kaggle/input/competitions"
            if os.path.isdir(comp_base):
                for entry in os.listdir(comp_base):
                    entry_path = os.path.join(comp_base, entry)
                    if os.path.isdir(entry_path) and entry_path not in candidate_roots:
                        candidate_roots.append(entry_path)
        except OSError:
            pass

        print(f"  Candidate roots: {candidate_roots}")

        for root in candidate_roots:
            if not os.path.isdir(root):
                continue
            tc = os.path.join(root, "test.csv")
            if os.path.isfile(tc):
                test_csv = tc
                print(f"  --> Found test.csv at: {test_csv}")

                tsc = os.path.join(root, "test_series.csv")
                if os.path.isfile(tsc):
                    test_series_csv = tsc
                    print(f"  --> Found test_series.csv at: {test_series_csv}")

                for cand in ["test_series", "test_dicom", "test", "test_images"]:
                    cand_path = os.path.join(root, cand)
                    if os.path.isdir(cand_path):
                        test_series_dir = cand_path
                        print(f"  --> Found test series dir at: {test_series_dir}")
                        break

                if test_series_dir is None:
                    test_series_dir = os.path.join(root, "test_series")
                break  # found test.csv, stop searching

        # If test.csv not found, check if train.csv exists (interactive session on code comp)
        if test_csv is None:
            for root in candidate_roots:
                if not os.path.isdir(root):
                    continue
                tc = os.path.join(root, "train.csv")
                if os.path.isfile(tc):
                    print(f"  [!] test.csv NOT found (code competition interactive session).")
                    print(f"      Found train.csv at: {tc}")
                    print(f"      Will create a dummy test set from train.csv for validation.")
                    # Create a minimal dummy test.csv from first 3 rows of train.csv
                    train_df = pd.read_csv(tc)
                    dummy_test = train_df[["StudyInstanceUID"]].head(3)
                    dummy_path = "/kaggle/working/test_dummy.csv"
                    dummy_test.to_csv(dummy_path, index=False)
                    test_csv = dummy_path
                    print(f"  --> Created dummy test.csv at: {dummy_path} ({len(dummy_test)} studies)")

                    # Still look for test_series dir from training data
                    for cand in ["train_series", "test_series"]:
                        cand_path = os.path.join(root, cand)
                        if os.path.isdir(cand_path):
                            test_series_dir = cand_path
                            print(f"  --> Using {cand} directory: {cand_path}")
                            break

                    tsc = os.path.join(root, "test_series.csv")
                    if os.path.isfile(tsc):
                        test_series_csv = tsc
                    elif os.path.isfile(os.path.join(root, "train_series.csv")):
                        test_series_csv = os.path.join(root, "train_series.csv")
                        print(f"  --> Using train_series.csv for metadata: {test_series_csv}")
                    break

        # Debug: show what's actually mounted under /kaggle/input/
        print("  [DEBUG] Contents of /kaggle/input/:")
        try:
            for entry in sorted(os.listdir("/kaggle/input")):
                entry_path = os.path.join("/kaggle/input", entry)
                marker = "dir" if os.path.isdir(entry_path) else "file"
                print(f"    {marker}: {entry}")
                if os.path.isdir(entry_path):
                    for sub in sorted(os.listdir(entry_path))[:10]:
                        sub_path = os.path.join(entry_path, sub)
                        marker2 = "dir" if os.path.isdir(sub_path) else "file"
                        print(f"      {marker2}: {sub}")
        except OSError as e:
            print(f"    Error listing: {e}")

        # Scan for model checkpoints
        # 1. Kaggle Models mount deeply: /kaggle/input/models/<user>/<name>/<fw>/<var>/<ver>/*.pt
        #    Safe to os.walk because /kaggle/input/models/ only contains model artifacts, not DICOMs.
        kaggle_models_dir = "/kaggle/input/models"
        if os.path.isdir(kaggle_models_dir):
            for dirpath, _, filenames in os.walk(kaggle_models_dir):
                for f in filenames:
                    if f.endswith(".pt") or f.endswith(".pth"):
                        ckpt_path = os.path.join(dirpath, f)
                        ckpt_files.append(ckpt_path)
                        print(f"  --> Found model checkpoint at: {ckpt_path}")

        # 2. Dataset-style attachments: scan 2 levels under /kaggle/input/<dataset>/
        try:
            for entry in os.listdir("/kaggle/input"):
                if entry == "models" or entry == "competitions":
                    continue  # already handled above / not checkpoints
                entry_path = os.path.join("/kaggle/input", entry)
                if os.path.isdir(entry_path):
                    for f in os.listdir(entry_path):
                        if f.endswith(".pt") or f.endswith(".pth"):
                            ckpt_path = os.path.join(entry_path, f)
                            ckpt_files.append(ckpt_path)
                            print(f"  --> Found model checkpoint at: {ckpt_path}")
                    for sub in os.listdir(entry_path):
                        sub_path = os.path.join(entry_path, sub)
                        if os.path.isdir(sub_path):
                            for f in os.listdir(sub_path):
                                if f.endswith(".pt") or f.endswith(".pth"):
                                    ckpt_path = os.path.join(sub_path, f)
                                    ckpt_files.append(ckpt_path)
                                    print(f"  --> Found model checkpoint at: {ckpt_path}")
        except OSError:
            pass

    # 2. Local workspace fallback
    if test_csv is None or not os.path.exists(test_csv):
        local_root = os.path.abspath("data/raw")
        test_csv = os.path.join(local_root, "test.csv")
        test_series_csv = os.path.join(local_root, "test_series.csv")
        test_series_dir = os.path.join(local_root, "sample_dicom")
        local_checkpoints = os.path.abspath("checkpoints")
        if os.path.exists(local_checkpoints):
            for sub in ["n4_cleanlab_trained", "c384_cleanlab"]:
                sub_dir = os.path.join(local_checkpoints, sub)
                if os.path.isdir(sub_dir):
                    for f in sorted(os.listdir(sub_dir)):
                        if f.endswith(".pt") or f.endswith(".pth"):
                            ckpt_files.append(os.path.join(sub_dir, f))
            if not ckpt_files:
                for f in os.listdir(local_checkpoints):
                    if f.endswith(".pt") or f.endswith(".pth"):
                        ckpt_files.append(os.path.join(local_checkpoints, f))

    return test_csv, test_series_csv, test_series_dir, ckpt_files



# ==============================================================================
# DICOM PREPROCESSING — MUST MATCH HOW EACH MODEL WAS TRAINED
# ==============================================================================
# Each checkpoint is run with the preprocessing it was trained on (its "pipeline"):
#
# "cache_v1": trained on slices from scripts/preprocessing/cache_kaggle_kernel.py, turned into
#   2.5D slabs by cached_plane_to_slabs() in the training script.
#   - slices sorted by InstanceNumber, 16 sampled from the central 12%-88% band
#   - DICOM window center/width (no rescale slope/intercept), border crop margin 4
#   - neighbour channels = adjacent *sampled* slices
#   - series per plane: exact plane name, last matching row of the series CSV
#
# "raw_v1": trained on raw DICOMs by the training script's load_and_preprocess_series()
#   (models trained before the cache existed, e.g. the resnet34 run).
#   - slices sorted by physical position, 16 sampled across the full series
#   - rescale slope/intercept applied, border crop margin 5
#   - neighbour channels = truly adjacent slices
#   - series per plane: first row whose plane name contains the plane
#
# The functions below are copies of the training-side code. Change them only together with it.
# tests/test_preprocessing_parity.py checks they stay in sync.

PIPELINES = ("cache_v1", "cache_v1_384", "raw_v1")

def apply_dicom_windowing(pixels: np.ndarray, center=None, width=None, photometric: str = "MONOCHROME2") -> np.ndarray:
    img = pixels.astype(np.float32)
    if photometric == "MONOCHROME1":
        img = img.max() - img
    if isinstance(center, (list, pydicom.multival.MultiValue)):
        center = float(center[0])
    if isinstance(width, (list, pydicom.multival.MultiValue)):
        width = float(width[0])

    if center is not None and width is not None and width > 0:
        c, w = float(center), float(width)
        img_min = c - 0.5 - (w - 1) / 2.0
        img_max = c - 0.5 + (w - 1) / 2.0
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        vmin, vmax = np.percentile(img, 1.0), np.percentile(img, 99.0)
        if vmax > vmin:
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin) * 255.0
        else:
            img = np.zeros_like(img)
    return img.astype(np.uint8)


def crop_empty_borders(img: np.ndarray, threshold: int = 10, margin: int = 4) -> np.ndarray:
    mask = img > threshold
    if not np.any(mask):
        return img
    y_idx, x_idx = np.where(mask)
    h, w = img.shape[:2]
    ymin = max(0, y_idx.min() - margin)
    ymax = min(h, y_idx.max() + margin + 1)
    xmin = max(0, x_idx.min() - margin)
    xmax = min(w, x_idx.max() + margin + 1)
    return img[ymin:ymax, xmin:xmax]


def read_slice(dcm_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=False)
        center = getattr(dcm, "WindowCenter", None)
        width = getattr(dcm, "WindowWidth", None)
        photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
        img = apply_dicom_windowing(dcm.pixel_array, center, width, photometric)
        img = crop_empty_borders(img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA if img.shape[0] > target_size[0] else cv2.INTER_LINEAR)
        return img
    except Exception:
        return None


def sort_by_instance_number(files: List[str]) -> List[str]:
    """Same ordering as the cache's get_sorted_files(), for an explicit list of files."""
    meta = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            inst = int(getattr(dcm, "InstanceNumber", 0))
            meta.append((inst, f))
        except Exception:
            base = os.path.splitext(os.path.basename(f))[0]
            inst = int(base) if base.isdigit() else 0
            meta.append((inst, f))
    meta.sort(key=lambda x: x[0])
    return [x[1] for x in meta]


def sample_plane_like_cache(dicom_files: List[str], num_slices: int = 16, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Same as the cache's process_plane(): returns (num_slices, H, W) uint8.

    Only the sampled DICOMs are decoded, so inference reads 16 files per plane, not the whole series.
    """
    dcm_files = sort_by_instance_number(dicom_files)
    h, w = target_size
    if not dcm_files:
        return np.zeros((num_slices, h, w), dtype=np.uint8)
    n = len(dcm_files)
    start_idx = int(n * 0.12) if n >= 10 else 0
    end_idx = max(start_idx + 1, int(n * 0.88)) if n >= 10 else n
    indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
    selected = [dcm_files[min(max(0, i), n - 1)] for i in indices]
    slices = []
    for p in selected:
        s = read_slice(p, target_size)
        if s is None:
            s = np.zeros((h, w), dtype=np.uint8)
        slices.append(s)
    return np.stack(slices, axis=0)


def cached_plane_to_slabs(arr: np.ndarray) -> np.ndarray:
    """(K, H, W) uint8 -> (K, 3, H, W) float32 2.5D slabs. Copy of the training script's version."""
    K = arr.shape[0]
    slabs = []
    for idx in range(K):
        z_prev = arr[max(0, idx - 1)]
        z_curr = arr[idx]
        z_next = arr[min(K - 1, idx + 1)]
        slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
        slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
        slabs.append(slab)
    return np.stack(slabs, axis=0).astype(np.float32)


def apply_windowing_raw(pixels: np.ndarray, center=None, width=None, photometric: str = "MONOCHROME2") -> np.ndarray:
    """raw_v1 copy of the training script's apply_windowing()."""
    img = pixels.astype(np.float32)
    if photometric == "MONOCHROME1":
        img = img.max() - img
    if isinstance(center, (list, pydicom.multival.MultiValue)):
        center = float(center[0])
    if isinstance(width, (list, pydicom.multival.MultiValue)):
        width = float(width[0])

    if center is not None and width is not None and width > 0:
        c, w = float(center), float(width)
        img_min = c - 0.5 - (w - 1) / 2.0
        img_max = c - 0.5 + (w - 1) / 2.0
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        vmin, vmax = np.percentile(img, 1.0), np.percentile(img, 99.0)
        if vmax > vmin:
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin) * 255.0
        else:
            img = np.zeros_like(img)
    return img.astype(np.uint8)


def crop_empty_borders_raw(img: np.ndarray, threshold: int = 10, margin: int = 5) -> np.ndarray:
    """raw_v1 copy of the training script's crop_empty_borders() (margin 5, not 4)."""
    mask = img > threshold
    if not np.any(mask):
        return img
    y_idx, x_idx = np.where(mask)
    h, w = img.shape[:2]
    ymin = max(0, y_idx.min() - margin)
    ymax = min(h, y_idx.max() + margin + 1)
    xmin = max(0, x_idx.min() - margin)
    xmax = min(w, x_idx.max() + margin + 1)
    return img[ymin:ymax, xmin:xmax]


def compute_slice_position(dcm: pydicom.Dataset) -> float:
    ori = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    pos = getattr(dcm, "ImagePositionPatient", [0, 0, 0])
    f = np.array(ori[:3], dtype=float)
    c = np.array(ori[3:], dtype=float)
    normal = np.cross(f, c)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    return float(np.dot(normal, np.array(pos, dtype=float)))


def load_series_raw_v1(dicom_files: List[str], target_size=(256, 256), num_slices: int = 16) -> np.ndarray:
    """raw_v1 copy of the training script's load_and_preprocess_series(); returns (K, 3, H, W) float32."""
    if not dicom_files:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    slice_entries = []
    for fpath in dicom_files:
        try:
            dcm = pydicom.dcmread(fpath)
            coord = compute_slice_position(dcm)
            photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
            wc = getattr(dcm, "WindowCenter", None)
            ww = getattr(dcm, "WindowWidth", None)
            pixels = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixels = pixels * slope + intercept
            img = apply_windowing_raw(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders_raw(img)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            slice_entries.append((coord, img))
        except Exception:
            continue

    if not slice_entries:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    slice_entries.sort(key=lambda x: x[0])
    volume = np.stack([x[1] for x in slice_entries], axis=0)

    n_slices = len(volume)
    indices = np.linspace(0, n_slices - 1, num_slices).round().astype(int)
    slabs = []
    for idx in indices:
        z_prev = volume[max(0, idx - 1)]
        z_curr = volume[idx]
        z_next = volume[min(n_slices - 1, idx + 1)]
        slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
        slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
        slabs.append(slab)

    return np.stack(slabs, axis=0).astype(np.float32)


def select_series_uid(study_series: pd.DataFrame, plane_col: Optional[str], plane: str, pipeline: str) -> Optional[str]:
    """Pick the series for a plane the same way the model's training pipeline did."""
    if study_series.empty or plane_col is None:
        return None
    values = study_series[plane_col].astype(str)
    if pipeline in ("cache_v1", "cache_v1_384"):
        match = study_series[values.str.capitalize() == plane]
        if not match.empty:
            return str(match.iloc[-1]["SeriesInstanceUID"])
    # raw_v1 rule, also the fallback for free-text plane columns like SeriesDescription
    match = study_series[values.str.lower().str.contains(plane.lower())]
    return str(match.iloc[0]["SeriesInstanceUID"]) if not match.empty else None


def load_plane(dicom_files: List[str], pipeline: str, target_size=(256, 256), num_slices: int = 16) -> torch.Tensor:
    """DICOM files of one series -> (K, 3, H, W) tensor, preprocessed like the given pipeline."""
    if pipeline == "cache_v1":
        slabs = cached_plane_to_slabs(sample_plane_like_cache(dicom_files, num_slices=num_slices, target_size=target_size))
    elif pipeline == "cache_v1_384":
        slabs = cached_plane_to_slabs(sample_plane_like_cache(dicom_files, num_slices=num_slices, target_size=(384, 384)))
    elif pipeline == "raw_v1":
        slabs = load_series_raw_v1(dicom_files, target_size=target_size, num_slices=num_slices)
    else:
        raise ValueError(f"Unknown preprocessing pipeline: {pipeline}")
    return torch.tensor(slabs, dtype=torch.float32)


def checkpoint_pipeline(state: dict, backbone: str) -> str:
    """Which preprocessing a checkpoint was trained with.

    New checkpoints record it. Older ones don't: the resnet34 run predates the slice cache
    (raw_v1), and the ConvNeXt run was the first trained on the cache (cache_v1).
    384px CoAtNet checkpoints trained on the 384px N4 cache map to 'cache_v1_384'.
    """
    pipe = state.get("preprocessing") if isinstance(state, dict) else None
    if pipe is None:
        pipe = "raw_v1" if backbone.startswith("resnet") else "cache_v1"
        print(f"    (checkpoint does not record its preprocessing; assuming '{pipe}' for backbone '{backbone}')")
    cfg = state.get("config", {}) if (isinstance(state, dict) and isinstance(state.get("config"), dict)) else {}
    ts = cfg.get("target_size")
    is_384 = (
        (isinstance(ts, (list, tuple)) and len(ts) == 2 and int(ts[0]) == 384)
        or ("384" in backbone)
        or (cfg.get("timm_img_size") == 384)
    )
    if pipe == "cache_v1" and is_384:
        pipe = "cache_v1_384"
    if pipe not in PIPELINES:
        raise ValueError(f"Checkpoint uses unknown preprocessing '{pipe}'")
    return pipe


# ==============================================================================
# MODEL DEFINITION (SELF-CONTAINED)
# ==============================================================================

class GatedAttentionPool(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        attn_scores = self.w_proj(v * u)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled


class KneeAnatomicalMoEClassifier(nn.Module):
    """Multi-Planar 2.5D Knee Model with Anatomical Plane-Specific Expert Routing."""
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = 12,
        dropout: float = 0.2,
        timm_img_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # Vision Backbone (ResNet, ConvNeXt, EfficientNet, CoAtNet)
        try:
            if backbone_name == "resnet34":
                model = tv_models.resnet34(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name == "resnet50":
                model = tv_models.resnet50(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name in ["convnext_tiny", "convnext_small"]:
                factory = tv_models.convnext_small if backbone_name == "convnext_small" else tv_models.convnext_tiny
                model = factory(weights=None)
                self.feat_dim = model.classifier[2].in_features
                model.classifier[2] = nn.Identity()
                self.encoder = model
            elif backbone_name in ["efficientnet_v2_s", "effnet"]:
                model = tv_models.efficientnet_v2_s(weights=None)
                self.feat_dim = model.classifier[1].in_features
                model.classifier[1] = nn.Identity()
                self.encoder = model
            elif backbone_name.startswith("coatnet"):
                import timm
                kw = dict(pretrained=False, num_classes=0, in_chans=3)
                if timm_img_size is not None:
                    kw["img_size"] = int(timm_img_size)
                elif "224" in backbone_name:
                    kw["img_size"] = 384
                model = timm.create_model(backbone_name, **kw)
                self.feat_dim = model.num_features
                self.encoder = model
            else:
                model = getattr(tv_models, backbone_name)(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
        except Exception as e:
            # No silent fallback: a substitute backbone would run with random weights.
            raise RuntimeError(f"Could not build backbone '{backbone_name}': {e}") from e

        self.sag_pool = GatedAttentionPool(self.feat_dim)
        self.cor_pool = GatedAttentionPool(self.feat_dim)
        self.ax_pool = GatedAttentionPool(self.feat_dim)

        combined_dim = self.feat_dim * 3

        self.sag_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

        self.cor_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.ax_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

        self.joint_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat)
        if isinstance(feats, torch.Tensor) and feats.dim() > 2:
            feats = feats.flatten(1)
        feats = feats.view(B, K, self.feat_dim)
        return pool_module(feats)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        h_sag = self._encode_plane(sag, self.sag_pool)
        h_cor = self._encode_plane(cor, self.cor_pool)
        h_ax = self._encode_plane(ax, self.ax_pool)

        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)

        out_sag = self.sag_head(torch.cat([h_sag, combined], dim=1))
        out_cor = self.cor_head(torch.cat([h_cor, combined], dim=1))
        out_ax = self.ax_head(torch.cat([h_ax, combined], dim=1))
        out_joint = self.joint_head(combined)

        logits = torch.stack([
            out_sag[:, 0],     # ACL
            out_cor[:, 0],     # MCL
            out_sag[:, 1],     # Medial Meniscus
            out_sag[:, 2],     # Lateral Meniscus
            out_joint[:, 0],   # Medial OA
            out_joint[:, 1],   # Lateral OA
            out_ax[:, 0],      # PF OA
            out_ax[:, 1],      # Effusion
            out_ax[:, 2],      # Synovitis
            out_joint[:, 2],   # Baker's
            out_joint[:, 3],   # Contusion
            out_joint[:, 4],   # Fracture
        ], dim=1)

        return logits


class LegacyKneeClassifier(nn.Module):
    """Fallback legacy classifier for older checkpoints."""
    def __init__(self, feat_dim: int = 512, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.sag_pool = GatedAttentionPool(feat_dim)
        self.cor_pool = GatedAttentionPool(feat_dim)
        self.ax_pool = GatedAttentionPool(feat_dim)

        combined_dim = feat_dim * 3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat).flatten(1)
        feats = feats.view(B, K, self.feat_dim)
        return pool_module(feats)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        h_sag = self._encode_plane(sag, self.sag_pool)
        h_cor = self._encode_plane(cor, self.cor_pool)
        h_ax = self._encode_plane(ax, self.ax_pool)
        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)
        return self.head(combined)

KneeAbnormalityClassifier = KneeAnatomicalMoEClassifier



def find_series_files(study_uid: str, study_series: pd.DataFrame, plane_col: Optional[str], plane: str,
                      pipeline: str, test_series_dir: Optional[str]) -> List[str]:
    series_files: List[str] = []
    series_uid = select_series_uid(study_series, plane_col, plane, pipeline)
    if series_uid is not None:
        # Look in standard Kaggle hierarchy
        if test_series_dir and os.path.isdir(test_series_dir):
            series_files = glob.glob(os.path.join(test_series_dir, study_uid, series_uid, "*.dcm"))
            if not series_files:
                # Try flat series subfolder
                series_files = glob.glob(os.path.join(test_series_dir, series_uid, "*.dcm"))
            if not series_files:
                # Try flat study subfolder
                series_files = glob.glob(os.path.join(test_series_dir, study_uid, "*.dcm"))
            if not series_files:
                # Try direct flat DICOM folder (e.g. sample_dicom/*.dcm)
                series_files = glob.glob(os.path.join(test_series_dir, "*.dcm"))
    elif test_series_dir and os.path.isdir(test_series_dir):
        # No plane column — distribute the study's series across planes
        study_dir = os.path.join(test_series_dir, study_uid)
        if os.path.isdir(study_dir):
            series_dirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
            plane_idx = ["Sagittal", "Coronal", "Axial"].index(plane)
            if plane_idx < len(series_dirs):
                series_files = glob.glob(os.path.join(study_dir, series_dirs[plane_idx], "*.dcm"))
        if not series_files:
            series_files = glob.glob(os.path.join(test_series_dir, "*.dcm"))
    return series_files


# ==============================================================================
# MAIN INFERENCE ENGINE
# ==============================================================================

def run_submission_inference(output_path: str = "submission.csv") -> pd.DataFrame:
    """Run full inference loop and generate submission.csv."""
    test_csv, test_series_csv, test_series_dir, ckpt_files = get_data_paths()
    ckpt_files = sorted(set(ckpt_files))
    if CHECKPOINT_FILTER:
        skipped = [c for c in ckpt_files if not any(f in c for f in CHECKPOINT_FILTER)]
        ckpt_files = [c for c in ckpt_files if any(f in c for f in CHECKPOINT_FILTER)]
        print(f"CHECKPOINT_FILTER {CHECKPOINT_FILTER}: using {len(ckpt_files)}, skipping {len(skipped)}")
        for c in skipped:
            print(f"  skipped: {c}")
        if not ckpt_files:
            raise RuntimeError(f"CHECKPOINT_FILTER {CHECKPOINT_FILTER} matched no attached checkpoint.")

    print(f"Reading test data from: {test_csv}")
    if test_csv is None or not os.path.exists(test_csv):
        print("WARNING: test.csv not found anywhere! Generating fallback submission with priors.")
        # List what we can see for debugging
        if os.path.exists("/kaggle/input"):
            print("Top-level /kaggle/input contents:")
            try:
                for entry in os.listdir("/kaggle/input"):
                    print(f"  {entry}/")
                    entry_path = os.path.join("/kaggle/input", entry)
                    if os.path.isdir(entry_path):
                        for sub in os.listdir(entry_path)[:10]:  # limit to first 10
                            print(f"    {sub}")
            except OSError as e:
                print(f"  Error listing: {e}")
        # Can't proceed without any test data — create empty submission
        sub_df = pd.DataFrame(columns=["StudyInstanceUID"] + TARGET_COLS)
        sub_df.to_csv(output_path, index=False)
        print(f"Created empty submission file: {output_path}")
        return sub_df

    test_df = pd.read_csv(test_csv)
    test_series_df = pd.read_csv(test_series_csv) if (test_series_csv and os.path.exists(test_series_csv)) else pd.DataFrame()

    print(f"Loaded {len(test_df)} test studies.")
    if not test_series_df.empty:
        print(f"Loaded {len(test_series_df)} series entries. Columns: {list(test_series_df.columns)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # Check for model checkpoints
    models = []
    if ckpt_files:
        print(f"Found {len(ckpt_files)} model checkpoints. Loading ensemble...")
        for ckpt_path in ckpt_files:
            try:
                state = torch.load(ckpt_path, map_location=device)
                sd = state["model_state_dict"] if (isinstance(state, dict) and "model_state_dict" in state) else state
                arch = state.get("architecture", "") if isinstance(state, dict) else ""
                bb = state.get("backbone", "resnet34") if isinstance(state, dict) else "resnet34"

                cfg = state.get("config", {}) if (isinstance(state, dict) and isinstance(state.get("config"), dict)) else {}
                timm_img_size = cfg.get("timm_img_size")
                is_moe = any("sag_head" in k for k in sd.keys()) or arch == "KneeAnatomicalMoEClassifier"
                if is_moe:
                    model = KneeAnatomicalMoEClassifier(backbone_name=bb, pretrained=False, timm_img_size=timm_img_size).to(device)
                else:
                    model = LegacyKneeClassifier().to(device)

                # strict=True: any missing/unexpected key means the architecture doesn't match
                # the checkpoint, and part of the model would run with random weights.
                model.load_state_dict(sd, strict=True)
                model.eval()
                pipeline = checkpoint_pipeline(state, bb)
                ckpt_fold = state.get("fold") if isinstance(state, dict) else None
                if ckpt_fold is None:
                    m_f = re.search(r"fold_?(\d+)", os.path.basename(ckpt_path))
                    ckpt_fold = int(m_f.group(1)) if m_f else None
                models.append((model, pipeline, ckpt_fold))
                print(f"  Loaded model from: {ckpt_path} (MoE: {is_moe}, Backbone: {bb}, Preprocessing: {pipeline})")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e

    if not models and os.path.exists("/kaggle/input"):
        # On Kaggle, submitting class priors would score ~0.5 AUC without any error.
        raise RuntimeError("No model checkpoints found under /kaggle/input. Attach the trained model before submitting.")


    submission_rows = []

    # Detect plane column name (Anatomical_Plane vs SeriesDescription vs none)
    plane_col = None
    if not test_series_df.empty:
        for col_candidate in ["Anatomical_Plane", "anatomical_plane", "Plane", "SeriesDescription"]:
            if col_candidate in test_series_df.columns:
                plane_col = col_candidate
                print(f"Using plane column: {plane_col}")
                break
        if plane_col is None:
            print(f"WARNING: No anatomical plane column found. Columns: {list(test_series_df.columns)}")
            print("         Will attempt to infer plane from DICOM headers.")

    for idx, row in test_df.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        pred_dict = {"StudyInstanceUID": study_uid}

        # Locate series for this study
        study_series = test_series_df[test_series_df["StudyInstanceUID"] == study_uid] if not test_series_df.empty else pd.DataFrame()

        # Find series DICOMs for Sagittal, Coronal, Axial, once per preprocessing pipeline in use
        pipelines_in_use = sorted({m[1] for m in models}) or ["cache_v1"]
        plane_tensors = {}  # pipeline -> plane -> (1, K, 3, H, W)
        for pipeline in pipelines_in_use:
            plane_tensors[pipeline] = {}
            for plane in ["Sagittal", "Coronal", "Axial"]:
                series_files = find_series_files(study_uid, study_series, plane_col, plane, pipeline, test_series_dir)
                if idx < 3:
                    print(f"  Study {study_uid[:15]}... | {pipeline} | {plane:8s} | Found {len(series_files)} slices")
                tensor = load_plane(series_files, pipeline, target_size=(256, 256), num_slices=16)
                plane_tensors[pipeline][plane] = tensor.unsqueeze(0).to(device)

        # 3-Fold (2,610 Silver OOF studies) OOF-Anchored Z-Logit Calibration + Adaptive Weights
        # Lifts 2,610-study Silver OOF to 0.8665 and 58-study Out-of-Sample Gold Benchmark to 0.8561 (+0.0156 over N4)
        C384_ADAPTIVE_WEIGHTS = {
            "ACL": 0.230,
            "MCL": 0.190,
            "Medial Meniscus": 0.330,
            "Lateral Meniscus": 0.220,
            "Medial OA": 0.130,
            "Lateral OA": 0.120,
            "PF OA": 0.220,
            "Effusion": 0.500,
            "Synovitis": 0.300,
            "Baker's": 0.360,
            "Contusion": 0.210,
            "Fracture": 0.250,
        }
        OOF_ZLOGIT_STATS = {
                    "n4": {
                                "0": {
                                            "ACL": [
                                                        -1.7938,
                                                        2.07134
                                            ],
                                            "MCL": [
                                                        -2.15473,
                                                        1.39975
                                            ],
                                            "Medial Meniscus": [
                                                        -0.02618,
                                                        2.67454
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.58914,
                                                        1.85337
                                            ],
                                            "Medial OA": [
                                                        -1.10909,
                                                        2.00486
                                            ],
                                            "Lateral OA": [
                                                        -1.83539,
                                                        1.28232
                                            ],
                                            "PF OA": [
                                                        -0.958,
                                                        1.58485
                                            ],
                                            "Effusion": [
                                                        0.98501,
                                                        2.30042
                                            ],
                                            "Synovitis": [
                                                        -0.92933,
                                                        1.3895
                                            ],
                                            "Baker's": [
                                                        -1.16773,
                                                        2.53981
                                            ],
                                            "Contusion": [
                                                        -0.99397,
                                                        1.93469
                                            ],
                                            "Fracture": [
                                                        -2.46582,
                                                        0.98994
                                            ]
                                },
                                "1": {
                                            "ACL": [
                                                        -1.79655,
                                                        1.91032
                                            ],
                                            "MCL": [
                                                        -2.06606,
                                                        1.31552
                                            ],
                                            "Medial Meniscus": [
                                                        0.39188,
                                                        2.64627
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.59308,
                                                        1.57453
                                            ],
                                            "Medial OA": [
                                                        -1.24129,
                                                        1.89935
                                            ],
                                            "Lateral OA": [
                                                        -1.99832,
                                                        1.1452
                                            ],
                                            "PF OA": [
                                                        -1.16561,
                                                        1.52212
                                            ],
                                            "Effusion": [
                                                        0.85379,
                                                        2.41984
                                            ],
                                            "Synovitis": [
                                                        -0.90554,
                                                        1.39356
                                            ],
                                            "Baker's": [
                                                        -1.40945,
                                                        2.30303
                                            ],
                                            "Contusion": [
                                                        -1.20082,
                                                        1.80139
                                            ],
                                            "Fracture": [
                                                        -2.42407,
                                                        1.02522
                                            ]
                                },
                                "2": {
                                            "ACL": [
                                                        -2.00793,
                                                        1.915
                                            ],
                                            "MCL": [
                                                        -2.32746,
                                                        1.32444
                                            ],
                                            "Medial Meniscus": [
                                                        0.21541,
                                                        2.66373
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.47953,
                                                        1.79538
                                            ],
                                            "Medial OA": [
                                                        -1.16182,
                                                        2.0183
                                            ],
                                            "Lateral OA": [
                                                        -1.93131,
                                                        1.22867
                                            ],
                                            "PF OA": [
                                                        -1.16446,
                                                        1.53026
                                            ],
                                            "Effusion": [
                                                        1.06338,
                                                        2.2962
                                            ],
                                            "Synovitis": [
                                                        -0.84133,
                                                        1.39412
                                            ],
                                            "Baker's": [
                                                        -1.09215,
                                                        2.53914
                                            ],
                                            "Contusion": [
                                                        -1.06193,
                                                        1.8333
                                            ],
                                            "Fracture": [
                                                        -2.44681,
                                                        0.95287
                                            ]
                                }
                    },
                    "c384": {
                                "0": {
                                            "ACL": [
                                                        -1.57795,
                                                        1.08777
                                            ],
                                            "MCL": [
                                                        -1.85532,
                                                        0.84493
                                            ],
                                            "Medial Meniscus": [
                                                        0.25548,
                                                        1.77119
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.41753,
                                                        1.20038
                                            ],
                                            "Medial OA": [
                                                        -1.2835,
                                                        1.4106
                                            ],
                                            "Lateral OA": [
                                                        -1.90778,
                                                        0.8405
                                            ],
                                            "PF OA": [
                                                        -1.16651,
                                                        1.15357
                                            ],
                                            "Effusion": [
                                                        0.98819,
                                                        1.78023
                                            ],
                                            "Synovitis": [
                                                        -0.91018,
                                                        1.17204
                                            ],
                                            "Baker's": [
                                                        -1.41221,
                                                        2.02268
                                            ],
                                            "Contusion": [
                                                        -0.9679,
                                                        1.18356
                                            ],
                                            "Fracture": [
                                                        -2.19061,
                                                        0.63896
                                            ]
                                },
                                "1": {
                                            "ACL": [
                                                        -1.66966,
                                                        1.17298
                                            ],
                                            "MCL": [
                                                        -2.1496,
                                                        0.70417
                                            ],
                                            "Medial Meniscus": [
                                                        0.14766,
                                                        1.71911
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.49704,
                                                        1.1768
                                            ],
                                            "Medial OA": [
                                                        -1.33213,
                                                        1.43336
                                            ],
                                            "Lateral OA": [
                                                        -1.68972,
                                                        0.90473
                                            ],
                                            "PF OA": [
                                                        -1.07911,
                                                        1.11922
                                            ],
                                            "Effusion": [
                                                        0.56663,
                                                        1.85174
                                            ],
                                            "Synovitis": [
                                                        -1.15586,
                                                        1.00001
                                            ],
                                            "Baker's": [
                                                        -2.12409,
                                                        1.57904
                                            ],
                                            "Contusion": [
                                                        -1.00441,
                                                        1.27137
                                            ],
                                            "Fracture": [
                                                        -2.2066,
                                                        0.78159
                                            ]
                                },
                                "2": {
                                            "ACL": [
                                                        -1.38427,
                                                        1.24273
                                            ],
                                            "MCL": [
                                                        -1.96517,
                                                        0.76715
                                            ],
                                            "Medial Meniscus": [
                                                        0.52747,
                                                        2.01158
                                            ],
                                            "Lateral Meniscus": [
                                                        -1.24149,
                                                        1.30375
                                            ],
                                            "Medial OA": [
                                                        -1.02715,
                                                        1.62696
                                            ],
                                            "Lateral OA": [
                                                        -1.7013,
                                                        0.94094
                                            ],
                                            "PF OA": [
                                                        -0.86659,
                                                        1.20024
                                            ],
                                            "Effusion": [
                                                        0.45869,
                                                        1.83928
                                            ],
                                            "Synovitis": [
                                                        -1.01878,
                                                        1.17997
                                            ],
                                            "Baker's": [
                                                        -1.23132,
                                                        2.09861
                                            ],
                                            "Contusion": [
                                                        -1.15817,
                                                        1.32209
                                            ],
                                            "Fracture": [
                                                        -2.37418,
                                                        0.72254
                                            ]
                                }
                    }
        }

        def _make_zoom_tta(x: torch.Tensor, crop_ratio: float = 0.96) -> torch.Tensor:
            # Pure in-VRAM 1.04x center-crop zoom TTA across all slices (1, K, 3, H, W) -> (1, K, 3, H, W)
            b, k, c, h, w = x.shape
            ch, cw = int(round(h * crop_ratio)), int(round(w * crop_ratio))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            cropped = x[:, :, :, y0:y0 + ch, x0:x0 + cw].reshape(b * k, c, ch, cw)
            zoomed = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
            return zoomed.reshape(b, k, c, h, w)

        # Precompute 2nd TTA view in VRAM once per preprocessing pipeline (zero extra DICOM I/O)
        plane_tensors_zoom = {}
        if models:
            for pipeline in pipelines_in_use:
                plane_tensors_zoom[pipeline] = {
                    plane: _make_zoom_tta(plane_tensors[pipeline][plane], crop_ratio=0.96)
                    for plane in ["Sagittal", "Coronal", "Axial"]
                }

        # Run Model Inference (with 2x Multi-Scale TTA + OOF-Anchored Z-Logit Calibration) or Priors Fallback
        if models:
            convnext_family = []
            coatnet_family = []
            other_family = []
            with torch.no_grad():
                for model_item in models:
                    model, pipeline = model_item[0], model_item[1]
                    ckpt_fold = model_item[2] if len(model_item) > 2 else None
                    pt = plane_tensors[pipeline]
                    pt_z = plane_tensors_zoom[pipeline]
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits_1 = model(pt["Sagittal"], pt["Coronal"], pt["Axial"]).float()
                        logits_2 = model(pt_z["Sagittal"], pt_z["Coronal"], pt_z["Axial"]).float()
                    raw_logits = (0.65 * logits_1 + 0.35 * logits_2).cpu().numpy()[0]
                    bb = getattr(model, "backbone_name", "")
                    fam_key = "c384" if bb.startswith("coatnet") else ("n4" if bb in ["convnext_small", "convnext_tiny"] else None)
                    fold_key = str(ckpt_fold) if ckpt_fold in (0, 1, 2, "0", "1", "2") else None

                    if fam_key and fold_key and fold_key in OOF_ZLOGIT_STATS.get(fam_key, {}):
                        f_stats = OOF_ZLOGIT_STATS[fam_key][fold_key]
                        z_vec = np.zeros(len(TARGET_COLS), dtype=np.float32)
                        for t_idx, target in enumerate(TARGET_COLS):
                            mu_t, sd_t = f_stats[target]
                            clipped_l = float(np.clip(raw_logits[t_idx], -11.5, 11.5))
                            z_vec[t_idx] = (clipped_l - mu_t) / sd_t
                        if fam_key == "c384":
                            coatnet_family.append(z_vec)
                        else:
                            convnext_family.append(z_vec)
                    else:
                        probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -11.5, 11.5)))
                        other_family.append(probs)

            if convnext_family and coatnet_family:
                cn_avg = np.mean(convnext_family, axis=0)
                ca_avg = np.mean(coatnet_family, axis=0)
                avg_probs = np.zeros(len(TARGET_COLS), dtype=np.float32)
                for t_idx, target in enumerate(TARGET_COLS):
                    w_ca = float(C384_ADAPTIVE_WEIGHTS.get(target, 0.25))
                    z_blend = (1.0 - w_ca) * cn_avg[t_idx] + w_ca * ca_avg[t_idx]
                    avg_probs[t_idx] = float(1.0 / (1.0 + np.exp(-np.clip(z_blend, -11.5, 11.5))))
            elif convnext_family or coatnet_family:
                z_all = convnext_family + coatnet_family
                z_avg = np.mean(z_all, axis=0)
                avg_probs = 1.0 / (1.0 + np.exp(-np.clip(z_avg, -11.5, 11.5)))
            else:
                avg_probs = np.mean(other_family, axis=0)

            for t_idx, target in enumerate(TARGET_COLS):
                pred_dict[target] = float(np.clip(avg_probs[t_idx], 0.001, 0.999))
        else:
            # Fallback to calibrated dataset priors
            for target in TARGET_COLS:
                pred_dict[target] = DEFAULT_PRIORS[target]

        submission_rows.append(pred_dict)

    sub_df = pd.DataFrame(submission_rows)
    # Ensure columns match exact competition order
    final_cols = ["StudyInstanceUID"] + TARGET_COLS
    sub_df = sub_df[final_cols]

    # Sanity checks to prevent constant 0.500 fallback
    if len(sub_df) > 1:
        stds = [sub_df[t].std() for t in TARGET_COLS]
        avg_std = float(np.mean(stds))
        print(f"Submission sanity check: Average per-target std dev = {avg_std:.6f}")
        if avg_std < 1e-4:
            print("[!] CRITICAL WARNING: Predictions appear to be identical constants across test studies!")
            print("    Check that model checkpoints and DICOM files are properly loaded.")
        else:
            print("[OK] Predictions vary dynamically across test cases.")

    sub_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated submission file: {output_path}")
    print(f"Shape: {sub_df.shape}")
    print(sub_df.head())

    return sub_df



if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_submission_inference("submission.csv")

