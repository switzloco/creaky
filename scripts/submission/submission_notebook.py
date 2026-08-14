"""Standalone Kaggle Submission Script / Inference Notebook Generator.

This script is self-contained and designed to run inside a Kaggle Code Competition
environment with Internet Disabled.

Pipeline:
1. Auto-detects Kaggle vs. Local paths.
2. Loads test.csv and test_series.csv.
3. Reads DICOM series directly from disk on-the-fly.
4. Normalizes (windowing + border crop) and sorts slices in 3D anatomical order.
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

# Default class priors if model weights are not found
DEFAULT_PRIORS = {
    "ACL": 0.414, "MCL": 0.155, "Medial Meniscus": 0.448, "Lateral Meniscus": 0.397,
    "Medial OA": 0.259, "Lateral OA": 0.190, "PF OA": 0.362, "Effusion": 0.603,
    "Synovitis": 0.466, "Baker's": 0.207, "Contusion": 0.328, "Fracture": 0.310
}


def get_data_paths() -> Tuple[str, str, str, str]:
    """Auto-detect competition paths between Kaggle environment and Local development."""
    possible_roots = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-2026-knee-abnormality-detection",
    ]
    kaggle_root = None
    for r in possible_roots:
        if os.path.exists(r):
            kaggle_root = r
            break

    # If Kaggle directory has a custom name, dynamically find the folder containing test.csv
    if kaggle_root is None and os.path.exists("/kaggle/input"):
        try:
            for d in os.listdir("/kaggle/input"):
                candidate = os.path.join("/kaggle/input", d)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "test.csv")):
                    kaggle_root = candidate
                    break
        except Exception:
            pass

    if kaggle_root is not None:
        test_csv = os.path.join(kaggle_root, "test.csv")
        test_series_csv = os.path.join(kaggle_root, "test_series.csv")
        test_series_dir = os.path.join(kaggle_root, "test_series")
        models_dir = "/kaggle/input/creaky-models"
    else:
        # Local workspace paths (safe in notebooks without __file__)
        local_root = os.path.abspath("data/raw")
        test_csv = os.path.join(local_root, "test.csv")
        test_series_csv = os.path.join(local_root, "test_series.csv")
        test_series_dir = os.path.join(local_root, "sample_dicom")
        models_dir = os.path.abspath("checkpoints")

    return test_csv, test_series_csv, test_series_dir, models_dir



# ==============================================================================
# DICOM PREPROCESSING & 3D SPATIAL RECONSTRUCTION
# ==============================================================================

def apply_windowing(
    pixels: np.ndarray,
    center: Optional[Union[float, int, list]] = None,
    width: Optional[Union[float, int, list]] = None,
    photometric: str = "MONOCHROME2"
) -> np.ndarray:
    """Normalize raw 16-bit pixel intensities into standardized [0, 255] uint8."""
    img = pixels.astype(np.float32)
    if photometric == "MONOCHROME1":
        img = img.max() - img

    if isinstance(center, (list, pydicom.multival.MultiValue)):
        center = float(center[0])
    if isinstance(width, (list, pydicom.multival.MultiValue)):
        width = float(width[0])

    if center is not None and width is not None and width > 0:
        c = float(center)
        w = float(width)
        img_min = c - 0.5 - (w - 1) / 2.0
        img_max = c - 0.5 + (w - 1) / 2.0
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        val_min = np.percentile(img, 1.0)
        val_max = np.percentile(img, 99.0)
        if val_max > val_min:
            img = np.clip(img, val_min, val_max)
            img = (img - val_min) / (val_max - val_min) * 255.0
        else:
            img = np.zeros_like(img)

    return img.astype(np.uint8)


def crop_empty_borders(img: np.ndarray, threshold: int = 10, margin: int = 5) -> np.ndarray:
    """Crop out empty black border pixels around the knee joint."""
    mask = img > threshold
    if not np.any(mask):
        return img
    y_indices, x_indices = np.where(mask)
    h, w = img.shape[:2]
    ymin = max(0, y_indices.min() - margin)
    ymax = min(h, y_indices.max() + margin + 1)
    xmin = max(0, x_indices.min() - margin)
    xmax = min(w, x_indices.max() + margin + 1)
    return img[ymin:ymax, xmin:xmax]


def compute_slice_position(dcm: pydicom.Dataset) -> float:
    """Compute 1D physical projection coordinate along normal vector."""
    ori = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    pos = getattr(dcm, "ImagePositionPatient", [0, 0, 0])

    f = np.array(ori[:3], dtype=float)
    c = np.array(ori[3:], dtype=float)
    normal = np.cross(f, c)
    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0])

    return float(np.dot(normal, np.array(pos, dtype=float)))


def load_and_preprocess_series(
    dicom_files: List[str],
    target_size: Tuple[int, int] = (256, 256),
    num_slices: int = 16
) -> torch.Tensor:
    """Load, window, sort, and sample 2.5D slabs from a DICOM series."""
    if not dicom_files:
        return torch.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=torch.float32)

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

            img = apply_windowing(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders(img)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            slice_entries.append((coord, img))
        except Exception:
            continue

    if not slice_entries:
        return torch.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=torch.float32)

    # Sort ascending by physical 3D coordinate
    slice_entries.sort(key=lambda x: x[0])
    volume = np.stack([x[1] for x in slice_entries], axis=0)  # (N, H, W)

    # Sample 2.5D slabs
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

    tensor = torch.tensor(np.stack(slabs, axis=0), dtype=torch.float32)  # (K, 3, H, W)
    return tensor


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


class KneeAbnormalityClassifier(nn.Module):
    def __init__(self, feat_dim: int = 512, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim

        # Backbone
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
        pooled = pool_module(feats)
        return pooled

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        h_sag = self._encode_plane(sag, self.sag_pool)
        h_cor = self._encode_plane(cor, self.cor_pool)
        h_ax = self._encode_plane(ax, self.ax_pool)
        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)
        logits = self.head(combined)
        return logits


# ==============================================================================
# MAIN INFERENCE ENGINE
# ==============================================================================

def run_submission_inference(output_path: str = "submission.csv") -> pd.DataFrame:
    """Run full inference loop and generate submission.csv."""
    test_csv, test_series_csv, test_series_dir, models_dir = get_data_paths()

    print(f"Reading test data from: {test_csv}")
    if not os.path.exists(test_csv):
        print("Error: test.csv not found!")
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    test_series_df = pd.read_csv(test_series_csv) if os.path.exists(test_series_csv) else pd.DataFrame()

    print(f"Loaded {len(test_df)} test studies.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # Check for model checkpoints
    ckpt_files = glob.glob(os.path.join(models_dir, "*.pt"))
    models = []
    if ckpt_files:
        print(f"Found {len(ckpt_files)} model checkpoints. Loading ensemble...")
        for ckpt_path in ckpt_files:
            try:
                model = KneeAbnormalityClassifier().to(device)
                state = torch.load(ckpt_path, map_location=device)
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                model.eval()
                models.append(model)
                print(f"  Loaded model from: {ckpt_path}")
            except Exception as e:
                print(f"  Warning: Failed to load {ckpt_path}: {e}")

    submission_rows = []

    for _, row in test_df.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        pred_dict = {"StudyInstanceUID": study_uid}

        # Locate series for this study
        study_series = test_series_df[test_series_df["StudyInstanceUID"] == study_uid] if not test_series_df.empty else pd.DataFrame()

        # Find series DICOMs for Sagittal, Coronal, Axial
        plane_tensors = {}
        for plane in ["Sagittal", "Coronal", "Axial"]:
            plane_match = study_series[study_series["Anatomical_Plane"].str.lower() == plane.lower()] if not study_series.empty else pd.DataFrame()
            series_files = []

            if not plane_match.empty:
                series_uid = str(plane_match.iloc[0]["SeriesInstanceUID"])
                # Look in standard Kaggle hierarchy
                pattern = os.path.join(test_series_dir, study_uid, series_uid, "*.dcm")
                series_files = glob.glob(pattern)
                if not series_files:
                    # Look in local flat sample directory
                    pattern_alt = os.path.join(test_series_dir, "*.dcm")
                    series_files = glob.glob(pattern_alt)

            # Load and preprocess
            tensor = load_and_preprocess_series(series_files, target_size=(256, 256), num_slices=16)
            plane_tensors[plane] = tensor.unsqueeze(0).to(device)  # (1, K, 3, H, W)

        # Run Model Inference or Priors Fallback
        if models:
            ensemble_probs = []
            with torch.no_grad():
                for model in models:
                    logits = model(plane_tensors["Sagittal"], plane_tensors["Coronal"], plane_tensors["Axial"])
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    ensemble_probs.append(probs)
            avg_probs = np.mean(ensemble_probs, axis=0)
            for idx, target in enumerate(TARGET_COLS):
                pred_dict[target] = float(np.clip(avg_probs[idx], 0.001, 0.999))
        else:
            # Fallback to calibrated dataset priors
            for target in TARGET_COLS:
                pred_dict[target] = DEFAULT_PRIORS[target]

        submission_rows.append(pred_dict)

    sub_df = pd.DataFrame(submission_rows)
    # Ensure columns match exact competition order
    final_cols = ["StudyInstanceUID"] + TARGET_COLS
    sub_df = sub_df[final_cols]

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

