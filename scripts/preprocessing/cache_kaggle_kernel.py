"""Kaggle Caching Kernel for RSNA Knee Abnormality Detection.

Runs on Kaggle to pre-process all 4,407 raw DICOM training studies into compact,
fast-loading .npz arrays. Saves output to /kaggle/working/cached_slices/.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm


def get_paths() -> Tuple[str, str, str]:
    candidate_roots = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "data/raw"
    ]
    series_csv, series_dir = None, None
    for root in candidate_roots:
        sc = os.path.join(root, "train_series.csv")
        sd = os.path.join(root, "train_series")
        if os.path.isfile(sc) and os.path.isdir(sd):
            series_csv, series_dir = sc, sd
            break

    output_dir = "/kaggle/working/cached_slices" if os.path.exists("/kaggle") else "data/processed/cached_slices"
    return series_csv, series_dir, output_dir


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


def get_sorted_files(s_dir: str) -> List[str]:
    if not os.path.exists(s_dir):
        return []
    files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.lower().endswith(".dcm")]
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


def process_plane(s_dir: str, num_slices: int, target_size: Tuple[int, int]) -> np.ndarray:
    dcm_files = get_sorted_files(s_dir)
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


def process_study(study_uid: str, plane_series: Dict[str, str], series_root: str, out_dir: str, num_slices: int = 16, target_size: Tuple[int, int] = (256, 256)):
    out_file = os.path.join(out_dir, f"{study_uid}.npz")
    if os.path.exists(out_file):
        return True
    try:
        data = {}
        for plane in ["Sagittal", "Coronal", "Axial"]:
            s_uid = plane_series.get(plane)
            found_dir = None
            if s_uid:
                c1 = os.path.join(series_root, study_uid, s_uid)
                c2 = os.path.join(series_root, s_uid)
                if os.path.isdir(c1):
                    found_dir = c1
                elif os.path.isdir(c2):
                    found_dir = c2
            if found_dir is None:
                c3 = os.path.join(series_root, study_uid)
                if os.path.isdir(c3):
                    found_dir = c3
            if found_dir:
                data[plane.lower()] = process_plane(found_dir, num_slices, target_size)
            else:
                data[plane.lower()] = np.zeros((num_slices, target_size[0], target_size[1]), dtype=np.uint8)

        np.savez_compressed(out_file, sagittal=data["sagittal"], coronal=data["coronal"], axial=data["axial"])
        return True
    except Exception:
        return False


def main():
    series_csv, series_dir, out_dir = get_paths()
    if not series_csv or not os.path.exists(series_csv):
        print("Error: train_series.csv not found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(series_csv)
    p_col = "Anatomical_Plane" if "Anatomical_Plane" in df.columns else "Plane"
    df["norm_plane"] = df[p_col].astype(str).str.capitalize()

    studies: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        suid = str(row["StudyInstanceUID"])
        if suid not in studies:
            studies[suid] = {}
        plane = row["norm_plane"]
        if plane in ["Sagittal", "Coronal", "Axial"]:
            studies[suid][plane] = str(row["SeriesInstanceUID"])

    study_list = list(studies.keys())
    print(f"Loaded {len(study_list)} studies for caching into {out_dir}")

    workers = max(1, (os.cpu_count() or 4) - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_study, suid, studies[suid], series_dir, out_dir, 16, (256, 256)): suid for suid in study_list}
        ok = 0
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(study_list), desc="Caching")
        for fut in pbar:
            if fut.result():
                ok += 1
            pbar.set_postfix({"Cached": ok})

    print(f"Successfully cached {ok}/{len(study_list)} studies into {out_dir}")


if __name__ == "__main__":
    main()
