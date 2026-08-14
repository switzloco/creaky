"""Batch converter for DICOM series into standardized volumes (.npy) or PNG image slices."""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dicom_reader import read_dicom_slice
from series_sorter import sort_series_slices, compute_slice_position, determine_plane_from_orientation


def process_dicom_series(
    dicom_file_paths: List[str],
    target_size: Tuple[int, int] = (256, 256),
    crop_border: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process a collection of DICOM slice files belonging to a single series.

    Returns:
    - volume: np.ndarray of shape (N_slices, H, W) in uint8
    - series_info: dict summarizing metadata
    """
    slice_data = []

    for path in dicom_file_paths:
        try:
            img, meta = read_dicom_slice(path, target_size=target_size, crop_border=crop_border)
            meta["_img"] = img
            meta["_file_path"] = path
            slice_data.append(meta)
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")

    if not slice_data:
        return np.empty((0, target_size[0], target_size[1]), dtype=np.uint8), {}

    # Sort slices in 3D anatomical order
    sorted_slices = sort_series_slices(slice_data)

    # Stack images into 3D volume
    volume = np.stack([s["_img"] for s in sorted_slices], axis=0)

    first_meta = sorted_slices[0]
    series_info = {
        "StudyInstanceUID": first_meta.get("StudyInstanceUID", ""),
        "SeriesInstanceUID": first_meta.get("SeriesInstanceUID", ""),
        "SeriesDescription": first_meta.get("SeriesDescription", ""),
        "Anatomical_Plane": first_meta.get("Anatomical_Plane", ""),
        "Num_Slices": len(sorted_slices),
        "Slice_Height": volume.shape[1],
        "Slice_Width": volume.shape[2],
        "SliceThickness": first_meta.get("SliceThickness", 1.0),
        "PixelSpacing": first_meta.get("PixelSpacing", [1.0, 1.0])
    }

    return volume, series_info


def convert_dataset_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    output_format: str = "npy"
) -> pd.DataFrame:
    """Scan directory for DICOM files, group by SeriesInstanceUID, and export processed volumes."""
    print(f"Scanning for DICOM files in: {input_dir}")
    dicom_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    print(f"Found {len(dicom_files)} DICOM files.")
    if not dicom_files:
        return pd.DataFrame()

    # Group by series (extract header quickly or group during read)
    # Read headers to group
    series_groups: Dict[str, List[str]] = {}
    print("Grouping files by SeriesInstanceUID...")
    for fpath in dicom_files:
        # Check parent folder or read DICOM tag
        try:
            import pydicom
            hdr = pydicom.dcmread(fpath, stop_before_pixels=True)
            series_uid = str(getattr(hdr, "SeriesInstanceUID", "unknown_series"))
            study_uid = str(getattr(hdr, "StudyInstanceUID", "unknown_study"))
            key = f"{study_uid}_{series_uid}"
            if key not in series_groups:
                series_groups[key] = []
            series_groups[key].append(fpath)
        except Exception:
            continue

    print(f"Found {len(series_groups)} distinct series to process.")

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for key, fpaths in tqdm(series_groups.items(), desc="Converting Series"):
        volume, info = process_dicom_series(fpaths, target_size=target_size)
        if len(volume) == 0:
            continue

        out_fname = f"{key}.npy"
        out_path = os.path.join(output_dir, out_fname)
        np.save(out_path, volume)

        info["Output_File"] = out_fname
        summary_rows.append(info)

    summary_df = pd.DataFrame(summary_rows)
    metadata_csv_path = os.path.join(output_dir, "series_metadata.csv")
    summary_df.to_csv(metadata_csv_path, index=False)
    print(f"Saved processed series metadata to: {metadata_csv_path}")

    return summary_df


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    sample_dir = "data/raw/sample_dicom"
    out_dir = "data/processed/sample_processed"
    if os.path.exists(sample_dir):
        df = convert_dataset_directory(sample_dir, out_dir, target_size=(256, 256))
        print(df)
