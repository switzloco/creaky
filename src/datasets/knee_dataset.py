"""PyTorch Dataset & Collation for Multi-Planar Knee MRI Studies.

Loads 3D volumes across Sagittal, Coronal, and Axial series per study.
Extracts 2.5D slice slabs [z-1, z, z+1] and formats 12-target label vectors.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def sample_2_5d_slabs(volume: np.ndarray, num_slices: int = 16) -> np.ndarray:
    """Sample K 2.5D slabs from a 3D volume (N, H, W).

    Each slab is composed of 3 adjacent slices: [z-1, z, z+1].
    Output shape: (num_slices, 3, H, W) as float32 in [0, 1].
    """
    n_slices, h, w = volume.shape
    if n_slices == 0:
        return np.zeros((num_slices, 3, h, w), dtype=np.float32)

    # Uniform slice index selection across volume depth
    if n_slices <= num_slices:
        indices = np.linspace(0, n_slices - 1, num_slices).round().astype(int)
    else:
        indices = np.linspace(0, n_slices - 1, num_slices).round().astype(int)

    slabs = []
    for idx in indices:
        z_prev = volume[max(0, idx - 1)]
        z_curr = volume[idx]
        z_next = volume[min(n_slices - 1, idx + 1)]
        slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
        # Normalize with ImageNet mean/std
        slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
        slabs.append(slab)

    return np.stack(slabs, axis=0)  # (num_slices, 3, H, W)


class KneeStudyDataset(Dataset):
    """Dataset for full-study knee MRI classification across multiple anatomical planes."""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        series_meta_df: pd.DataFrame,
        volumes_dir: str,
        num_slices_per_plane: int = 16,
        target_cols: Optional[List[str]] = None,
        is_training: bool = True
    ):
        self.labels_df = labels_df.copy()
        self.series_meta = series_meta_df.copy()
        self.volumes_dir = volumes_dir
        self.num_slices = num_slices_per_plane
        self.target_cols = target_cols or DEFAULT_TARGETS
        self.is_training = is_training

        # Group series by StudyInstanceUID
        self.studies = self.labels_df["StudyInstanceUID"].unique().tolist()
        self.series_by_study = self.series_meta.groupby("StudyInstanceUID")

    def __len__(self) -> int:
        return len(self.studies)

    def _load_plane_volume(self, study_uid: str, plane: str) -> np.ndarray:
        """Load or create placeholder volume for a specific anatomical plane."""
        if study_uid in self.series_by_study.groups:
            study_series = self.series_by_study.get_group(study_uid)
            plane_match = study_series[study_series["Anatomical_Plane"].str.lower() == plane.lower()]
            if not plane_match.empty:
                fname = plane_match.iloc[0]["Output_File"]
                fpath = os.path.join(self.volumes_dir, fname)
                if os.path.exists(fpath):
                    return np.load(fpath)

        # Fallback: return dummy volume (1, 256, 256)
        return np.zeros((1, 256, 256), dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        study_uid = self.studies[idx]
        study_row = self.labels_df[self.labels_df["StudyInstanceUID"] == study_uid].iloc[0]

        # Extract 2.5D slabs for each plane
        planes = ["Sagittal", "Coronal", "Axial"]
        plane_tensors = {}

        for plane in planes:
            vol = self._load_plane_volume(study_uid, plane)
            slabs = sample_2_5d_slabs(vol, num_slices=self.num_slices)
            plane_tensors[plane] = torch.tensor(slabs, dtype=torch.float32)

        # Extract target labels
        targets = np.array([float(study_row.get(col, 0.5)) for col in self.target_cols], dtype=np.float32)
        
        # Soft label mask: 1.0 for gold/regex, 0.0 for soft_unk
        source = study_row.get("label_source", "regex")
        weight = 1.0 if source in ["gold", "regex", "llm"] else 0.5
        weights = np.full_like(targets, weight, dtype=np.float32)

        return {
            "study_uid": study_uid,
            "sagittal": plane_tensors["Sagittal"],  # (num_slices, 3, H, W)
            "coronal": plane_tensors["Coronal"],    # (num_slices, 3, H, W)
            "axial": plane_tensors["Axial"],        # (num_slices, 3, H, W)
            "targets": torch.tensor(targets, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32)
        }
