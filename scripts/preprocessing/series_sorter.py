"""Calculates 3D physical coordinates and sorts DICOM series slices in anatomical order."""

from typing import List, Tuple, Dict, Any
import numpy as np


def compute_slice_normal(orientation: List[float]) -> np.ndarray:
    """Compute the unit normal vector perpendicular to the slice plane.

    ImageOrientationPatient contains direction cosines of first row (F) and first column (C).
    Normal vector N = F x C (cross product).
    """
    if len(orientation) != 6:
        return np.array([0.0, 0.0, 1.0])

    f = np.array(orientation[:3], dtype=float)
    c = np.array(orientation[3:], dtype=float)
    normal = np.cross(f, c)
    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0])
    return normal


def compute_slice_position(position: List[float], orientation: List[float]) -> float:
    """Compute the 1D physical projection coordinate along the normal vector."""
    normal = compute_slice_normal(orientation)
    pos = np.array(position[:3], dtype=float)
    return float(np.dot(normal, pos))


def determine_plane_from_orientation(orientation: List[float]) -> str:
    """Determine the anatomical acquisition plane (Sagittal, Coronal, Axial) from direction cosines.

    - Sagittal: slicing perpendicular to X axis (|Nx| is largest)
    - Coronal:  slicing perpendicular to Y axis (|Ny| is largest)
    - Axial:    slicing perpendicular to Z axis (|Nz| is largest)
    """
    normal = compute_slice_normal(orientation)
    abs_normal = np.abs(normal)
    max_idx = int(np.argmax(abs_normal))

    if max_idx == 0:
        return "Sagittal"
    elif max_idx == 1:
        return "Coronal"
    else:
        return "Axial"


def sort_series_slices(slice_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort a list of slice metadata dictionaries along their physical 3D coordinate."""
    for item in slice_metadata_list:
        pos = item.get("ImagePositionPatient", [0, 0, 0])
        ori = item.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        item["_projected_coord"] = compute_slice_position(pos, ori)
        if "Anatomical_Plane" not in item or not item["Anatomical_Plane"]:
            item["Anatomical_Plane"] = determine_plane_from_orientation(ori)

    # Sort ascending by projected coordinate
    sorted_slices = sorted(slice_metadata_list, key=lambda x: x["_projected_coord"])
    return sorted_slices


if __name__ == "__main__":
    # Test on known orientation vectors
    sag_ori = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]  # Normal = [-1, 0, 0]
    cor_ori = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # Normal = [0, 1, 0]
    ax_ori = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]    # Normal = [0, 0, 1]

    print("Plane classification test:")
    print("Sagittal orientation ->", determine_plane_from_orientation(sag_ori))
    print("Coronal orientation  ->", determine_plane_from_orientation(cor_ori))
    print("Axial orientation    ->", determine_plane_from_orientation(ax_ori))
