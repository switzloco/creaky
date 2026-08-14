"""DICOM slice reader with intensity normalization (windowing), border cropping, and resizing."""

import os
from typing import Optional, Tuple, Union
import numpy as np
import pydicom
import cv2


def apply_windowing(
    pixels: np.ndarray,
    center: Optional[Union[float, int, list]] = None,
    width: Optional[Union[float, int, list]] = None,
    photometric: str = "MONOCHROME2",
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """Normalize raw 16-bit pixel intensities into standardized [0, 255] uint8.

    Applies standard DICOM VOI LUT linear windowing if center and width exist,
    otherwise uses percentile clipping.
    """
    img = pixels.astype(np.float32)

    # Invert if MONOCHROME1 (where 0 is white and high values are black)
    if photometric == "MONOCHROME1":
        img = img.max() - img

    # Handle multi-value window parameters
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
        # Fallback: robust percentile scaling
        val_min = np.percentile(img, p_low)
        val_max = np.percentile(img, p_high)
        if val_max > val_min:
            img = np.clip(img, val_min, val_max)
            img = (img - val_min) / (val_max - val_min) * 255.0
        else:
            img = np.zeros_like(img)

    return img.astype(np.uint8)


def crop_empty_borders(img: np.ndarray, threshold: int = 10, margin: int = 5) -> np.ndarray:
    """Crop out empty black border pixels around the knee tissue."""
    mask = img > threshold
    if not np.any(mask):
        return img

    y_indices, x_indices = np.where(mask)
    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    # Add small padding margin
    h, w = img.shape[:2]
    ymin = max(0, ymin - margin)
    ymax = min(h, ymax + margin + 1)
    xmin = max(0, xmin - margin)
    xmax = min(w, xmax + margin + 1)

    return img[ymin:ymax, xmin:xmax]


def read_dicom_slice(
    dicom_path: str,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    crop_border: bool = True
) -> Tuple[np.ndarray, dict]:
    """Read a DICOM file, apply windowing, crop black borders, and resize.

    Returns:
    - normalized_img: np.ndarray of shape (H, W) in uint8 [0, 255]
    - metadata: dict of essential DICOM spatial and sequence tags
    """
    dcm = pydicom.dcmread(dicom_path)

    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)

    # Convert slope/intercept if present
    pixels = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixels = pixels * slope + intercept

    # Normalize to uint8 [0, 255]
    img = apply_windowing(pixels, center=wc, width=ww, photometric=photometric)

    # Optional border crop
    if crop_border:
        img = crop_empty_borders(img)

    # Resize
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    metadata = {
        "StudyInstanceUID": str(getattr(dcm, "StudyInstanceUID", "")),
        "SeriesInstanceUID": str(getattr(dcm, "SeriesInstanceUID", "")),
        "SOPInstanceUID": str(getattr(dcm, "SOPInstanceUID", "")),
        "SeriesDescription": str(getattr(dcm, "SeriesDescription", "")),
        "ImagePositionPatient": [float(x) for x in getattr(dcm, "ImagePositionPatient", [0, 0, 0])],
        "ImageOrientationPatient": [float(x) for x in getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])],
        "SliceThickness": float(getattr(dcm, "SliceThickness", 1.0)),
        "PixelSpacing": [float(x) for x in getattr(dcm, "PixelSpacing", [1.0, 1.0])],
        "OriginalShape": list(dcm.pixel_array.shape)
    }

    return img, metadata


if __name__ == "__main__":
    sample_file = "data/raw/sample_dicom/1.2.826.0.1.3680043.8.498.10492923471392639089206565125595901837.dcm"
    if os.path.exists(sample_file):
        img, meta = read_dicom_slice(sample_file, target_size=(256, 256))
        print("Successfully read DICOM slice:")
        print(f"  Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
        print(f"  Metadata: {meta}")
