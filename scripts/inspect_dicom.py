"""Inspect DICOM header tags, spatial coordinates, and pixel values of real sample DICOM."""
import os
import sys
import pydicom
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

sample_dir = "data/raw/sample_dicom"
files = [f for f in os.listdir(sample_dir) if f.endswith(".dcm")]

print(f"Found {len(files)} sample DICOM files.\n")

for f in files:
    path = os.path.join(sample_dir, f)
    dcm = pydicom.dcmread(path)
    
    print("=" * 70)
    print(f"File: {f}")
    print("=" * 70)
    print(f"StudyInstanceUID:          {getattr(dcm, 'StudyInstanceUID', 'N/A')}")
    print(f"SeriesInstanceUID:         {getattr(dcm, 'SeriesInstanceUID', 'N/A')}")
    print(f"SOPInstanceUID:            {getattr(dcm, 'SOPInstanceUID', 'N/A')}")
    print(f"Modality:                  {getattr(dcm, 'Modality', 'N/A')}")
    print(f"SeriesDescription:         {getattr(dcm, 'SeriesDescription', 'N/A')}")
    print(f"Rows x Columns:            {getattr(dcm, 'Rows', 'N/A')} x {getattr(dcm, 'Columns', 'N/A')}")
    print(f"BitsAllocated / Stored:    {getattr(dcm, 'BitsAllocated', 'N/A')} / {getattr(dcm, 'BitsStored', 'N/A')}")
    print(f"PixelRepresentation:       {getattr(dcm, 'PixelRepresentation', 'N/A')}")
    print(f"PhotometricInterpretation: {getattr(dcm, 'PhotometricInterpretation', 'N/A')}")
    print(f"WindowCenter:              {getattr(dcm, 'WindowCenter', 'N/A')}")
    print(f"WindowWidth:               {getattr(dcm, 'WindowWidth', 'N/A')}")
    print(f"RescaleIntercept / Slope:  {getattr(dcm, 'RescaleIntercept', 'N/A')} / {getattr(dcm, 'RescaleSlope', 'N/A')}")
    print(f"ImagePositionPatient:      {getattr(dcm, 'ImagePositionPatient', 'N/A')}")
    print(f"ImageOrientationPatient:   {getattr(dcm, 'ImageOrientationPatient', 'N/A')}")
    print(f"SliceThickness:            {getattr(dcm, 'SliceThickness', 'N/A')}")
    print(f"PixelSpacing:              {getattr(dcm, 'PixelSpacing', 'N/A')}")
    
    # Pixel array stats
    pixels = dcm.pixel_array
    print(f"\nPixel Array dtype:         {pixels.dtype}")
    print(f"Pixel Array shape:         {pixels.shape}")
    print(f"Min / Max / Mean:          {pixels.min()} / {pixels.max()} / {pixels.mean():.2f}")
