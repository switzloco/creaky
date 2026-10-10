import os
import sys
sys.path.insert(0, ".")
import pydicom
import numpy as np
import cv2

# Import functions from scripts/preprocessing/cache_kaggle_kernel.py
from scripts.preprocessing.cache_kaggle_kernel import apply_dicom_windowing, crop_empty_borders

def test_slicing_indices():
    # Test slice banding logic for various series lengths
    for n in [15, 20, 24, 28, 32, 36]:
        start_idx = int(n * 0.02) if n >= 10 else 0
        end_idx = max(start_idx + 1, int(n * 0.98)) if n >= 10 else n
        indices = np.linspace(start_idx, end_idx - 1, 16, dtype=int)
        print(f"Total slices: {n} -> Slices kept: [{start_idx} to {end_idx-1}] -> Selected indices: {list(indices)}")
        assert len(indices) == 16
        assert indices[0] >= 0
        assert indices[-1] < n
        # Verify that index 0 or 1 is included (preserving lateral periphery)
        assert indices[0] <= 1, f"Peripheral slice clipped! Got start index {indices[0]}"
    print("All slice banding tests passed successfully!")

if __name__ == "__main__":
    test_slicing_indices()
