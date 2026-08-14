"""Unit tests for DICOM preprocessing, windowing, and series sorting."""
import os
import sys
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.abspath(os.path.join(current_dir, "..", "scripts", "preprocessing"))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from dicom_reader import apply_windowing, crop_empty_borders
from series_sorter import compute_slice_normal, compute_slice_position, determine_plane_from_orientation, sort_series_slices


class TestDICOMPreprocessing(unittest.TestCase):

    def test_windowing_linear(self):
        """Test linear windowing formula maps center/width properly to [0, 255]."""
        # Pixel values around center=1000, width=500
        raw_pixels = np.array([[750, 1000, 1250]], dtype=np.uint16)
        out = apply_windowing(raw_pixels, center=1000, width=500)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[0, 0]), 0)     # Minimum bound
        self.assertAlmostEqual(int(out[0, 1]), 128, delta=2)  # Center mapped to mid-gray
        self.assertEqual(int(out[0, 2]), 255)   # Maximum bound


    def test_crop_empty_borders(self):
        """Test bounding box crop removes pure black borders."""
        # 100x100 canvas with 20x20 bright square in center
        canvas = np.zeros((100, 100), dtype=np.uint8)
        canvas[40:60, 40:60] = 200
        cropped = crop_empty_borders(canvas, threshold=10, margin=2)
        # Cropped should be much smaller than 100x100
        self.assertLess(cropped.shape[0], 50)
        self.assertLess(cropped.shape[1], 50)
        self.assertGreaterEqual(cropped.max(), 200)

    def test_plane_determination(self):
        """Test anatomical plane identification from orientation vectors."""
        sag_ori = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]  # Normal = [-1, 0, 0]
        cor_ori = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # Normal = [0, 1, 0]
        ax_ori = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]    # Normal = [0, 0, 1]

        self.assertEqual(determine_plane_from_orientation(sag_ori), "Sagittal")
        self.assertEqual(determine_plane_from_orientation(cor_ori), "Coronal")
        self.assertEqual(determine_plane_from_orientation(ax_ori), "Axial")

    def test_slice_sorting_order(self):
        """Test sorting slices by physical 3D coordinate along the normal vector."""
        ori = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Axial normal [0, 0, 1]
        slices = [
            {"ImagePositionPatient": [0, 0, 30.0], "ImageOrientationPatient": ori, "name": "slice3"},
            {"ImagePositionPatient": [0, 0, 10.0], "ImageOrientationPatient": ori, "name": "slice1"},
            {"ImagePositionPatient": [0, 0, 20.0], "ImageOrientationPatient": ori, "name": "slice2"},
        ]
        sorted_res = sort_series_slices(slices)
        ordered_names = [s["name"] for s in sorted_res]
        self.assertEqual(ordered_names, ["slice1", "slice2", "slice3"])


if __name__ == "__main__":
    unittest.main()
