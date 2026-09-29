"""Train/inference preprocessing parity.

There are two training pipelines, and the submission notebook re-implements both on raw
test DICOMs:
  - cache_v1: slices from the Kaggle caching kernel (scripts/preprocessing/cache_kaggle_kernel.py),
    turned into 2.5D slabs by cached_plane_to_slabs() in the training script.
  - raw_v1: the training script's load_and_preprocess_series() on raw DICOMs.
If the submission drifts from either, the model is scored on inputs it never saw in
training, and nothing errors.

These tests build synthetic DICOM series and check that every path produces identical
arrays. The functions are pulled out of each script with `ast`, so torch is not needed.

Run: python -m unittest tests/test_preprocessing_parity.py
"""
import ast
import os
import random
import tempfile
import types
import unittest
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_KERNEL = os.path.join(ROOT, "scripts", "preprocessing", "cache_kaggle_kernel.py")
CACHE_LOCAL = os.path.join(ROOT, "scripts", "preprocessing", "cache_fast_slices.py")
TRAINING = os.path.join(ROOT, "scripts", "training", "train_kaggle_notebook.py")
SUBMISSION = os.path.join(ROOT, "scripts", "submission", "submission_notebook.py")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# Just enough of torch for functions that wrap their numpy result in a tensor.
FAKE_TORCH = types.SimpleNamespace(
    tensor=lambda a, dtype=None: np.asarray(a, dtype=np.float32),
    zeros=lambda shape, dtype=None: np.zeros(shape, dtype=np.float32),
    float32=np.float32,
    Tensor=np.ndarray,
)


def load_functions(path: str, names: List[str]) -> Dict[str, object]:
    """Exec only the named top-level functions of a script, without running its imports."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise AssertionError(f"{os.path.basename(path)} is missing functions: {sorted(missing)}")
    ns = {
        "np": np, "cv2": cv2, "pydicom": pydicom, "os": os, "pd": pd, "torch": FAKE_TORCH,
        "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Union": Union,
        "PIPELINES": ("cache_v1", "raw_v1"),
        "IMAGENET_MEAN": IMAGENET_MEAN, "IMAGENET_STD": IMAGENET_STD,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


def write_series(folder: str, n_slices: int, size: int = 320, with_window: bool = True, seed: int = 0,
                 rescale_slope: float = 1.0) -> None:
    """Write a synthetic sagittal series.

    File names do not follow InstanceNumber order, and physical position runs *opposite* to
    InstanceNumber, so a path that sorts the wrong way produces different arrays.
    """
    rng = np.random.default_rng(seed)
    order = list(range(1, n_slices + 1))
    random.Random(seed).shuffle(order)
    for file_idx, inst in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = os.path.join(folder, f"{generate_uid()}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = inst
        ds.Rows = ds.Columns = size
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]    # sagittal, normal = (-1, 0, 0)
        ds.ImagePositionPatient = [inst * 3.0, -100.0, 100.0]  # position along normal = -3 * inst
        if rescale_slope != 1.0:
            ds.RescaleSlope = rescale_slope
            ds.RescaleIntercept = 0
        if with_window:
            ds.WindowCenter = 600
            ds.WindowWidth = 1000
        # Dark border + a bright blob whose position depends on InstanceNumber,
        # so wrong ordering or sampling produces different arrays.
        img = np.zeros((size, size), dtype=np.uint16)
        c = 40 + inst * 4
        img[30:size - 30, 30:size - 30] = rng.integers(100, 900, (size - 60, size - 60), dtype=np.uint16)
        img[c:c + 40, c:c + 40] = 1200
        ds.PixelData = img.tobytes()
        ds.save_as(path, enforce_file_format=True)


class TestPreprocessingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        common = ["apply_dicom_windowing", "crop_empty_borders"]
        cls.kernel = load_functions(CACHE_KERNEL, common + ["read_slice", "get_sorted_files", "process_plane"])
        cls.local = load_functions(CACHE_LOCAL, common + ["read_and_preprocess_slice", "get_sorted_dicom_files", "process_plane_series"])
        cls.train = load_functions(TRAINING, ["cached_plane_to_slabs", "apply_windowing", "crop_empty_borders",
                                              "compute_slice_position", "load_and_preprocess_series"])
        cls.sub = load_functions(SUBMISSION, common + [
            "read_slice", "sort_by_instance_number", "sample_plane_like_cache", "cached_plane_to_slabs",
            "apply_windowing_raw", "crop_empty_borders_raw", "compute_slice_position", "load_series_raw_v1",
            "select_series_uid", "load_plane"])

    # ---- cache_v1 ---------------------------------------------------------------------------

    def _check_cache_pipeline(self, n_slices: int, **series_kwargs):
        with tempfile.TemporaryDirectory() as d:
            write_series(d, n_slices, seed=n_slices, **series_kwargs)
            files = [os.path.join(d, f) for f in os.listdir(d)]

            cached = self.kernel["process_plane"](d, 16, (256, 256))
            local = self.local["process_plane_series"](d, 16, (256, 256))
            inference = self.sub["sample_plane_like_cache"](files, 16, (256, 256))

            self.assertEqual(cached.shape, (16, 256, 256))
            self.assertEqual(cached.dtype, np.uint8)
            self.assertGreater(cached.std(), 0, "synthetic slices should not be blank")
            np.testing.assert_array_equal(cached, inference, err_msg="submission != Kaggle caching kernel")
            np.testing.assert_array_equal(cached, local, err_msg="local cache script != Kaggle caching kernel")

            train_slabs = self.train["cached_plane_to_slabs"](cached)
            self.assertEqual(train_slabs.shape, (16, 3, 256, 256))
            np.testing.assert_array_equal(train_slabs, self.sub["load_plane"](files, "cache_v1"),
                                          err_msg="cache_v1: submission slabs != training slabs")

    def test_cache_long_series_uses_central_band(self):
        self._check_cache_pipeline(30)

    def test_cache_short_series_uses_all_slices(self):
        self._check_cache_pipeline(8)  # < 10 slices: no banding

    def test_cache_percentile_windowing_without_window_tags(self):
        self._check_cache_pipeline(20, with_window=False)

    def test_cache_ignores_rescale(self):
        self._check_cache_pipeline(20, rescale_slope=2.0)

    def test_cache_sorted_by_instance_number_not_filename(self):
        with tempfile.TemporaryDirectory() as d:
            write_series(d, 12)
            files = sorted(os.path.join(d, f) for f in os.listdir(d))
            ordered = self.sub["sort_by_instance_number"](files)
            numbers = [int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber) for f in ordered]
            self.assertEqual(numbers, sorted(numbers))
            self.assertNotEqual(files, ordered, "test setup: file names should not already be in order")

    def test_cache_empty_series_gives_zeros(self):
        out = self.sub["sample_plane_like_cache"]([], 16, (256, 256))
        self.assertEqual(out.shape, (16, 256, 256))
        self.assertFalse(out.any())

    # ---- raw_v1 -----------------------------------------------------------------------------

    def _check_raw_pipeline(self, n_slices: int, **series_kwargs):
        with tempfile.TemporaryDirectory() as d:
            write_series(d, n_slices, seed=n_slices, **series_kwargs)
            files = [os.path.join(d, f) for f in os.listdir(d)]
            trained_on = self.train["load_and_preprocess_series"](files, (256, 256), 16)
            inference = self.sub["load_plane"](files, "raw_v1")
            self.assertEqual(trained_on.shape, (16, 3, 256, 256))
            np.testing.assert_array_equal(trained_on, inference, err_msg="raw_v1: submission != training")

    def test_raw_matches_training(self):
        self._check_raw_pipeline(30)

    def test_raw_applies_rescale_like_training(self):
        self._check_raw_pipeline(20, rescale_slope=2.0)

    def test_raw_percentile_windowing_without_window_tags(self):
        self._check_raw_pipeline(20, with_window=False)

    def test_pipelines_really_differ(self):
        """Guards the test itself: if both pipelines gave the same output, the checks above prove nothing."""
        with tempfile.TemporaryDirectory() as d:
            write_series(d, 30)
            files = [os.path.join(d, f) for f in os.listdir(d)]
            self.assertFalse(np.array_equal(self.sub["load_plane"](files, "cache_v1"),
                                            self.sub["load_plane"](files, "raw_v1")))

    # ---- series selection -------------------------------------------------------------------

    def test_series_selection_rules(self):
        study = pd.DataFrame({
            "StudyInstanceUID": ["s"] * 4,
            "SeriesInstanceUID": ["sag_a", "cor", "sag_b", "ax"],
            "Anatomical_Plane": ["Sagittal", "Coronal", "sagittal", "Axial"],
        })
        select = self.sub["select_series_uid"]
        # Caching kernel: capitalized exact match, last row wins (dict overwrite).
        kernel_choice = {}
        for _, row in study.iterrows():
            kernel_choice[str(row["Anatomical_Plane"]).capitalize()] = row["SeriesInstanceUID"]
        for plane in ["Sagittal", "Coronal", "Axial"]:
            self.assertEqual(select(study, "Anatomical_Plane", plane, "cache_v1"), kernel_choice[plane])
        # Training raw path: first row whose plane contains the name (case-insensitive).
        self.assertEqual(select(study, "Anatomical_Plane", "Sagittal", "raw_v1"), "sag_a")
        self.assertIsNone(select(pd.DataFrame(), None, "Sagittal", "cache_v1"))


if __name__ == "__main__":
    unittest.main()
