"""Tests that our multithreaded header-first `load_series_raw_v1` and `sample_plane_like_cache`
produce 100% bit-for-bit identical tensors to the original single-threaded functions.
"""

import glob
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pydicom
from tests.test_preprocessing_parity import write_series, load_functions, SUBMISSION, IMAGENET_MEAN, IMAGENET_STD


class TestFastPartCParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        write_series(self.tmp, n_slices=45)
        self.files = sorted(glob.glob(os.path.join(self.tmp, "*.dcm")))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parity(self):
        fns = load_functions(
            SUBMISSION,
            [
                "apply_dicom_windowing",
                "crop_empty_borders",
                "read_slice",
                "sort_by_instance_number",
                "sample_plane_like_cache",
                "cached_plane_to_slabs",
                "apply_windowing_raw",
                "crop_empty_borders_raw",
                "compute_slice_position",
                "load_series_raw_v1",
            ],
        )

        orig_cache = fns["sample_plane_like_cache"](self.files, num_slices=16, target_size=(384, 384))
        orig_raw = fns["load_series_raw_v1"](self.files, target_size=(256, 256), num_slices=16)

        read_slice = fns["read_slice"]
        compute_slice_position = fns["compute_slice_position"]
        apply_windowing_raw = fns["apply_windowing_raw"]
        crop_empty_borders_raw = fns["crop_empty_borders_raw"]

        # Fast sample_plane_like_cache
        def fast_hdr_inst(f):
            try:
                dcm = pydicom.dcmread(f, stop_before_pixels=True)
                return (int(getattr(dcm, "InstanceNumber", 0)), f)
            except Exception:
                base = os.path.splitext(os.path.basename(f))[0]
                return (int(base) if base.isdigit() else 0, f)

        with ThreadPoolExecutor(max_workers=16) as ex:
            meta = list(ex.map(fast_hdr_inst, self.files))
        meta.sort(key=lambda x: x[0])
        dcm_files = [x[1] for x in meta]
        n = len(dcm_files)
        start_idx = int(n * 0.12) if n >= 10 else 0
        end_idx = max(start_idx + 1, int(n * 0.88)) if n >= 10 else n
        indices = np.linspace(start_idx, end_idx - 1, 16, dtype=int)
        selected = [dcm_files[min(max(0, i), n - 1)] for i in indices]
        with ThreadPoolExecutor(max_workers=16) as ex:
            slices = list(ex.map(lambda p: read_slice(p, (384, 384)), selected))
        slices = [s if s is not None else np.zeros((384, 384), dtype=np.uint8) for s in slices]
        fast_cache = np.stack(slices, axis=0)

        np.testing.assert_array_equal(orig_cache, fast_cache)

        # Fast load_series_raw_v1 (header-first + selective pixel decode)
        def fast_hdr_pos(fpath):
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
                return (compute_slice_position(dcm), fpath)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=16) as ex:
            hdrs = [h for h in ex.map(fast_hdr_pos, self.files) if h is not None]
        hdrs.sort(key=lambda x: x[0])
        n_slices = len(hdrs)
        indices = np.linspace(0, n_slices - 1, 16).round().astype(int)
        needed_idx = sorted({max(0, i - 1) for i in indices} | set(indices) | {min(n_slices - 1, i + 1) for i in indices})

        def decode_one(idx):
            fpath = hdrs[idx][1]
            dcm = pydicom.dcmread(fpath)
            photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
            wc = getattr(dcm, "WindowCenter", None)
            ww = getattr(dcm, "WindowWidth", None)
            pixels = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixels = pixels * slope + intercept
            img = apply_windowing_raw(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders_raw(img)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            return idx, img

        with ThreadPoolExecutor(max_workers=16) as ex:
            decoded = dict(ex.map(decode_one, needed_idx))

        slabs = []
        for idx in indices:
            z_prev = decoded[max(0, idx - 1)]
            z_curr = decoded[idx]
            z_next = decoded[min(n_slices - 1, idx + 1)]
            slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
            slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
            slabs.append(slab)
        fast_raw = np.stack(slabs, axis=0).astype(np.float32)

        np.testing.assert_allclose(orig_raw, fast_raw, rtol=0, atol=0)
        print("100% BIT-FOR-BIT PARITY VERIFIED FOR BOTH cache_v1_384 AND raw_v1!")


if __name__ == "__main__":
    unittest.main()
