"""
Automated verification suite for Creaky-CV5 (OpenCV AI Competition 2026 · Powered by AWS).
Tests HIPAA Safe-Harbor pixel PHI scrubbing, FOV contour auto-cropping, multi-modal CLAHE +
Scharr cortical edge enhancement, Agentic Focal Zoom & Articular Surface Contact biomarkers,
and the 144-slice OpenCV 5 + AWS COOL hardware benchmark.
"""

from __future__ import annotations

import unittest
import numpy as np

from opencv_hackathon.cv5_engine import (
    agentic_focal_zoom_and_contours,
    crop_fov_with_contours,
    enhance_slice_multimodal,
    run_opencv_cool_benchmark,
    scrub_burned_in_phi_opencv5,
)


class TestOpenCVHackathonEngine(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.synthetic_mri = rng.integers(15, 210, size=(384, 384, 3), dtype=np.uint8)
        # Simulate burned-in high-intensity corner PHI glyph
        self.synthetic_mri[4:12, 10:60] = 250

    def test_hipaa_safe_harbor_phi_scrubber(self) -> None:
        scrubbed, audit = scrub_burned_in_phi_opencv5(self.synthetic_mri)
        self.assertEqual(scrubbed.shape, self.synthetic_mri.shape)
        self.assertTrue(audit["hipaa_safe_harbor_45cfr164_514"])
        self.assertTrue(audit["zero_egress_aws_cool_vpc"])
        self.assertTrue(audit["safe_harbor_sha256"].startswith("SH256-"))
        self.assertGreaterEqual(audit["peripheral_phi_pixels_scrubbed"], 0)

    def test_fov_contour_autocrop(self) -> None:
        gray = np.zeros((384, 384), dtype=np.uint8)
        gray[60:320, 70:310] = 140
        cropped, bbox = crop_fov_with_contours(gray)
        self.assertLess(cropped.shape[0], 384)
        self.assertLess(cropped.shape[1], 384)
        self.assertEqual(len(bbox), 4)

    def test_multimodal_clahe_and_scharr_edge(self) -> None:
        res = enhance_slice_multimodal(self.synthetic_mri)
        for key in ("gray", "clahe_gray", "clahe_rgb", "scharr_mag", "scharr_overlay"):
            self.assertIn(key, res)
        self.assertEqual(res["clahe_rgb"].shape, (384, 384, 3))
        self.assertEqual(res["scharr_overlay"].shape, (384, 384, 3))

    def test_agentic_focal_zoom_and_biomarkers(self) -> None:
        out = agentic_focal_zoom_and_contours(
            self.synthetic_mri,
            None,
            zoom_factor=1.65,
            synthetic_center=(0.52, 0.48),
        )
        for k in ("raw_b64", "clahe_b64", "scharr_b64", "contour_b64", "focal_zoom_b64", "clean_zoom_b64"):
            self.assertTrue(len(out[k]) > 100)
        bio = out["biomarkers"]
        self.assertEqual(bio["zoom_factor"], 1.65)
        self.assertGreater(bio["roi_area_px"], 0)
        self.assertGreaterEqual(bio["articular_surface_contact_pct"], 12.0)
        self.assertLessEqual(bio["articular_surface_contact_pct"], 99.4)

    def test_144_slice_opencv_cool_benchmark(self) -> None:
        bench = run_opencv_cool_benchmark(num_slices=36)
        self.assertEqual(bench["num_slices_processed"], 36)
        self.assertGreater(bench["total_study_ms"], 0.0)
        self.assertGreater(bench["slices_per_sec"], 10.0)
        self.assertIn("cv2_dnn_onnx_slabs_ms", bench["stages_ms"])


if __name__ == "__main__":
    unittest.main()
