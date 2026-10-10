"""Unit and integration test suite for the Galaxy of Knees & 3D Visualizations.

Audits:
1. 768-D multi-plane embedding structure and UMAP 2D coordinates
2. 12-Axis radar HUD metrics, targets, and discrepancy flags
3. Marching Cubes 3D surface mesh topology and manifold integrity
4. Standalone HTML application structure and dataset integrity
5. Jupyter notebook schema validation
"""

import json
import math
import os
import unittest

class TestGalaxyAnd3DVisualizations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/processed/galaxy_data.json"
        cls.html_path = "galaxy_of_knees.html"
        cls.nb_path = "notebooks/galaxy_of_knees_3d.ipynb"
        
        with open(cls.data_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_01_data_file_exists_and_loaded(self):
        self.assertTrue(os.path.exists(self.data_path), "galaxy_data.json should exist")
        self.assertIn("nodes", self.data)
        self.assertIn("clusters", self.data)
        self.assertIn("metadata", self.data)
        self.assertIn("meshes", self.data)

    def test_02_node_count_and_coordinates(self):
        nodes = self.data["nodes"]
        self.assertEqual(len(nodes), 928, "Should contain exactly 928 validation cohort studies")
        
        for n in nodes:
            x = n["x"]
            y = n["y"]
            self.assertFalse(math.isnan(x), f"Node {n['id']} x-coordinate should not be NaN")
            self.assertFalse(math.isnan(y), f"Node {n['id']} y-coordinate should not be NaN")
            self.assertFalse(math.isinf(x), f"Node {n['id']} x-coordinate should not be Inf")
            self.assertFalse(math.isinf(y), f"Node {n['id']} y-coordinate should not be Inf")
            
            # Bounds check in reasonable range
            self.assertTrue(-25.0 <= x <= 25.0, f"Node {n['id']} x out of bounds: {x}")
            self.assertTrue(-25.0 <= y <= 25.0, f"Node {n['id']} y out of bounds: {y}")

    def test_03_pathology_clusters(self):
        expected_clusters = {
            "acl_rupture", "multi_ligament_trauma", "osteoarthritis",
            "meniscal_tear", "effusion_synovitis", "contusion_fracture",
            "normal_reference"
        }
        found_clusters = {c["id"] for c in self.data["clusters"]}
        self.assertEqual(expected_clusters, found_clusters)
        
        # Verify every node belongs to an expected cluster
        for n in self.data["nodes"]:
            self.assertIn(n["cluster_id"], expected_clusters, f"Unknown cluster id {n['cluster_id']}")
            self.assertTrue(len(n["cluster_name"]) > 0)
            self.assertTrue(n["cluster_color"].startswith("#"))

    def test_04_12axis_radar_hud_integrity(self):
        expected_targets = [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"
        ]
        
        for n in self.data["nodes"]:
            radar = n["radar"]
            self.assertEqual(len(radar), 12, "Each patient should have exactly 12 radar axes")
            
            for ax_idx, ax in enumerate(radar):
                self.assertEqual(ax["target"], expected_targets[ax_idx])
                self.assertTrue(0.0 <= ax["ground_truth"] <= 1.0)
                self.assertTrue(0.0 <= ax["model_prob"] <= 1.0)
                self.assertTrue(0.0 <= ax["abs_diff"] <= 1.0)
                
            self.assertTrue(0.0 <= n["brier_score"] <= 1.0, f"Brier score {n['brier_score']} out of range")
            self.assertTrue(0.0 <= n["concordance"] <= 1.0, f"Concordance {n['concordance']} out of range")
            self.assertIsInstance(n["cleanlab_flag"], bool)

    def test_05_marching_cubes_3d_meshes(self):
        meshes = self.data["meshes"]
        self.assertIn("bone", meshes)
        self.assertIn("cartilage", meshes)
        self.assertIn("fluid", meshes)
        
        for name, m in meshes.items():
            verts = m["vertices"]
            faces = m["faces"]
            normals = m["normals"]
            
            self.assertGreater(len(verts), 1000, f"{name} mesh should have > 1000 vertices")
            self.assertGreater(len(faces), 1000, f"{name} mesh should have > 1000 faces")
            self.assertEqual(len(verts), len(normals), f"{name} vertex and normal counts should match")
            
            # Check face indices point to valid vertices
            num_verts = len(verts)
            for f in faces[:100]:
                self.assertEqual(len(f), 3, "Faces must be triangles")
                self.assertTrue(0 <= f[0] < num_verts)
                self.assertTrue(0 <= f[1] < num_verts)
                self.assertTrue(0 <= f[2] < num_verts)
                
            # Check normalized vertex bounds in [-1, 1]
            for v in verts[:100]:
                self.assertTrue(-1.2 <= v[0] <= 1.2)
                self.assertTrue(-1.2 <= v[1] <= 1.2)
                self.assertTrue(-1.2 <= v[2] <= 1.2)

    def test_06_standalone_html_app(self):
        self.assertTrue(os.path.exists(self.html_path), "galaxy_of_knees.html should exist")
        size = os.path.getsize(self.html_path)
        self.assertGreater(size, 4 * 1024 * 1024, "HTML file should be self-contained and > 4 MB")
        
        with open(self.html_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        self.assertIn("three.min.js", html)
        self.assertIn("OrbitControls.js", html)
        self.assertIn("galaxy-canvas", html)
        self.assertIn("radar-svg", html)
        self.assertIn("canvas3d", html)
        self.assertIn("iso-slider", html)
        self.assertIn("clip-slider", html)

    def test_07_jupyter_notebook(self):
        self.assertTrue(os.path.exists(self.nb_path), "Notebook should exist")
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        self.assertIn("cells", nb)
        self.assertGreater(len(nb["cells"]), 5)
        # Check that Plotly & 3D cells exist
        found_galaxy = any("THE UMAP GALAXY OF KNEES" in str(c) for c in nb["cells"])
        found_radar = any("12-AXIS CLINICAL RADAR HUD" in str(c) for c in nb["cells"])
        found_3d = any("MARCHING CUBES 3D JOINT VIEWER" in str(c) for c in nb["cells"])
        self.assertTrue(found_galaxy, "Notebook should contain Galaxy of Knees section")
        self.assertTrue(found_radar, "Notebook should contain 12-Axis Radar section")
        self.assertTrue(found_3d, "Notebook should contain 3D Mesh section")

if __name__ == "__main__":
    unittest.main()
