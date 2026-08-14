"""Unit tests for the RegexLabeler module."""
import os
import sys
import unittest

# Ensure scripts/label_mining is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.abspath(os.path.join(current_dir, "..", "scripts", "label_mining"))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from regex_labeler import RegexLabeler, TARGET_COLS


class TestRegexLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = RegexLabeler()

    def test_target_columns_match(self):
        """Ensure the target column list has exactly 12 findings."""
        self.assertEqual(len(TARGET_COLS), 12)
        self.assertIn("ACL", TARGET_COLS)
        self.assertIn("Baker's", TARGET_COLS)

    def test_english_acl_positive(self):
        report = "Impression: Complete full-thickness tear of the anterior cruciate ligament."
        result = self.labeler.extract_from_report(report)
        self.assertEqual(result["ACL"], 1.0)

    def test_english_acl_negative(self):
        report = "Findings: The ACL and MCL are intact and unremarkable. No evidence of effusion."
        result = self.labeler.extract_from_report(report)
        self.assertEqual(result["ACL"], 0.0)
        self.assertEqual(result["MCL"], 0.0)
        self.assertEqual(result["Effusion"], 0.0)

    def test_spanish_meniscus_positive(self):
        report = "Técnica: RMN de rodilla. Rotura de menisco interno y menisco externo. Derrame articular."
        result = self.labeler.extract_from_report(report)
        self.assertEqual(result["Medial Meniscus"], 1.0)
        self.assertEqual(result["Lateral Meniscus"], 1.0)
        self.assertEqual(result["Effusion"], 1.0)

    def test_spanish_meniscus_negative(self):
        report = "Hallazgos: Menisco medial y lateral de morfología conservada sin signos de rotura."
        result = self.labeler.extract_from_report(report)
        self.assertEqual(result["Medial Meniscus"], 0.0)
        self.assertEqual(result["Lateral Meniscus"], 0.0)

    def test_french_fracture(self):
        report = "Constatations: Fracture du plateau tibial medial. Épanchement abondant."
        result = self.labeler.extract_from_report(report)
        self.assertEqual(result["Fracture"], 1.0)
        self.assertEqual(result["Effusion"], 1.0)

    def test_hedged_uncertainty_maps_to_none(self):
        report = "Impression: Equivocal signal in the posterior horn, cannot exclude subtle tear."
        result = self.labeler.extract_from_report(report)
        # Should not falsely trigger 1.0
        self.assertNotEqual(result["Medial Meniscus"], 1.0)


if __name__ == "__main__":
    unittest.main()
