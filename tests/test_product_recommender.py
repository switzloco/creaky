import pytest
from src.clinical.product_recommender import (
    STRYKER_PRODUCT_CATALOG,
    SurgicalTier,
    recommend_stryker_products,
)


def test_catalog_has_all_12_rsna_findings():
    expected_findings = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    for finding in expected_findings:
        assert finding in STRYKER_PRODUCT_CATALOG, f"Missing finding: {finding}"
        item = STRYKER_PRODUCT_CATALOG[finding]
        assert "primary_product" in item
        assert "category" in item
        assert "surgical_tier" in item
        assert "procedural_role" in item
        assert "back_table_kit" in item
        assert len(item["back_table_kit"]) > 0


def test_single_acl_tear_recommendation():
    findings = {"ACL": 0.95, "MCL": 0.05, "Medial Meniscus": 0.10}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert plan.patient_findings["ACL"] is True
    assert plan.patient_findings["MCL"] is False
    assert len(plan.primary_recommendations) == 1

    rec = plan.primary_recommendations[0]
    assert rec.finding_key == "ACL"
    assert "ProCinch" in rec.product_name
    assert rec.surgical_tier == SurgicalTier.LIGAMENT_RECONSTRUCTION
    assert any("VersiTomic" in s for s in rec.secondary_products)
    assert len(plan.consolidated_back_table) > 0


def test_acl_meniscus_combo_synergy():
    findings = {"ACL": 0.98, "Medial Meniscus": 0.92, "Effusion": 0.85}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert len(plan.primary_recommendations) == 3
    # Check that synergy rule fired
    assert any("ACL RECONSTRUCTION + MENISCAL PRESERVATION COMBO" in syn for syn in plan.synergy_patterns)
    # Check that both ProCinch and AIR+ are present
    product_names = [r.product_name for r in plan.primary_recommendations]
    assert any("ProCinch" in p for p in product_names)
    assert any("AIR+" in p for p in product_names)


def test_multi_ligament_instability():
    findings = {"ACL": 0.90, "MCL": 0.88}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert any("MULTI-LIGAMENT INSTABILITY" in syn for syn in plan.synergy_patterns)
    product_names = [r.product_name for r in plan.primary_recommendations]
    assert any("ProCinch" in p for p in product_names)
    assert any("Iconix" in p for p in product_names)


def test_osteoarthritis_with_bone_contusion():
    findings = {"PF OA": 0.95, "Contusion": 0.80}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert any("SUBCHONDRAL INSUFFICIENCY WITH OA" in syn for syn in plan.synergy_patterns)
    product_names = [r.product_name for r in plan.primary_recommendations]
    assert any("Mako" in p or "ProChondrix" in p for p in product_names)
    assert any("BIO4" in p or "Conservative" in p for p in product_names)


def test_clean_healthy_knee():
    findings = {k: 0.02 for k in STRYKER_PRODUCT_CATALOG.keys()}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert len(plan.primary_recommendations) == 0
    assert len(plan.synergy_patterns) == 0
    assert plan.highest_surgical_tier == SurgicalTier.CONSERVATIVE_DIAGNOSTIC
    assert "Conservative" in plan.clinical_summary


def test_disclaimer_present_and_explicit():
    findings = {"ACL": 0.95}
    plan = recommend_stryker_products(findings, threshold=0.5)

    assert hasattr(plan, "disclaimer")
    assert "NOT an official Stryker recommendation" in plan.disclaimer
    assert "data" in plan.disclaimer.lower() or "literature" in plan.disclaimer.lower()

