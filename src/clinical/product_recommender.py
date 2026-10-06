"""Stryker Sports Medicine Product Alignment & Clinical Decision Engine.

Translates 12 RSNA knee abnormality binary predictions / probabilities into
procedurally matched orthopedic surgical implants, back-table instrumentation,
and clinical case planning based on Stryker's latest Sports Medicine, Joint Preservation,
and Enabling Technology portfolio (including ProCinch®, AIR+®, VersiTomic®, Iconix®,
Mako® SmartRobotics, and Triathlon®).

CLINICAL SCRUTINY NOTE:
- Acute traumatic bone bruises (contusions) are managed conservatively (RICE, protected weight-bearing).
  Subchondral augmentation or osteochondral restoration (BIO4®, ProChondrix CR®) is indicated for chronic
  subchondral defects / osteonecrosis, or compartment resurfacing via Mako® Partial Knee.
- Subchondroplasty® (AccuFill®) is a Zimmer Biomet trademark, NOT Stryker. Stryker's bone graft portfolio
  centers on BIO4® and Vitoss® bioactive scaffolds.
- All surgical indications must be established independently by a board-certified orthopedic surgeon.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


class SurgicalTier(str, Enum):
    """Surgical invasion and procedural complexity tier."""
    CONSERVATIVE_DIAGNOSTIC = "Conservative Management / Diagnostic Monitoring"
    ARTHROSCOPIC_DEBRIDEMENT = "Arthroscopic Resection / Fluid Control"
    SOFT_TISSUE_PRESERVATION = "Soft Tissue Repair & Joint Preservation"
    LIGAMENT_RECONSTRUCTION = "Ligament Reconstruction & Cortical Fixation"
    SUBCHONDRAL_BIOLOGIC = "Subchondral Bone Grafting & Biologics"
    ROBOTIC_ARTHROPLASTY = "Robotic-Arm Assisted Arthroplasty (Mako®)"
    PERIARTICULAR_FIXATION = "Rigid Periarticular Fracture Fixation"


@dataclass
class ProductRecommendation:
    """Individual product indication and procedural rationale."""
    finding_key: str
    finding_name: str
    product_name: str
    secondary_products: List[str]
    category: str
    surgical_tier: SurgicalTier
    procedural_role: str
    mechanism_of_action: str
    clinical_scrutiny_note: str
    back_table_kit: List[str]
    confidence_score: float = 1.0


@dataclass
class CasePlan:
    """Consolidated surgical case plan covering primary and synergy recommendations."""
    patient_findings: Dict[str, bool]
    confidence_scores: Dict[str, float]
    primary_recommendations: List[ProductRecommendation]
    synergy_patterns: List[str]
    consolidated_back_table: List[str]
    highest_surgical_tier: SurgicalTier
    clinical_summary: str


# ==============================================================================
# 12-FINDING VERIFIED STRYKER SPORTS MEDICINE & JOINT CARE PRODUCT CATALOG
# ==============================================================================

STRYKER_PRODUCT_CATALOG: Dict[str, Dict[str, Any]] = {
    "ACL": {
        "finding_name": "Anterior Cruciate Ligament (ACL) Tear",
        "primary_product": "ProCinch® Adjustable Cortical Button",
        "secondary_products": ["VersiTomic® Flexible Reaming System", "Biofiber® / Biosteon® Interference Screws", "Pivot Guardian™ Guide"],
        "category": "Ligament Reconstruction & Suspensory Cortical Fixation",
        "surgical_tier": SurgicalTier.LIGAMENT_RECONSTRUCTION,
        "procedural_role": "Anatomic femoral and tibial tunnel creation with high-strength suspensory cortical graft fixation.",
        "mechanism_of_action": "IntelliBraid™ continuous loop technology delivers maximum ultimate tensile load and minimizes cyclic displacement during graft incorporation. VersiTomic flexible retrograde reamer creates independent anatomic tunnels without knee hyperflexion.",
        "clinical_scrutiny_note": "Clinically robust: Suspensory adjustable cortical fixation is current gold standard for soft-tissue and BTB graft reconstruction, providing reproducible graft tensioning at the femoral aperture.",
        "back_table_kit": [
            "ProCinch ST / RT cortical implants (with tensioning sutures)",
            "VersiTomic flexible reamers (diameters 7.0–11.0mm in 0.5mm increments)",
            "Femoral footprint aimer and tibial drill guide",
            "Graft preparation station and sizing cylinders"
        ]
    },
    "MCL": {
        "finding_name": "Medial Collateral Ligament (MCL) Tear / Avulsion",
        "primary_product": "Iconix® All-Suture Anchor Platform",
        "secondary_products": ["ReelX STT® Knotless Specific-Tissue Tensioning Anchor", "Omega® Suture Anchor", "VersiPass® Suture Passer"],
        "category": "Soft-Tissue-to-Bone Reattachment",
        "surgical_tier": SurgicalTier.SOFT_TISSUE_PRESERVATION,
        "procedural_role": "Anatomic re-fixation of torn superficial/deep MCL fibers to the femoral epicondyle or proximal tibia.",
        "mechanism_of_action": "Small 1.4mm/2.3mm drill holes conserve cortical bone stock while all-suture anchor expands subcortically for high pullout strength. ReelX STT enables knotless, incremental tensioning under valgus stress.",
        "clinical_scrutiny_note": "Grade 1 and 2 isolated MCL sprains heal non-operatively with hinged bracing. Surgical anchor fixation is indicated for Grade 3 complete tears with multi-ligament instability (e.g. ACL + MCL) or bony avulsions.",
        "back_table_kit": [
            "Iconix 1, 2, or 3-suture anchor packs (1.4mm / 2.3mm)",
            "ReelX STT knotless driver with tensioning reel",
            "MCL tissue grasper and curved drill guide",
            "Needled high-tensile suture loops for soft tissue oversewing"
        ]
    },
    "Medial Meniscus": {
        "finding_name": "Medial Meniscus Tear (Inner Cartilage Pad)",
        "primary_product": "AIR+® All-Inside Meniscal Repair System",
        "secondary_products": ["SharpShooter® Inside-Out Suture Passer", "VersiPass® Meniscal Suture Passer", "Meniscal Root Repair Instrument Kit"],
        "category": "Meniscal Repair & Native Joint Preservation",
        "surgical_tier": SurgicalTier.SOFT_TISSUE_PRESERVATION,
        "procedural_role": "Anatomic compression and stabilization of medial meniscal body, posterior horn, or root tears.",
        "mechanism_of_action": "Flexible low-profile needle conforms to tight posterior medial compartment without chondral scuffing; pre-tied sliding PEEK implants provide 360° hoop-stress restoration. VersiTomic transtibial guide enables root re-attachment.",
        "clinical_scrutiny_note": "Meniscal repair is prioritized in vascular zones (red-red and red-white) to prevent early osteoarthritis. For complex degenerative macerations unsuitable for repair, partial meniscectomy using Formula® shavers is indicated.",
        "back_table_kit": [
            "AIR+ straight and curved needle delivery devices",
            "SharpShooter inside-out cannula set with safety spoons",
            "Meniscal depth probe and rasp for edge decortication",
            "Transtibial suture passing guide for meniscal root tears"
        ]
    },
    "Lateral Meniscus": {
        "finding_name": "Lateral Meniscus Tear (Outer Cartilage Pad)",
        "primary_product": "AIR+® All-Inside Meniscal Repair System",
        "secondary_products": ["SharpShooter® Inside-Out System", "VersiPass® Suture Passer", "Zone-Specific Meniscal Cannulas"],
        "category": "Meniscal Preservation & Shock Absorption",
        "surgical_tier": SurgicalTier.SOFT_TISSUE_PRESERVATION,
        "procedural_role": "Preservation of lateral meniscus shock absorption, stabilizing tears adjacent to the popliteus tendon hiatus.",
        "mechanism_of_action": "Flexible needle navigates lateral joint tightness and protects posterior neurovascular structures while compressing radial and horizontal cleavage tears.",
        "clinical_scrutiny_note": "Preserving the lateral meniscus is critical because it carries ~70% of lateral compartment load; total or subtotal removal leads to rapid lateral compartment chondrosis.",
        "back_table_kit": [
            "AIR+ reverse-curved delivery guns",
            "Popliteal safety retractor spoon",
            "Meniscal tissue probe and micro-shaver blades"
        ]
    },
    "Medial OA": {
        "finding_name": "Medial Compartment Osteoarthritis",
        "primary_product": "Mako® SmartRobotics Partial Knee (Triathlon PKR / Restoris MCK)",
        "secondary_products": ["Mako Robotic Power System (Mako RPS)", "Triathlon® Medial Stabilized Insert", "ProChondrix CR® Viable Allograft"],
        "category": "Robotic-Arm Assisted Arthroplasty (Mako®)",
        "surgical_tier": SurgicalTier.ROBOTIC_ARTHROPLASTY,
        "procedural_role": "Targeted resurfacing of isolated medial compartment arthritis while sparing native ACL, PCL, and lateral compartment.",
        "mechanism_of_action": "CT-based 3D preoperative planning and haptic robotic-arm stereotactic cutting boundaries ensure precise bone cuts, ligamentous balancing, and rapid postoperative recovery with Triathlon PKR implants.",
        "clinical_scrutiny_note": "Gold standard for isolated unicompartmental osteoarthritis in active patients with intact cruciate ligaments and absence of inflammatory joint disease.",
        "back_table_kit": [
            "Mako optical tracking arrays and tibial/femoral bone pins",
            "Triathlon PKR trial components and insert impactors",
            "ProChondrix CR biopsy punch and delivery tamp (if focal defect)"
        ]
    },
    "Lateral OA": {
        "finding_name": "Lateral Compartment Osteoarthritis",
        "primary_product": "Mako® SmartRobotics Partial Knee / Triathlon® Total Knee System",
        "secondary_products": ["Mako Robotic Power System (Mako RPS)", "ProChondrix CR® Viable Osteochondral Allograft"],
        "category": "Robotic-Arm Arthroplasty / Joint Reconstruction",
        "surgical_tier": SurgicalTier.ROBOTIC_ARTHROPLASTY,
        "procedural_role": "Robotic unicompartmental lateral resurfacing or total knee replacement tailored to lateral compartment kinematics.",
        "mechanism_of_action": "Addresses asymmetric lateral wear and valgus alignment. Mako dynamic joint balancing accounts for lateral joint laxity patterns during flexion-extension arc.",
        "clinical_scrutiny_note": "Isolated lateral OA is less common than medial OA (~10:1 ratio); when present, precise balancing is essential because lateral compartment naturally translates more than medial.",
        "back_table_kit": [
            "Mako lateral robotic array and Triathlon instrumentation",
            "Trial lateral inserts and femoral shells",
            "ProChondrix CR cryopreserved graft bath (if isolated focal lesion)"
        ]
    },
    "PF OA": {
        "finding_name": "Patellofemoral Osteoarthritis (Kneecap Cartilage Wear)",
        "primary_product": "Mako® SmartRobotics Patellofemoral Arthroplasty (PFA)",
        "secondary_products": ["ProChondrix CR® Viable Allograft", "Restoris® Trochlear Resurfacing Implants"],
        "category": "Robotic-Arm PFA / Biologic Chondral Resurfacing",
        "surgical_tier": SurgicalTier.ROBOTIC_ARTHROPLASTY,
        "procedural_role": "Resurfacing of isolated trochlear groove and patella articular surfaces with personalized kinematic tracking.",
        "mechanism_of_action": "Robotic tactile feedback prevents patellar maltracking and anterior knee pain by establishing smooth, personalized kinematic patellofemoral tracking without altering tibiofemoral compartments.",
        "clinical_scrutiny_note": "Indicated when severe isolated anterior knee pain and grade IV cartilage loss are unresponsive to physical therapy and injections, and tibiofemoral joint spaces remain intact.",
        "back_table_kit": [
            "Mako PFA trochlear cutting guide and patellar reamer",
            "Patellar clamp and peg drill set",
            "Trial trochlear shells (sizes 1–6)"
        ]
    },
    "Effusion": {
        "finding_name": "Joint Effusion (Capsular Fluid Swelling)",
        "primary_product": "CrossFlow® Integrated Arthroscopy Pump System",
        "secondary_products": ["FloControl® Automated Irrigation System", "High-Flow Cannula Set"],
        "category": "Arthroscopic Fluid Management & Joint Distention",
        "surgical_tier": SurgicalTier.ARTHROSCOPIC_DEBRIDEMENT,
        "procedural_role": "Continuous true intra-articular pressure monitoring, automated fluid evacuation, and joint distention during surgery.",
        "mechanism_of_action": "Dual-cassette motor system prevents joint collapse and fluid extravasation, immediately evacuating bloody effusions/debris to maintain a crystal-clear visual field.",
        "clinical_scrutiny_note": "Clinically, an effusion is a symptom of underlying pathology (ligament tear, meniscal injury, synovitis, or fracture). CrossFlow manages intraoperative fluid mechanics while the primary cause is addressed.",
        "back_table_kit": [
            "CrossFlow day cassette and patient tubing set",
            "High-flow arthroscopic inflow/outflow sheath",
            "Pressure-sensing cannula transducer"
        ]
    },
    "Synovitis": {
        "finding_name": "Synovitis (Inflammatory Synovial Hypertrophy)",
        "primary_product": "Formula® Shaver Handpieces & Aggressive Plus Blades",
        "secondary_products": ["Crossfire® 2 Integrated Console", "Serfas Energy® RF Ablation Probe"],
        "category": "Motorized Tissue Resection & Hemostatic Radiofrequency",
        "surgical_tier": SurgicalTier.ARTHROSCOPIC_DEBRIDEMENT,
        "procedural_role": "Rapid, controlled arthroscopic synovectomy in suprapatellar pouch and gutters with simultaneous coagulation.",
        "mechanism_of_action": "Rotating inner shaver blade resects hypertrophic synovium while Serfas RF plasma field provides precise coblation and vessel sealing without thermal necrosis.",
        "clinical_scrutiny_note": "Clinically indicated for recalcitrant chronic synovitis, pigmented villonodular synovitis (PVNS), or secondary synovitis complicating meniscal and ligament tears.",
        "back_table_kit": [
            "Crossfire 2 console with footswitch control",
            "Formula 4.0mm Aggressive Plus and Resector blades",
            "Serfas Energy 90° and 50° articulating RF suction probes"
        ]
    },
    "Baker's": {
        "finding_name": "Baker's Cyst (Popliteal Cyst)",
        "primary_product": "Crossfire® 2 Console & Formula® Resection Blades",
        "secondary_products": ["Serfas Energy® RF Probe", "70° Arthroscopic Scope with Posteromedial Portal"],
        "category": "Arthroscopic Posteromedial Cyst Decompression",
        "surgical_tier": SurgicalTier.ARTHROSCOPIC_DEBRIDEMENT,
        "procedural_role": "Internal decompression and enlargement of communicating valve between joint capsule and popliteal cyst.",
        "mechanism_of_action": "Shaver and RF ablation enlarge the one-way fibrous valve between semimembranosus and medial gastrocnemius, equalizing pressure and permanently decompressing the cyst.",
        "clinical_scrutiny_note": "Baker's cysts in adults are almost universally secondary to intra-articular pathology (especially medial meniscus tears or OA). Treating only the cyst without fixing the meniscal tear leads to high recurrence rates.",
        "back_table_kit": [
            "70° high-definition arthroscope",
            "Formula 3.5mm full-radius resector shaver",
            "Curved radiofrequency probe for posteromedial capsule"
        ]
    },
    "Contusion": {
        "finding_name": "Bone Contusion / Subchondral Marrow Edema",
        "primary_product": "Conservative Management Protocol (Primary) / BIO4® Bone Graft Substitute (Secondary)",
        "secondary_products": ["ProChondrix CR® Viable Osteochondral Allograft", "Vitoss® Bioactive Bone Matrix"],
        "category": "Conservative Rehabilitation / Subchondral Biologics",
        "surgical_tier": SurgicalTier.CONSERVATIVE_DIAGNOSTIC,
        "procedural_role": "Protected weight-bearing and monitoring for acute microfractures; biologic grafting or joint resurfacing only if chronic subchondral collapse occurs.",
        "mechanism_of_action": "Acute bone bruises naturally remodel over 6–12 weeks. In chronic avascular necrosis or subchondral insufficiency, BIO4 viable bone graft provides osteogenic, osteoinductive, and osteoconductive scaffold.",
        "clinical_scrutiny_note": "CRITICAL CLINICAL DISTINCTION: Acute traumatic bone contusions (e.g. pivot-shift bruises from ACL tears) must NOT undergo invasive bone cement injections; standard of care is protected weight-bearing. Subchondral augmentation is reserved for chronic, non-resolving lesions with risk of collapse.",
        "back_table_kit": [
            "Hinged knee brace and crutches (conservative)",
            "BIO4 viable bone matrix syringe (if chronic subchondral debridement/curettage is planned)"
        ]
    },
    "Fracture": {
        "finding_name": "Knee Fracture (Tibial Plateau / Patellar)",
        "primary_product": "AxSOS® 3 Locking Plate System (Proximal Tibia)",
        "secondary_products": ["Asnis® III Cannulated Screw System", "VariAx® Knee Plating Platform"],
        "category": "Rigid Periarticular Fracture Fixation",
        "surgical_tier": SurgicalTier.PERIARTICULAR_FIXATION,
        "procedural_role": "Anatomic reduction and subchondral raft screw support for split/depressed tibial plateau fractures.",
        "mechanism_of_action": "Low-profile anatomically pre-contoured plates support subchondral articular fragments; cannulated lag screws achieve rigid interfragmentary compression under fluoroscopy.",
        "clinical_scrutiny_note": "Depressed tibial plateau fractures > 2–3mm require elevation of the joint surface, subchondral bone grafting, and rigid buttress plate fixation to prevent post-traumatic osteoarthritis.",
        "back_table_kit": [
            "AxSOS 3 lateral/medial proximal tibia locking plates",
            "Asnis III 4.0mm, 5.0mm, and 6.5mm cannulated screws and guide wires",
            "Articular fragment reduction tamps and bone reduction forceps"
        ]
    }
}


# ==============================================================================
# CLINICAL DECISION & RECOMMENDATION ENGINE
# ==============================================================================

def recommend_stryker_products(
    findings: Dict[str, Union[bool, float]],
    threshold: float = 0.5,
    report_text: Optional[str] = None
) -> CasePlan:
    """Evaluates multi-label model predictions and derives tailored Stryker product plans.

    Args:
        findings: Dictionary mapping finding name to boolean or float confidence.
        threshold: Decision threshold for boolean conversion (default: 0.50).
        report_text: Optional ground-truth radiology report text to confirm findings.

    Returns:
        CasePlan containing matched products, synergy patterns, and back-table checklist.
    """
    boolean_findings: Dict[str, bool] = {}
    confidence_scores: Dict[str, float] = {}

    for key, val in findings.items():
        clean_key = key
        for cat_k in STRYKER_PRODUCT_CATALOG.keys():
            if cat_k.lower() == key.lower():
                clean_key = cat_k
                break

        if isinstance(val, bool):
            boolean_findings[clean_key] = val
            confidence_scores[clean_key] = 1.0 if val else 0.0
        elif isinstance(val, (int, float)):
            conf = float(val)
            confidence_scores[clean_key] = conf
            boolean_findings[clean_key] = conf >= threshold
        else:
            boolean_findings[clean_key] = False
            confidence_scores[clean_key] = 0.0

    primary_recs: List[ProductRecommendation] = []
    active_keys: Set[str] = {k for k, v in boolean_findings.items() if v}

    for k in active_keys:
        if k in STRYKER_PRODUCT_CATALOG:
            cat = STRYKER_PRODUCT_CATALOG[k]
            rec = ProductRecommendation(
                finding_key=k,
                finding_name=cat["finding_name"],
                product_name=cat["primary_product"],
                secondary_products=list(cat["secondary_products"]),
                category=cat["category"],
                surgical_tier=cat["surgical_tier"],
                procedural_role=cat["procedural_role"],
                mechanism_of_action=cat["mechanism_of_action"],
                clinical_scrutiny_note=cat.get("clinical_scrutiny_note", ""),
                back_table_kit=list(cat["back_table_kit"]),
                confidence_score=confidence_scores.get(k, 1.0)
            )
            primary_recs.append(rec)

    # Sort recommendations by procedural tier hierarchy
    tier_priority = {
        SurgicalTier.PERIARTICULAR_FIXATION: 7,
        SurgicalTier.ROBOTIC_ARTHROPLASTY: 6,
        SurgicalTier.LIGAMENT_RECONSTRUCTION: 5,
        SurgicalTier.SUBCHONDRAL_BIOLOGIC: 4,
        SurgicalTier.SOFT_TISSUE_PRESERVATION: 3,
        SurgicalTier.ARTHROSCOPIC_DEBRIDEMENT: 2,
        SurgicalTier.CONSERVATIVE_DIAGNOSTIC: 1
    }
    primary_recs.sort(key=lambda r: tier_priority.get(r.surgical_tier, 0), reverse=True)

    # --------------------------------------------------------------------------
    # SYNERGY & MULTI-PATHOLOGY CLINICAL PATTERN DETECTION
    # --------------------------------------------------------------------------
    synergies: List[str] = []

    # 1. Multi-Ligament / Terrible Triad pattern
    has_acl = boolean_findings.get("ACL", False)
    has_mcl = boolean_findings.get("MCL", False)
    has_med_men = boolean_findings.get("Medial Meniscus", False)
    has_lat_men = boolean_findings.get("Lateral Meniscus", False)

    if has_acl and has_mcl:
        synergies.append(
            "⚡ MULTI-LIGAMENT INSTABILITY (ACL + MCL): Recommend concurrent tunnel planning. "
            "Deploy VersiTomic® to avoid tunnel convergence and pair ProCinch® suspensory button "
            "with Iconix® / ReelX STT® anchor fixation for anatomic valgus restraint."
        )

    # 2. ACL + Meniscus Combo (O'Donoghue / Pivot-Shift complex)
    if has_acl and (has_med_men or has_lat_men):
        synergies.append(
            "⚡ ACL RECONSTRUCTION + MENISCAL PRESERVATION COMBO: Plan single-stage joint restoration. "
            "Prepare ProCinch® femoral/tibial fixation alongside AIR+® all-inside meniscal repair. "
            "Preserving the posterior meniscal horn protects the ACL graft from excessive rotational stress."
        )

    # 3. Osteoarthritis + Bone Marrow Contusion (BML) complex
    has_oa = any(boolean_findings.get(k, False) for k in ["Medial OA", "Lateral OA", "PF OA"])
    has_contusion = boolean_findings.get("Contusion", False)

    if has_oa and has_contusion:
        synergies.append(
            "⚡ SUBCHONDRAL INSUFFICIENCY WITH OA: Persistent marrow edema in an osteoarthritic joint "
            "indicates subchondral trabecular failure. Evaluate for BIO4® bone graft augmentation or "
            "proceed with Mako® SmartRobotics Partial Knee Arthroplasty (Triathlon PKR)."
        )

    # 4. Joint Effusion + Synovitis + Baker's Cyst tri-complex
    has_effusion = boolean_findings.get("Effusion", False)
    has_synovitis = boolean_findings.get("Synovitis", False)
    has_bakers = boolean_findings.get("Baker's", False)

    if (has_effusion and has_synovitis) or has_bakers:
        synergies.append(
            "⚡ INFLAMMATORY / CYSTIC TRICOMPLEX: Integrate CrossFlow® continuous fluid management "
            "with Crossfire® 2 console for aggressive Formula® shaver synovectomy and Serfas Energy® "
            "posteromedial capsular release for Baker's cyst internal drainage."
        )

    # 5. Acute Trauma / Fracture complex
    has_fracture = boolean_findings.get("Fracture", False)
    if has_fracture and (has_effusion or has_contusion):
        synergies.append(
            "⚡ PERIARTICULAR FRACTURE WITH HEMARTHROSIS: Prioritize rigid AxSOS® 3 plate / Asnis® III "
            "screw anatomic reduction under fluoroscopy, followed by CrossFlow® joint decompression."
        )

    # Compile consolidated back-table instrument checklist
    consolidated_kit: List[str] = []
    seen_items: Set[str] = set()
    for rec in primary_recs:
        for item in rec.back_table_kit:
            if item not in seen_items:
                seen_items.add(item)
                consolidated_kit.append(item)

    # Determine highest procedural tier
    highest_tier = primary_recs[0].surgical_tier if primary_recs else SurgicalTier.CONSERVATIVE_DIAGNOSTIC

    # Generate clinical case summary
    if not primary_recs:
        summary = "No positive knee abnormalities detected. Conservative rehabilitation, monitoring, and RICE protocol indicated."
    else:
        active_names = [r.finding_name.split("(")[0].strip() for r in primary_recs]
        products = [r.product_name for r in primary_recs]
        summary = (
            f"Active Findings ({len(primary_recs)}): {', '.join(active_names)}. "
            f"Primary Surgical Systems: {', '.join(products[:3])}. "
            f"Highest Care Tier: {highest_tier.value}."
        )

    return CasePlan(
        patient_findings=boolean_findings,
        confidence_scores=confidence_scores,
        primary_recommendations=primary_recs,
        synergy_patterns=synergies,
        consolidated_back_table=consolidated_kit,
        highest_surgical_tier=highest_tier,
        clinical_summary=summary
    )
