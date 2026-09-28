"""
🦴 TypeSafe Jev as a Feature Encoder for Medical NLP (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Demonstration: Using Jev's "System One" decision primitives to extract
calibrated, structured features from multilingual radiology reports.

Instead of parsing free-text LLM outputs with fragile regex, Jev returns
typed probabilities directly — no hallucinations, no JSON parsing failures,
and every answer includes a calibrated confidence score.

v2 enhancements:
  • Clinically rigorous edge-case prompts (e.g. trace vs marked effusion,
    isolated OA marrow edema vs traumatic bone contusion, synovitis criteria).
  • Structured persona prompting: "Act as expert MSK radiologist".
  • 24 structured features: 12 calibrated probabilities + 12 normalized severities.

Author: Nicholas Switzer (@switzloco)
Competition: RSNA Knee Abnormality Detection (2026)
"""

import os
import sys
import json
import requests

# Fix Windows console encoding for emoji output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL   = "jev-latest"
API_KEY     = os.environ.get("JEV_API_KEY", "")

# ── The 12 RSNA Knee Abnormality Targets with Clinical Nuance ──────────────────

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

NOUL_INSTRUCTIONS = {
    "ACL": (
        "Does this knee MRI radiology report describe anterior cruciate ligament "
        "(ACL) pathology? Answer YES for: explicit tear, rupture, complete or "
        "partial discontinuity, abnormal signal within the ligament, injury, or "
        "a ruptured post-surgical ACL graft. Answer NO for: an intact native ACL, "
        "an intact ACL graft, or if explicitly described as normal/preserved."
    ),
    "MCL": (
        "Does this knee MRI radiology report describe medial collateral ligament "
        "(MCL) pathology? Answer YES for: explicit tear, rupture, sprain, "
        "abnormal signal, thickening with surrounding edema, or injury. "
        "Answer NO for: explicitly normal, intact, or preserved MCL."
    ),
    "Medial Meniscus": (
        "Does this knee MRI radiology report describe medial meniscus pathology? "
        "Answer YES for: explicit tear, rupture, truncation, extrusion, displaced "
        "fragment, bucket-handle tear, maceration, abnormal morphology, or "
        "intrameniscal signal reaching the articular surface (grade 3). "
        "Answer NO for: explicitly normal or intact meniscus."
    ),
    "Lateral Meniscus": (
        "Does this knee MRI radiology report describe lateral meniscus pathology? "
        "Answer YES for: explicit tear, rupture, truncation, extrusion, displaced "
        "fragment, bucket-handle tear, maceration, abnormal morphology, or "
        "intrameniscal signal reaching the articular surface (grade 3). "
        "Answer NO for: explicitly normal or intact lateral meniscus."
    ),
    "Medial OA": (
        "Does this knee MRI radiology report describe osteoarthritis of the "
        "MEDIAL tibiofemoral compartment? Answer YES for: medial cartilage loss, "
        "chondral thinning or fissuring, medial joint space narrowing, medial "
        "marginal osteophytes, or subchondral sclerosis/cysts in the medial compartment."
    ),
    "Lateral OA": (
        "Does this knee MRI radiology report describe osteoarthritis of the "
        "LATERAL tibiofemoral compartment? Answer YES for: lateral cartilage "
        "loss, chondral thinning or fissuring, lateral joint space narrowing, "
        "lateral marginal osteophytes, or subchondral sclerosis/cysts in the lateral compartment."
    ),
    "PF OA": (
        "Does this knee MRI radiology report describe patellofemoral "
        "osteoarthritis (PF OA)? Answer YES for: patellofemoral cartilage loss, "
        "patellar or trochlear chondromalacia, patellar or trochlear osteophytes, "
        "or patellofemoral subchondral changes."
    ),
    "Effusion": (
        "Does this knee MRI radiology report describe clinically significant "
        "joint effusion? Answer YES for: moderate, large, marked, or significant "
        "joint effusion, or fluid distending the suprapatellar recess/bursa. "
        "Answer NO for: explicitly absent effusion, OR if fluid is described "
        "ONLY as minimal, trace, small, or physiological. Fluid in a Baker's cyst is NOT effusion."
    ),
    "Synovitis": (
        "Does this knee MRI radiology report describe synovitis or synovial "
        "inflammation? Answer YES for: explicit synovitis, synovial thickening, "
        "synovial hypertrophy/proliferation, Hoffa fat pad edema/impingement. "
        "IMPORTANT: Joint fluid or effusion ALONE without synovial thickening is NOT synovitis."
    ),
    "Baker's": (
        "Does this knee MRI radiology report describe a Baker cyst (popliteal "
        "cyst / gastrocnemius-semimembranosus bursa collection)? Answer YES for: "
        "explicit Baker cyst, popliteal cyst, or fluid collection in the popliteal fossa."
    ),
    "Contusion": (
        "Does this knee MRI radiology report describe a bone contusion or bone "
        "bruise? Answer YES for: explicit bone contusion, bone bruise, or "
        "traumatic bone marrow edema. Do NOT select YES for isolated subchondral "
        "edema attributed purely to chronic osteoarthritis without trauma wording."
    ),
    "Fracture": (
        "Does this knee MRI radiology report describe a fracture of any bone "
        "about the knee? Answer YES for: explicit acute or subacute fracture, "
        "cortical disruption, avulsion fracture, occult fracture, stress/insufficiency fracture."
    ),
}

SEVERITY_SCALE = ["absent", "mild", "moderate", "severe", "critical"]

def build_state_prompt(report_text: str) -> str:
    return (
        "Act as an expert musculoskeletal radiologist.\n"
        "Read the following knee MRI radiology report in its original language.\n"
        "Use ONLY explicitly stated findings from the report text. Do not infer "
        "from clinical history, indications, or outside assumptions.\n\n"
        "KNEE MRI RADIOLOGY REPORT:\n"
        '"""\n'
        f"{report_text.strip()}\n"
        '"""\n\n'
        "Task: Evaluate the status and severity of all requested knee structures "
        "based strictly on the report text above."
    )

def build_jev_payload(report_text: str) -> dict:
    questions = {}
    for target in TARGETS:
        key = target.replace("'", "").replace(" ", "_")
        questions[f"{key}_present"] = {
            "type": "noul",
            "instructions": NOUL_INSTRUCTIONS[target] + " Consider any language.",
        }
        questions[f"{key}_severity"] = {
            "type": "score",
            "instructions": f"Rate severity of {target} pathology in this report. If absent, rate 'absent'.",
            "criteria": SEVERITY_SCALE,
        }

    return {
        "model": JEV_MODEL,
        "state": build_state_prompt(report_text),
        "questions": questions,
    }

def encode_report(report_text: str) -> dict:
    payload = build_jev_payload(report_text)
    resp = requests.post(
        JEV_API_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def print_results(report_text: str, response: dict, label: str = ""):
    answers = response["answers"]
    usage = response.get("usage", {})

    print(f"\n{'━' * 72}")
    if label:
        print(f"  📋 {label}")
    print(f"{'─' * 72}")
    print(f"  Report: {report_text[:120]}{'...' if len(report_text) > 120 else ''}")
    print(f"{'─' * 72}")
    print(f"  {'Target':<22} {'Prob':>6}  {'Severity':>8}  {'Confidence':>10}  Signal")
    print(f"  {'─' * 66}")

    for target in TARGETS:
        key = target.replace("'", "").replace(" ", "_")
        noul = answers.get(f"{key}_present", {})
        score = answers.get(f"{key}_severity", {})

        prob = noul.get("noul", 0.0)
        sev_val = score.get("score", 0.0)
        sev_conf = score.get("confidence", 0.0)
        sev_norm = sev_val / (len(SEVERITY_SCALE) - 1)

        if prob >= 0.8:
            signal = "🔴 DETECTED"
        elif prob >= 0.35:
            signal = "🟡 possible"
        else:
            signal = "🟢 absent"

        print(f"  {target:<22} {prob:>5.2f}   {sev_norm:>7.3f}   {sev_conf:>9.2f}   {signal}")

    in_tok = usage.get("input_tokens", 0)
    cost = (in_tok / 1_000_000) * 0.042
    print(f"  {'─' * 66}")
    print(f"  Tokens: {in_tok:,}  |  Cost: ${cost:.5f}  |  Model: {response.get('model', '?')}")
    print(f"{'━' * 72}")

DEMO_REPORTS = [
    {
        "label": "🇪🇸 Spanish — Meniscal tear + OA + Effusion",
        "text": (
            "Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. "
            "Signo de necrosis avascular subcondral en el cóndilo femoral medial. "
            "Artrosis femorotibial medial. Derrame. Impresión: Rotura de menisco "
            "interno. Artrosis femorotibial medial. Derrame."
        ),
    },
    {
        "label": "🇬🇧 English — Negative Control with Trace Fluid (Not Clinical Effusion)",
        "text": (
            "MRI of the right knee without contrast. Findings: The ACL and PCL "
            "are intact. The medial and lateral menisci are normal. Trace physiological "
            "fluid in the suprapatellar pouch, not meeting criteria for true effusion. "
            "No bone contusion or fracture. Impression: Normal examination."
        ),
    },
    {
        "label": "🇳🇱 Dutch — ACL tear + Traumatic bone bruise",
        "text": (
            "MRI knie links. Bevindingen: Complete ruptuur van de voorste "
            "kruisband. Beenmergoedeem ter hoogte van het laterale tibiaplateau "
            "en laterale femurcondyl, passend bij bone bruise. Intacte menisci. "
            "Conclusie: VKB ruptuur met geassocieerde bone bruise lateraal compartiment."
        ),
    },
]

def main():
    if not API_KEY:
        print("❌ Set your JEV_API_KEY environment variable first!")
        return

    print("\n" + "=" * 72)
    print("  🦴 TypeSafe Jev × RSNA Knee Abnormality Detection (v2)")
    print("  High-Precision Clinical Encoder — 24 Calibrated Features")
    print("=" * 72)

    total_tokens = 0
    total_cost = 0.0

    for demo in DEMO_REPORTS:
        response = encode_report(demo["text"])
        print_results(demo["text"], response, label=demo["label"])

        usage = response.get("usage", {})
        tokens = usage.get("input_tokens", 0)
        total_tokens += tokens
        total_cost += (tokens / 1_000_000) * 0.042

    print(f"\n{'=' * 72}")
    print(f"  📊 Summary")
    print(f"  Reports encoded:  {len(DEMO_REPORTS)}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Total cost:       ${total_cost:.5f}")
    print(f"{'=' * 72}\n")

if __name__ == "__main__":
    main()
