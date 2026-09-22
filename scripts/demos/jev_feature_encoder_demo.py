"""
🦴 TypeSafe Jev as a Feature Encoder for Medical NLP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Demonstration: Using Jev's "System One" decision primitives to extract
calibrated, structured features from multilingual radiology reports.

Instead of parsing free-text LLM outputs with fragile regex, Jev returns
typed probabilities directly — no hallucinations, no JSON parsing failures,
and every answer includes a calibrated confidence score.

This demo shows how to encode knee MRI radiology reports (in any language)
into 24 float features per study — ready to feed into your ML pipeline
as soft labels or auxiliary features for ensembling.

Author: Nicholas Switzer (@switzloco)
Competition: RSNA Knee Abnormality Detection (2026)
"""

# ── Setup ──────────────────────────────────────────────────────────────────────
# pip install requests python-dotenv
# Set your JEV_API_KEY environment variable or put it in a .env file
# Sign up at https://console.typesafe.ai for a free API key

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

# ── The 12 RSNA Knee Abnormality Targets ───────────────────────────────────────

TARGETS = {
    "ACL":              "an anterior cruciate ligament (ACL) tear or injury",
    "MCL":              "a medial collateral ligament (MCL) tear or injury",
    "Medial Meniscus":  "a medial meniscus tear or degeneration",
    "Lateral Meniscus": "a lateral meniscus tear or degeneration",
    "Medial OA":        "medial compartment osteoarthritis",
    "Lateral OA":       "lateral compartment osteoarthritis",
    "PF OA":            "patellofemoral osteoarthritis",
    "Effusion":         "joint effusion (excess fluid in the knee joint)",
    "Synovitis":        "synovitis (inflammation of the synovial membrane)",
    "Baker's":          "a Baker's cyst (popliteal cyst)",
    "Contusion":        "a bone contusion or bone bruise",
    "Fracture":         "a bone fracture",
}

SEVERITY_SCALE = ["absent", "mild", "moderate", "severe", "critical"]


# ── Build the Jev Request ──────────────────────────────────────────────────────

def build_jev_payload(report_text: str) -> dict:
    """
    Construct a single Jev API request with 24 questions:
      - 12 Noul (boolean probability): "Is [condition] present?"
      - 12 Score (ordinal severity):   "How severe is [condition]?"

    All 24 questions are evaluated in PARALLEL in a single API call.
    """
    questions = {}

    for target, description in TARGETS.items():
        key = target.replace("'", "").replace(" ", "_")

        # Noul → returns a calibrated probability (0.0 to 1.0)
        questions[f"{key}_present"] = {
            "type": "noul",
            "instructions": (
                f"Does this knee MRI radiology report describe or indicate "
                f"{description}? The report may be in any language (English, "
                f"Spanish, Dutch, French, German, Turkish, Greek, etc.)."
            ),
        }

        # Score → returns an interpolated value on the severity gradient
        questions[f"{key}_severity"] = {
            "type": "score",
            "instructions": (
                f"Rate the severity of {description} as described in this "
                f"knee MRI radiology report. If not mentioned or absent, "
                f"rate as 'absent'. The report may be in any language."
            ),
            "criteria": SEVERITY_SCALE,
        }

    return {
        "model": JEV_MODEL,
        "state": report_text,
        "questions": questions,
    }


def encode_report(report_text: str) -> dict:
    """Send a report to Jev and return structured features."""
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


# ── Pretty Printer ─────────────────────────────────────────────────────────────

def print_results(report_text: str, response: dict, label: str = ""):
    """Format Jev results as a readable table."""
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

        # Visual indicator
        if prob >= 0.8:
            signal = "🔴 DETECTED"
        elif prob >= 0.4:
            signal = "🟡 possible"
        else:
            signal = "🟢 absent"

        print(f"  {target:<22} {prob:>5.2f}   {sev_norm:>7.3f}   {sev_conf:>9.2f}   {signal}")

    in_tok = usage.get("input_tokens", 0)
    cost = (in_tok / 1_000_000) * 0.042
    print(f"  {'─' * 66}")
    print(f"  Tokens: {in_tok:,}  |  Cost: ${cost:.5f}  |  Model: {response.get('model', '?')}")
    print(f"{'━' * 72}")


# ── Demo ───────────────────────────────────────────────────────────────────────

# Three real-world-style reports in different languages to demonstrate
# Jev's multilingual capability and calibrated output.

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
        "label": "🇬🇧 English — Normal knee (negative control)",
        "text": (
            "MRI of the right knee without contrast. Findings: The ACL and PCL "
            "are intact. The medial and lateral menisci are normal in morphology "
            "and signal. The medial and lateral collateral ligaments are intact. "
            "No joint effusion. No fracture or bone contusion identified. "
            "The articular cartilage is preserved. Impression: Normal MRI of "
            "the right knee."
        ),
    },
    {
        "label": "🇳🇱 Dutch — ACL tear + bone contusion",
        "text": (
            "MRI knie links. Bevindingen: Complete ruptuur van de voorste "
            "kruisband. Beenmergoedeem ter hoogte van het laterale tibiaplateau "
            "en laterale femurcondyl, passend bij bone bruise. Intacte menisci. "
            "Gering gewrichtseffusie. Conclusie: VKB ruptuur met geassocieerde "
            "bone bruise lateraal compartiment."
        ),
    },
]


def main():
    if not API_KEY:
        print("❌ Set your JEV_API_KEY environment variable first!")
        print("   Sign up at https://console.typesafe.ai")
        return

    print("\n" + "=" * 72)
    print("  🦴 TypeSafe Jev × RSNA Knee Abnormality Detection")
    print("  Feature Encoding Demo — 24 calibrated features per report")
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

    # Summary
    print(f"\n{'=' * 72}")
    print(f"  📊 Summary")
    print(f"  Reports encoded:  {len(DEMO_REPORTS)}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Total cost:       ${total_cost:.5f}")
    print(f"  Extrapolated to 4,407 reports: ~${4407 * (total_cost / len(DEMO_REPORTS)):.2f}")
    print(f"{'=' * 72}")
    print()
    print("  💡 Key Insight: These 24 float features can be used as:")
    print("     • Soft labels for training (instead of hard 0/1)")
    print("     • Auxiliary features for report-image fusion")
    print("     • An independent 'second reader' for ensemble denoising")
    print("     • Confidence-gated filters (trust only where prob > 0.8)")
    print()
    print("  🔗 TypeSafe Jev: https://typesafe.ai")
    print("  🔗 Competition:  https://kaggle.com/competitions/rsna-knee-abnormality-detection")
    print()


if __name__ == "__main__":
    main()
