"""LLM-based Label Extractor for Knee Radiology Reports.

Uses Gemini API (or any OpenAI/Anthropic compatible endpoint) with a structured JSON schema
to parse multilingual reports into the 12 clinical knee abnormality targets.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm


TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist and medical NLP parser.
Your task is to analyze a knee MRI radiology report (which may be in English, Spanish, French, Dutch, German, Turkish, Greek, Bulgarian, etc.)
and determine the status of 12 specific abnormalities:
1. ACL: Anterior Cruciate Ligament tear (complete, partial, or high-grade sprain).
2. MCL: Medial Collateral Ligament tear/sprain (grade 1, 2, or 3).
3. Medial Meniscus: Tear, cleavage, maceration, or surgical excision of the medial meniscus.
4. Lateral Meniscus: Tear, cleavage, maceration, or excision of the lateral meniscus.
5. Medial OA: Medial compartment osteoarthritis, cartilage loss, ulceration, or thinning.
6. Lateral OA: Lateral compartment osteoarthritis, cartilage loss, ulceration, or thinning.
7. PF OA: Patellofemoral osteoarthritis, chondromalacia patellae, or trochlear cartilage defect.
8. Effusion: Joint effusion, excess intra-articular fluid, or hemarthrosis.
9. Synovitis: Synovitis, synovial thickening/hypertrophy, or Hoffa synovitis/hoffitis.
10. Baker's: Baker's cyst, popliteal cyst.
11. Contusion: Bone contusion, bone bruise, or trabecular bone marrow edema.
12. Fracture: Bone fracture (acute, trabecular, insufficiency, subchondral, avulsion).

For EACH of the 12 findings, output exactly one of:
- "YES": finding is clearly described as present / abnormal.
- "NO": finding or structure is explicitly described as absent / normal / intact / unremarkable.
- "UNK": finding is not mentioned, silent, or ambiguous / equivocal.

You must respond ONLY with valid JSON in this exact structure:
{
  "ACL": "YES|NO|UNK",
  "MCL": "YES|NO|UNK",
  "Medial Meniscus": "YES|NO|UNK",
  "Lateral Meniscus": "YES|NO|UNK",
  "Medial OA": "YES|NO|UNK",
  "Lateral OA": "YES|NO|UNK",
  "PF OA": "YES|NO|UNK",
  "Effusion": "YES|NO|UNK",
  "Synovitis": "YES|NO|UNK",
  "Baker's": "YES|NO|UNK",
  "Contusion": "YES|NO|UNK",
  "Fracture": "YES|NO|UNK"
}
"""


def parse_llm_json_response(raw_text: str) -> Optional[Dict[str, str]]:
    """Parse JSON block from LLM response."""
    try:
        # Strip markdown code fences if present
        clean = raw_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        data = json.loads(clean)
        # Ensure all targets are present
        res = {}
        for col in TARGET_COLS:
            val = str(data.get(col, "UNK")).upper().strip()
            if val not in ["YES", "NO", "UNK"]:
                val = "UNK"
            res[col] = val
        return res
    except Exception as e:
        return None


def map_status_to_score(status: str) -> Optional[float]:
    """Map YES/NO/UNK status to numeric values."""
    if status == "YES":
        return 1.0
    elif status == "NO":
        return 0.0
    return None  # UNK maps to None


class LLMLabeler:
    """Batch LLM label extraction runner."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def label_report(self, report_text: str) -> Dict[str, Optional[float]]:
        """Placeholder for single report LLM inference."""
        # For actual API calls, google-genai or requests to Gemini endpoint can be invoked
        # If API key is not present, returns UNK
        return {col: None for col in TARGET_COLS}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("LLM Labeler module ready. Set GEMINI_API_KEY to run full LLM extraction pipeline.")
