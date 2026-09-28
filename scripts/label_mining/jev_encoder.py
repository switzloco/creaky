"""TypeSafe Jev 'System One' Encoder for Knee Radiology Reports (v2).

Generates calibrated soft-label features using Jev's typed decision primitives
(Noul + Score) as an independent 'second reader' alongside the Gemma 4 LLM labeler.

v2 upgrades:
  - Clinically precise, per-target instructions with edge-case handling
  - Structured state prompt ("Act as expert MSK radiologist")
  - Explicit disambiguation rules for Effusion/Synovitis/Contusion

Outputs 24 float columns per study:
  - 12 Noul probabilities (ACL_jev_prob, MCL_jev_prob, ...)
  - 12 Score values         (ACL_jev_severity, MCL_jev_severity, ...)

Usage:
  uv run python scripts/label_mining/jev_encoder.py                          # full run
  uv run python scripts/label_mining/jev_encoder.py --limit 5                # test on 5 reports
  uv run python scripts/label_mining/jev_encoder.py --validate               # compare vs 58 gold labels
"""

import os
import sys
import csv
import time
import json
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# ── Clinically precise Noul instructions (v2) ─────────────────────────────────

NOUL_INSTRUCTIONS = {
    "ACL": (
        "Does this knee MRI radiology report describe anterior cruciate ligament "
        "(ACL) pathology? Answer YES for: explicit tear, rupture, complete or "
        "partial discontinuity, abnormal signal within the ligament, injury, or "
        "a ruptured post-surgical ACL graft. Answer NO for: an intact native ACL, "
        "an intact ACL graft, or if explicitly described as normal/preserved. "
        "The report may be in any language."
    ),
    "MCL": (
        "Does this knee MRI radiology report describe medial collateral ligament "
        "(MCL) pathology? Answer YES for: explicit tear, rupture, sprain, "
        "abnormal signal, thickening with surrounding edema, or injury. "
        "Answer NO for: explicitly normal, intact, or preserved MCL. "
        "The report may be in any language."
    ),
    "Medial Meniscus": (
        "Does this knee MRI radiology report describe medial meniscus pathology? "
        "Answer YES for: explicit tear, rupture, truncation, extrusion, displaced "
        "fragment, bucket-handle tear, maceration, abnormal morphology, or "
        "intrameniscal signal reaching the articular surface (grade 3). "
        "Intrasubstance degeneration without surface extension (grade 1-2) alone "
        "is borderline - lean YES if the report emphasizes it. "
        "Answer NO for: explicitly normal or intact meniscus. "
        "The report may be in any language."
    ),
    "Lateral Meniscus": (
        "Does this knee MRI radiology report describe lateral meniscus pathology? "
        "Answer YES for: explicit tear, rupture, truncation, extrusion, displaced "
        "fragment, bucket-handle tear, maceration, abnormal morphology, or "
        "intrameniscal signal reaching the articular surface (grade 3). "
        "Answer NO for: explicitly normal or intact lateral meniscus. "
        "The report may be in any language."
    ),
    "Medial OA": (
        "Does this knee MRI radiology report describe osteoarthritis of the "
        "MEDIAL tibiofemoral compartment? Answer YES for: medial cartilage loss, "
        "chondral thinning or fissuring, medial joint space narrowing, medial "
        "marginal osteophytes, or subchondral sclerosis/cysts in the medial "
        "compartment. Answer NO for: normal medial compartment or if OA is only "
        "described in the lateral or patellofemoral compartment. "
        "The report may be in any language."
    ),
    "Lateral OA": (
        "Does this knee MRI radiology report describe osteoarthritis of the "
        "LATERAL tibiofemoral compartment? Answer YES for: lateral cartilage "
        "loss, chondral thinning or fissuring, lateral joint space narrowing, "
        "lateral marginal osteophytes, or subchondral sclerosis/cysts in the "
        "lateral compartment. Answer NO for: normal lateral compartment or if "
        "OA is only described in the medial or patellofemoral compartment. "
        "The report may be in any language."
    ),
    "PF OA": (
        "Does this knee MRI radiology report describe patellofemoral "
        "osteoarthritis (PF OA)? Answer YES for: patellofemoral cartilage loss, "
        "patellar or trochlear chondromalacia, patellar or trochlear osteophytes, "
        "or patellofemoral subchondral changes. Answer NO for: normal "
        "patellofemoral joint or if OA is only in tibiofemoral compartments. "
        "The report may be in any language."
    ),
    "Effusion": (
        "Does this knee MRI radiology report describe clinically significant "
        "joint effusion? Answer YES for: moderate, large, marked, or significant "
        "joint effusion, or fluid distending the suprapatellar recess/bursa, or "
        "effusion emphasized in the conclusion/impression section. "
        "Answer NO for: explicitly absent effusion ('no effusion'), OR if fluid "
        "is described ONLY as minimal, trace, small, or physiological. "
        "IMPORTANT: Fluid confined solely to a Baker cyst or peri-articular "
        "soft tissues is NOT joint effusion. "
        "The report may be in any language."
    ),
    "Synovitis": (
        "Does this knee MRI radiology report describe synovitis or synovial "
        "inflammation? Answer YES for: explicit synovitis, synovial thickening, "
        "synovial hypertrophy or proliferation, villonodular changes, Hoffa fat "
        "pad impingement or edema or abnormal signal, symptomatic plica "
        "thickening, or inflammatory suprapatellar bursitis. "
        "IMPORTANT: Joint fluid or effusion ALONE without explicit mention of "
        "synovial thickening or Hoffa edema is NOT synovitis. "
        "Answer NO for: explicitly normal/absent synovium or normal Hoffa. "
        "The report may be in any language."
    ),
    "Baker's": (
        "Does this knee MRI radiology report describe a Baker cyst (popliteal "
        "cyst / gastrocnemius-semimembranosus bursa collection)? Answer YES for: "
        "explicit Baker cyst, popliteal cyst, or fluid collection in the "
        "posterior popliteal fossa consistent with a cyst. "
        "Answer NO for: explicitly absent, or if only physiological trace fluid "
        "in a normal bursa is described. The report may be in any language."
    ),
    "Contusion": (
        "Does this knee MRI radiology report describe a bone contusion or bone "
        "bruise? Answer YES for: explicit bone contusion, bone bruise, or "
        "traumatic bone marrow edema. "
        "IMPORTANT: Do NOT answer YES for isolated subchondral edema that is "
        "attributed purely to chronic osteoarthritis without any trauma-related "
        "wording. Reactive marrow edema adjacent to degenerative changes alone "
        "is NOT a contusion. "
        "Answer NO for: explicitly absent or no marrow signal abnormality. "
        "The report may be in any language."
    ),
    "Fracture": (
        "Does this knee MRI radiology report describe a fracture of any bone "
        "about the knee? Answer YES for: explicit acute or subacute fracture, "
        "cortical disruption, avulsion fracture, occult fracture, stress "
        "fracture, insufficiency fracture, or osteochondral fracture. "
        "Answer NO for: explicitly absent or no fracture identified. "
        "The report may be in any language."
    ),
}

# ── Score (severity) instructions ──────────────────────────────────────────────

SCORE_INSTRUCTIONS = {
    "ACL": "Rate ACL pathology severity. 'absent' if normal/intact. 'mild' for sprain/partial. 'moderate' for significant partial tear. 'severe' for complete rupture. 'critical' for rupture with associated injuries.",
    "MCL": "Rate MCL pathology severity. 'absent' if normal. 'mild' for grade I sprain. 'moderate' for partial tear. 'severe' for complete tear. 'critical' for complete tear with avulsion.",
    "Medial Meniscus": "Rate medial meniscus pathology severity. 'absent' if normal. 'mild' for minor tear/degeneration. 'moderate' for definite tear. 'severe' for complex/displaced tear. 'critical' for macerated meniscus.",
    "Lateral Meniscus": "Rate lateral meniscus pathology severity. 'absent' if normal. 'mild' for minor tear. 'moderate' for definite tear. 'severe' for complex/displaced. 'critical' for macerated.",
    "Medial OA": "Rate medial compartment OA severity. 'absent' if normal. 'mild' for early thinning/small osteophytes. 'moderate' for definite cartilage loss. 'severe' for advanced loss/narrowing. 'critical' for bone-on-bone.",
    "Lateral OA": "Rate lateral compartment OA severity. 'absent' if normal. 'mild' for early changes. 'moderate' for definite loss. 'severe' for advanced. 'critical' for end-stage.",
    "PF OA": "Rate patellofemoral OA severity. 'absent' if normal. 'mild' for early chondromalacia. 'moderate' for definite loss. 'severe' for advanced with osteophytes. 'critical' for end-stage.",
    "Effusion": "Rate joint effusion severity. 'absent' if no or trace fluid. 'mild' for small. 'moderate' for moderate. 'severe' for large. 'critical' for massive/tense.",
    "Synovitis": "Rate synovitis severity. 'absent' if no synovial changes. 'mild' for subtle thickening. 'moderate' for definite synovitis/Hoffa edema. 'severe' for marked proliferation. 'critical' for extensive.",
    "Baker's": "Rate Baker cyst severity. 'absent' if none. 'mild' for small. 'moderate' for medium. 'severe' for large. 'critical' for ruptured with dissection.",
    "Contusion": "Rate bone contusion severity. 'absent' if none. 'mild' for small focal edema. 'moderate' for moderate edema. 'severe' for extensive. 'critical' for near-fracture or multiple contusions.",
    "Fracture": "Rate fracture severity. 'absent' if none. 'mild' for occult/stress. 'moderate' for non-displaced. 'severe' for displaced. 'critical' for comminuted/multi-fragment.",
}

def _safe_key(t: str) -> str:
    return t.replace("'", "").replace(" ", "_")

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
PRICE_PER_1M_INPUT = 0.042
SEVERITY_CRITERIA = ["absent", "mild", "moderate", "severe", "critical"]
NUM_WORKERS = 8
COST_SAFETY_LIMIT = 1.00
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def build_state_prompt(report_text: str) -> str:
    return (
        "Act as an expert musculoskeletal radiologist.\n"
        "Read the following knee MRI radiology report in its original language "
        "(English, Spanish, Dutch, French, German, Turkish, Greek, Bulgarian, etc.).\n"
        "Use ONLY explicitly stated findings. Do not infer from clinical history "
        "or indications.\n\nKNEE MRI RADIOLOGY REPORT:\n\"\"\"\n"
        + report_text.strip() + "\n\"\"\"\n\n"
        "Task: Evaluate status and severity of all requested knee structures "
        "based strictly on the report text above."
    )

class JevEncoder:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})

    def _build_payload(self, report_text: str) -> dict:
        questions = {}
        for target in TARGET_COLS:
            sk = _safe_key(target)
            questions[f"{sk}_present"] = {"type": "noul", "instructions": NOUL_INSTRUCTIONS[target]}
            questions[f"{sk}_severity"] = {"type": "score", "instructions": SCORE_INSTRUCTIONS[target], "criteria": SEVERITY_CRITERIA}
        return {"model": JEV_MODEL, "state": build_state_prompt(report_text), "questions": questions}

    def _parse_response(self, resp_json: dict) -> Optional[Dict[str, float]]:
        answers = resp_json.get("answers", {})
        if not answers: return None
        result = {}
        mx = len(SEVERITY_CRITERIA) - 1
        for target in TARGET_COLS:
            sk = _safe_key(target)
            result[f"{target}_jev_prob"] = float(answers.get(f"{sk}_present", {}).get("noul", 0.5))
            raw = float(answers.get(f"{sk}_severity", {}).get("score", 0.0))
            result[f"{target}_jev_severity"] = round(min(max(raw / mx, 0.0), 1.0), 4)
        return result

    def encode_report(self, report_text: str) -> Tuple[Optional[Dict[str, float]], int]:
        if not report_text or pd.isna(report_text): return None, 0
        payload = self._build_payload(str(report_text))
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = self.session.post(JEV_API_URL, json=payload, timeout=30)
                if resp.status_code == 429:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt)); continue
                resp.raise_for_status()
                body = resp.json()
                return self._parse_response(body), int(body.get("usage", {}).get("input_tokens", 0))
            except requests.exceptions.Timeout:
                time.sleep(RETRY_BACKOFF * (attempt + 1)); continue
            except requests.exceptions.HTTPError:
                print(f"\n[Jev HTTP {resp.status_code}] {resp.text[:200]} (attempt {attempt+1}/{RETRY_ATTEMPTS})")
                if resp.status_code >= 500: time.sleep(RETRY_BACKOFF * (2 ** attempt)); continue
                return None, 0
            except Exception as e:
                print(f"\n[Jev Error] {e}"); return None, 0
        return None, 0

def validate_against_gold(jev_csv: str, train_csv: str):
    from sklearn.metrics import roc_auc_score
    jev_df = pd.read_csv(jev_csv).set_index("StudyInstanceUID")
    gold = pd.read_csv(train_csv); gold = gold[gold["ACL"].notna()].set_index("StudyInstanceUID")
    common = gold.index.intersection(jev_df.index)
    if len(common) == 0: print("ERROR: No overlap."); return
    print(f"\nValidation: {len(common)} gold-standard studies.\n")
    print(f"{'Target':<20} {'AUC':>6}  {'Pos':>4} {'Neg':>4}")
    print("-" * 40)
    aucs = []
    for target in TARGET_COLS:
        y_true = gold.loc[common, target].values.astype(float)
        y_pred = jev_df.loc[common, f"{target}_jev_prob"].values.astype(float)
        n_pos, n_neg = int(y_true.sum()), len(y_true) - int(y_true.sum())
        if n_pos == 0 or n_neg == 0:
            print(f"{target:<20} {'N/A':>6}  {n_pos:>4} {n_neg:>4}  (single class)")
        else:
            auc = roc_auc_score(y_true, y_pred); aucs.append(auc)
            print(f"{target:<20} {auc:>6.3f}  {n_pos:>4} {n_neg:>4}")
    if aucs:
        print("-" * 40); print(f"{'Macro-averaged AUC':<20} {np.mean(aucs):>6.3f}")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8")
        except: pass
    parser = argparse.ArgumentParser(description="Jev Encoder v2")
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output", default="data/processed/jev_encoded_features.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
    except ImportError: pass

    if args.validate:
        if not os.path.exists(args.output): print(f"ERROR: {args.output} not found."); sys.exit(1)
        validate_against_gold(args.output, args.input); return

    api_key = os.environ.get("JEV_API_KEY")
    if not api_key: print("ERROR: JEV_API_KEY not set."); sys.exit(1)
    if not os.path.exists(args.input): print(f"ERROR: {args.input} not found."); sys.exit(1)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    if "Report" not in df.columns: print("ERROR: 'Report' column not found."); sys.exit(1)

    existing_uids = set(); file_exists = os.path.exists(args.output)
    if file_exists:
        try:
            edf = pd.read_csv(args.output)
            if "StudyInstanceUID" in edf.columns:
                existing_uids = set(edf["StudyInstanceUID"].astype(str))
                print(f"Resume: {len(existing_uids)} already encoded.")
        except: pass

    to_process = df[df["Report"].notna()].copy()
    to_process = to_process[~to_process["StudyInstanceUID"].astype(str).isin(existing_uids)]
    if args.limit > 0: to_process = to_process.head(args.limit)
    if len(to_process) == 0: print("All reports already encoded."); return

    print(f"\nEncoding {len(to_process)} reports with Jev v2 ({args.workers} workers)...")
    print(f"Model: {JEV_MODEL} | Safety limit: ${COST_SAFETY_LIMIT:.2f}\n")

    out_cols = ["StudyInstanceUID"]
    for t in TARGET_COLS: out_cols += [f"{t}_jev_prob", f"{t}_jev_severity"]

    encoder = JevEncoder(api_key); lock = threading.Lock()
    total_tokens = 0; total_cost = 0.0; success_count = 0; fail_count = 0; stop_flag = False
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def process_row(row_tuple):
        nonlocal total_tokens, total_cost, success_count, fail_count, stop_flag
        if stop_flag: return None
        _, row = row_tuple
        uid = str(row["StudyInstanceUID"])
        features, tok = encoder.encode_report(row["Report"])
        if features is None:
            with lock: fail_count += 1; pbar.update(1)
            return None
        features["StudyInstanceUID"] = uid
        with lock:
            total_tokens += tok; total_cost = (total_tokens / 1e6) * PRICE_PER_1M_INPUT
            success_count += 1; writer.writerow(features); f.flush()
            pbar.set_postfix({"cost": f"${total_cost:.4f}", "ok": success_count, "fail": fail_count}); pbar.update(1)
            if total_cost > COST_SAFETY_LIMIT: print(f"\n[!] SAFETY STOP: >${COST_SAFETY_LIMIT:.2f}!"); stop_flag = True
        return uid

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        if not file_exists or len(existing_uids) == 0: writer.writeheader()
        pbar = tqdm(total=len(to_process), desc="Jev v2 Encoding")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, r) for r in list(to_process.iterrows())]
            for future in as_completed(futures):
                if stop_flag: executor.shutdown(wait=False, cancel_futures=True); break
        pbar.close()
    print(f"\n{'='*50}\nDone! {success_count} encoded, {fail_count} failures.")
    print(f"Tokens: {total_tokens:,} | Cost: ${total_cost:.4f}\nOutput: {args.output}\n{'='*50}")

if __name__ == "__main__": main()
