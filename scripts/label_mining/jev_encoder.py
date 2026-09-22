"""TypeSafe Jev 'System One' Encoder for Knee Radiology Reports.

Generates calibrated soft-label features using Jev's typed decision primitives
(Noul + Score) as an independent 'second reader' alongside the Gemma 4 LLM labeler.

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

# Clinical descriptions for each target — used in Jev question instructions
# so the model understands the medical context, not just the abbreviation.
TARGET_DESCRIPTIONS = {
    "ACL":              "an anterior cruciate ligament (ACL) tear or injury",
    "MCL":              "a medial collateral ligament (MCL) tear or injury",
    "Medial Meniscus":  "a medial meniscus tear or degeneration",
    "Lateral Meniscus": "a lateral meniscus tear or degeneration",
    "Medial OA":        "medial compartment osteoarthritis (joint space narrowing, osteophytes, cartilage loss)",
    "Lateral OA":       "lateral compartment osteoarthritis (joint space narrowing, osteophytes, cartilage loss)",
    "PF OA":            "patellofemoral osteoarthritis",
    "Effusion":         "joint effusion (excess fluid in the knee joint)",
    "Synovitis":        "synovitis (inflammation of the synovial membrane)",
    "Baker's":          "a Baker's cyst (popliteal cyst)",
    "Contusion":        "a bone contusion or bone bruise",
    "Fracture":         "a bone fracture",
}

# Stable question key names (no apostrophes or spaces)
def _safe_key(target: str) -> str:
    return target.replace("'", "").replace(" ", "_")

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL   = "jev-latest"

# Pricing: $0.042 per 1M input tokens, output is free
PRICE_PER_1M_INPUT = 0.042

# Severity gradient for Score questions (5-level ordinal scale)
SEVERITY_CRITERIA = ["absent", "mild", "moderate", "severe", "critical"]

# Concurrency / safety
NUM_WORKERS = 8
COST_SAFETY_LIMIT = 1.00   # dollars — emergency stop
RETRY_ATTEMPTS = 3
RETRY_BACKOFF  = 1.5       # seconds, multiplied on each retry

# Project root for .env resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/label_mining/../../


# ── Jev API Client ─────────────────────────────────────────────────────────────

class JevEncoder:
    """Calls the TypeSafe Jev API to encode radiology reports as structured features."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _build_payload(self, report_text: str) -> dict:
        """Build a single Jev request with 12 Noul + 12 Score questions."""
        questions = {}

        for target in TARGET_COLS:
            desc = TARGET_DESCRIPTIONS[target]
            sk = _safe_key(target)

            # Noul: "Does this report describe [condition]?"
            questions[f"{sk}_present"] = {
                "type": "noul",
                "instructions": (
                    f"Does this knee MRI radiology report describe or indicate "
                    f"{desc}? Consider any language — the report may be in "
                    f"English, Spanish, Dutch, French, German, Turkish, Greek, "
                    f"Bulgarian, or another language."
                ),
            }

            # Score: "How severe is this finding?"
            questions[f"{sk}_severity"] = {
                "type": "score",
                "instructions": (
                    f"Rate the severity of {desc} as described in this knee MRI "
                    f"radiology report. If not mentioned or absent, rate as 'absent'. "
                    f"Consider any language."
                ),
                "criteria": SEVERITY_CRITERIA,
            }

        return {
            "model": JEV_MODEL,
            "state": report_text,
            "questions": questions,
        }

    def _parse_response(self, resp_json: dict) -> Optional[Dict[str, float]]:
        """Extract 24 float features from a Jev response."""
        answers = resp_json.get("answers", {})
        if not answers:
            return None

        result = {}
        max_severity = len(SEVERITY_CRITERIA) - 1

        for target in TARGET_COLS:
            sk = _safe_key(target)

            # Noul → probability (0.0 to 1.0)
            noul_answer = answers.get(f"{sk}_present", {})
            prob = noul_answer.get("noul", 0.5)
            result[f"{target}_jev_prob"] = float(prob)

            # Score → normalized severity (0–1 scale)
            score_answer = answers.get(f"{sk}_severity", {})
            raw_score = score_answer.get("score", 0.0)  # API returns "score", not "value"
            normalized = float(raw_score) / max_severity if max_severity > 0 else 0.0
            result[f"{target}_jev_severity"] = round(min(max(normalized, 0.0), 1.0), 4)

        return result

    def encode_report(self, report_text: str) -> Tuple[Optional[Dict[str, float]], int]:
        """Send a report to Jev and return (features_dict, input_token_count).

        Returns (None, 0) on unrecoverable failure.
        """
        if not report_text or pd.isna(report_text):
            return None, 0

        payload = self._build_payload(str(report_text))

        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = self.session.post(JEV_API_URL, json=payload, timeout=30)

                if resp.status_code == 429:
                    # Rate-limited — back off and retry
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                body = resp.json()

                # Extract actual token usage from response
                usage = body.get("usage", {})
                in_tokens = usage.get("input_tokens", 0)

                features = self._parse_response(body)
                return features, int(in_tokens)

            except requests.exceptions.Timeout:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            except requests.exceptions.HTTPError as e:
                print(f"\n[Jev HTTP {resp.status_code}] {resp.text[:200]} (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
                if resp.status_code >= 500:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt))
                    continue
                return None, 0
            except Exception as e:
                print(f"\n[Jev Error] {e}")
                return None, 0

        print(f"\n[Jev] Exhausted {RETRY_ATTEMPTS} retries.")
        return None, 0


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_against_gold(jev_csv: str, train_csv: str):
    """Compare Jev Noul probabilities against the 58 gold-standard expert labels.

    Prints per-target AUC and a macro-average.
    """
    from sklearn.metrics import roc_auc_score

    jev_df = pd.read_csv(jev_csv).set_index("StudyInstanceUID")
    train_df = pd.read_csv(train_csv)

    # Gold rows are the ones where ACL is not NaN
    gold = train_df[train_df["ACL"].notna()].copy()
    gold = gold.set_index("StudyInstanceUID")

    # Intersect
    common = gold.index.intersection(jev_df.index)
    if len(common) == 0:
        print("ERROR: No overlap between Jev output and gold labels.")
        return

    print(f"\nValidation: {len(common)} gold-standard studies found in Jev output.\n")
    print(f"{'Target':<20} {'AUC':>6}  {'Pos':>4} {'Neg':>4}")
    print("-" * 40)

    aucs = []
    for target in TARGET_COLS:
        y_true = gold.loc[common, target].values.astype(float)
        y_pred = jev_df.loc[common, f"{target}_jev_prob"].values.astype(float)

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            auc_str = "  N/A"
            print(f"{target:<20} {auc_str:>6}  {n_pos:>4} {n_neg:>4}  (single class)")
        else:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            print(f"{target:<20} {auc:>6.3f}  {n_pos:>4} {n_neg:>4}")

    if aucs:
        macro = np.mean(aucs)
        print("-" * 40)
        print(f"{'Macro-averaged AUC':<20} {macro:>6.3f}")
    else:
        print("\nCould not compute any AUC (not enough class variation in gold labels).")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="TypeSafe Jev Encoder — generate calibrated soft-label features from knee MRI reports"
    )
    parser.add_argument("--input",    default="data/raw/train.csv",                     help="Input CSV with Report column")
    parser.add_argument("--output",   default="data/processed/jev_encoded_features.csv", help="Output CSV")
    parser.add_argument("--limit",    type=int, default=0,                              help="Limit reports (for testing)")
    parser.add_argument("--validate", action="store_true",                              help="Validate Jev output vs 58 gold labels")
    parser.add_argument("--workers",  type=int, default=NUM_WORKERS,                    help="Parallel workers")
    args = parser.parse_args()

    # ── Load env ───────────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    # ── Validate-only mode ─────────────────────────────────────────────────
    if args.validate:
        if not os.path.exists(args.output):
            print(f"ERROR: Jev output file {args.output} not found. Run encoding first.")
            sys.exit(1)
        validate_against_gold(args.output, args.input)
        return

    # ── API key ────────────────────────────────────────────────────────────
    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        print("ERROR: JEV_API_KEY environment variable not set.")
        print("       Add it to your .env file or export it.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} not found.")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    if "Report" not in df.columns:
        print(f"ERROR: 'Report' column not found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # ── Resume support ─────────────────────────────────────────────────────
    existing_uids = set()
    file_exists = os.path.exists(args.output)
    if file_exists:
        try:
            existing_df = pd.read_csv(args.output)
            if "StudyInstanceUID" in existing_df.columns:
                existing_uids = set(existing_df["StudyInstanceUID"].astype(str))
                print(f"Resume mode: {len(existing_uids)} reports already encoded.")
        except Exception:
            pass

    # Filter to reports that have text and haven't been processed
    to_process = df[df["Report"].notna()].copy()
    to_process = to_process[~to_process["StudyInstanceUID"].astype(str).isin(existing_uids)]

    if args.limit > 0:
        to_process = to_process.head(args.limit)

    if len(to_process) == 0:
        print("All reports already encoded. Nothing to do.")
        return

    print(f"\nEncoding {len(to_process)} reports with Jev ({args.workers} workers)...")
    print(f"Model: {JEV_MODEL}")
    print(f"Cost safety limit: ${COST_SAFETY_LIMIT:.2f}\n")

    # ── Output columns ─────────────────────────────────────────────────────
    out_cols = ["StudyInstanceUID"]
    for t in TARGET_COLS:
        out_cols.append(f"{t}_jev_prob")
        out_cols.append(f"{t}_jev_severity")

    # ── Parallel processing ────────────────────────────────────────────────
    encoder = JevEncoder(api_key=api_key)
    lock = threading.Lock()
    total_tokens = 0
    total_cost = 0.0
    success_count = 0
    fail_count = 0
    stop_flag = False

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def process_row(row_tuple):
        nonlocal total_tokens, total_cost, success_count, fail_count, stop_flag
        if stop_flag:
            return None

        _, row = row_tuple
        uid = str(row["StudyInstanceUID"])
        report = row["Report"]

        features, tok_count = encoder.encode_report(report)

        if features is None:
            with lock:
                fail_count += 1
                pbar.update(1)
            return None

        features["StudyInstanceUID"] = uid

        with lock:
            total_tokens += tok_count
            total_cost = (total_tokens / 1_000_000) * PRICE_PER_1M_INPUT
            success_count += 1

            writer.writerow(features)
            f.flush()

            pbar.set_postfix({
                "cost": f"${total_cost:.4f}",
                "ok": success_count,
                "fail": fail_count,
            })
            pbar.update(1)

            if total_cost > COST_SAFETY_LIMIT:
                print(f"\n[!] SAFETY STOP: Cost exceeded ${COST_SAFETY_LIMIT:.2f}!")
                stop_flag = True

        return uid

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        if not file_exists or len(existing_uids) == 0:
            writer.writeheader()

        pbar = tqdm(total=len(to_process), desc="Jev Encoding")
        rows_list = list(to_process.iterrows())

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, r) for r in rows_list]
            for future in as_completed(futures):
                if stop_flag:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        pbar.close()

    print(f"\n{'='*50}")
    print(f"Done! Encoded {success_count} reports, {fail_count} failures.")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Output: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
