"""Gemma 4 LLM-based Label Extractor for Knee Radiology Reports.

Uses Gemini API (specifically targeting Gemma 4 / Gemini models) with a structured JSON schema
to parse multilingual reports into the 12 clinical knee abnormality targets, including
confidence weighting and severity extraction as part of the 'Two Readers' Kaggle strategy.
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional, Any
import pandas as pd
from tqdm import tqdm


TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist and medical NLP parser.
Your task is to analyze a knee MRI radiology report (which may be in English, Spanish, French, Dutch, German, Turkish, Greek, Bulgarian, etc.)
and determine the status of 12 specific abnormalities.

For EACH of the 12 findings, output a JSON object containing:
- "status": "YES" (clearly described as present/abnormal), "NO" (explicitly described as absent/normal/intact), or "UNK" (silent/not mentioned/ambiguous).
- "confidence": Float between 0.0 and 1.0. If "UNK" because it is completely unmentioned, use 0.1 or 0.0. If clearly stated, use 1.0.
- "severity": "mild", "moderate", "severe", or null if not applicable.

Respond ONLY with valid JSON in this exact structure:
{
  "ACL": {"status": "YES|NO|UNK", "confidence": 1.0, "severity": "mild|moderate|severe|null"},
  "MCL": {"status": "YES|NO|UNK", "confidence": 1.0, "severity": "mild|moderate|severe|null"},
  ... (all 12 findings)
}
"""

def parse_gemma_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON block from Gemma 4 / LLM response."""
    try:
        clean = raw_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        
        data = json.loads(clean)
        res = {}
        
        for col in TARGET_COLS:
            finding_data = data.get(col, {})
            status = str(finding_data.get("status", "UNK")).upper().strip()
            if status not in ["YES", "NO", "UNK"]:
                status = "UNK"
                
            try:
                confidence = float(finding_data.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
                
            res[col] = status
            res[f"{col}_weight"] = confidence
            
        return res
    except Exception as e:
        return None

def status_to_float(status: str) -> Optional[float]:
    """Map YES/NO/UNK status to numeric values."""
    if status == "YES":
        return 1.0
    elif status == "NO":
        return 0.0
    return 0.5  # UNK maps to 0.5 soft label


class GemmaLabeler:
    """Batch Gemma 4 / Gemini label extraction runner."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def label_report(self, report_text: str):
        """Call Gemini API using Flash to label the report, returning (data, in_tokens, out_tokens)."""
        if not self.api_key:
            print("WARNING: No GEMINI_API_KEY found. Returning UNK.")
            return self._dummy_response(), 0, 0
            
        from google import genai
        from google.genai import types
        
        try:
            client = genai.Client(api_key=self.api_key)
            
            # Upgraded to Gemini 3.5 Flash: state-of-the-art multilingual reasoning at Flash pricing
            model_id = 'gemini-3.5-flash'
            
            prompt = f"{SYSTEM_PROMPT}\n\nReport:\n{report_text}"
            
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            
            parsed = parse_gemma_json_response(response.text)
            if parsed:
                return parsed, in_tokens, out_tokens
            else:
                return self._dummy_response(), in_tokens, out_tokens
                
        except Exception as e:
            print(f"\n[API Error]: {e}")
            return None, 0, 0
            
    def _dummy_response(self) -> Dict[str, Any]:
        res = {}
        for col in TARGET_COLS:
            res[col] = 0.5
            res[f"{col}_weight"] = 0.1
        return res

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import time
    import csv
    
    parser = argparse.ArgumentParser(description="Run Gemini Flash API Labeler")
    parser.add_argument("--input", default="data/raw/train.csv", help="Input CSV")
    parser.add_argument("--output", default="data/processed/report_labels.csv", help="Output CSV")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of reports (for testing)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} not found.")
        sys.exit(1)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if 'Report' not in df.columns:
        if 'report_text' in df.columns:
            df.rename(columns={'report_text': 'Report'}, inplace=True)
        else:
            print(f"ERROR: 'Report' column not found in input CSV. Found columns: {df.columns.tolist()}")
            sys.exit(1)
            
    # Figure out what we've already done so we can resume
    existing_uids = set()
    file_exists = os.path.exists(args.output)
    if file_exists:
        try:
            existing_df = pd.read_csv(args.output)
            if "StudyInstanceUID" in existing_df.columns:
                existing_uids = set(existing_df["StudyInstanceUID"].astype(str))
                print(f"Resume Mode: Found {len(existing_uids)} reports already processed.")
        except Exception:
            pass

    # Filter to only unprocessed reports
    to_process = df[df['Report'].notna()].copy()
    to_process = to_process[~to_process["StudyInstanceUID"].astype(str).isin(existing_uids)]
    
    if args.limit > 0:
        to_process = to_process.head(args.limit)
        
    # Parallel execution settings
    NUM_WORKERS = 10
    print(f"Processing remaining {len(to_process)} reports using {NUM_WORKERS} parallel workers on Gemini 3.5 Flash...")
    
    labeler = GemmaLabeler(api_key=api_key)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Column ordering
    all_cols = []
    for c in TARGET_COLS:
        all_cols.append(c)
        all_cols.append(f"{c}_weight")
    all_cols.append("StudyInstanceUID")
    
    # Pricing for Gemini Flash
    PRICE_PER_1M_INPUT = 0.075
    PRICE_PER_1M_OUTPUT = 0.30
    
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    lock = threading.Lock()
    total_in_tokens = 0
    total_out_tokens = 0
    total_cost = 0.0
    stop_flag = False

    def process_row(row_tuple):
        global total_in_tokens, total_out_tokens, total_cost, stop_flag
        if stop_flag:
            return None
            
        _, row = row_tuple
        uid = str(row["StudyInstanceUID"])
        report = row["Report"]
        
        parsed, in_tok, out_tok = labeler.label_report(report)
        if parsed is None:
            return None
            
        parsed["StudyInstanceUID"] = uid
        
        with lock:
            total_in_tokens += in_tok
            total_out_tokens += out_tok
            cost_in = (total_in_tokens / 1_000_000) * PRICE_PER_1M_INPUT
            cost_out = (total_out_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
            total_cost = cost_in + cost_out
            
            writer.writerow(parsed)
            f.flush()
            
            pbar.set_postfix({"Cost": f"${total_cost:.4f}", "In/Out": f"{in_tok}/{out_tok}"})
            pbar.update(1)
            
            if total_cost > 2.00:
                print(f"\n[!] SAFETY STOP: Cost exceeded $2.00!")
                stop_flag = True
                
        return uid

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        if not file_exists or len(existing_uids) == 0:
            writer.writeheader()
            
        pbar = tqdm(total=len(to_process))
        rows_list = list(to_process.iterrows())
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_row, r) for r in rows_list]
            for future in as_completed(futures):
                if stop_flag:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
        pbar.close()
            
    print(f"Done! Total cost for this run: ${total_cost:.4f}")
