"""Assemble the master training label set (train_labels.csv).

Merges:
1. Gold standard labels (58 studies with expert manual annotation) -> priority 1
2. Regex extractions (multilingual rule matches) -> priority 2
3. LLM extractions (if available) -> priority 3
4. Soft prior (0.5) for unresolved UNKs -> priority 4

Outputs:
data/processed/train_labels.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from regex_labeler import RegexLabeler, TARGET_COLS


def assemble_training_labels(train_csv_path: str, output_csv_path: str, llm_csv_path: str = None) -> pd.DataFrame:
    """Build train_labels.csv from gold standard + regex + optional LLM."""
    print(f"Loading training data from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    total_studies = len(df)
    print(f"Total training studies: {total_studies}")

    # 1. Gold standard
    gold_mask = ~df["ACL"].isna()
    gold_count = gold_mask.sum()
    print(f"Expert gold-annotated studies: {gold_count}")

    # 2. Run regex labeler on all studies
    print("Running multilingual regex labeler across all studies...")
    labeler = RegexLabeler()
    regex_preds = labeler.label_dataframe(df).set_index("StudyInstanceUID")

    # 3. Load LLM preds if provided
    llm_preds = None
    if llm_csv_path and os.path.exists(llm_csv_path):
        print(f"Loading LLM labels from: {llm_csv_path}")
        llm_preds = pd.read_csv(llm_csv_path).set_index("StudyInstanceUID")

    # Combine into master dataframe
    assembled_rows = []
    
    for _, row in df.iterrows():
        study_uid = row["StudyInstanceUID"]
        out_row = {"StudyInstanceUID": study_uid}
        sources = []

        is_gold = not pd.isna(row["ACL"])
        reg_row = regex_preds.loc[study_uid] if study_uid in regex_preds.index else None

        for target in TARGET_COLS:
            val = 0.5
            weight = 0.1
            src = "soft_unk"

            if is_gold:
                val = float(row[target])
                weight = 1.0  # Gold standard is 100% confident
                src = "gold"
            else:
                regex_val = reg_row[target] if reg_row is not None else np.nan
                llm_val = None
                llm_weight = 0.5
                if llm_preds is not None and study_uid in llm_preds.index:
                    llm_val = llm_preds.loc[study_uid, target]
                    if f"{target}_weight" in llm_preds.columns:
                        llm_weight = float(llm_preds.loc[study_uid, f"{target}_weight"])

                if llm_val is not None and not pd.isna(llm_val):
                    val = float(llm_val)
                    weight = llm_weight
                    src = "llm"
                elif not pd.isna(regex_val):
                    val = float(regex_val)
                    weight = 0.8  # Regex is decently confident if it triggers
                    src = "regex"
                else:
                    val = 0.5  # Soft uncertainty label
                    weight = 0.1  # Very low confidence for UNK
                    src = "soft_unk"

            out_row[target] = val
            out_row[f"{target}_weight"] = weight
            sources.append(src)

        if "gold" in sources:
            out_row["label_source"] = "gold"
        elif "llm" in sources:
            out_row["label_source"] = "llm"
        elif "regex" in sources:
            out_row["label_source"] = "regex"
        else:
            out_row["label_source"] = "soft_unk"

        assembled_rows.append(out_row)

    out_df = pd.DataFrame(assembled_rows)
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved assembled training labels to: {output_csv_path}")
    print("Label source breakdown:")
    print(out_df["label_source"].value_counts())

    return out_df


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Assemble train_labels.csv")
    parser.add_argument("--train_csv", default="data/raw/train.csv")
    parser.add_argument("--out_csv", default="data/processed/train_labels.csv")
    parser.add_argument("--llm_csv", default=None)
    args = parser.parse_args()

    assemble_training_labels(args.train_csv, args.out_csv, args.llm_csv)
