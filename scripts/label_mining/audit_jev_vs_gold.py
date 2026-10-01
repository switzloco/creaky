"""Audit the Jev silver labels against the 58 gold studies, per target.

The question this answers: when Jev (the labeler our training labels come from) says a
finding is absent, how often is it actually absent? If Jev misses many gold positives for a
target, its negative labels for that target are unreliable and should carry less weight in
the loss. This replaces the earlier regex-based audit (scripts/audit_gold_disagreements.py),
which measured the fallback regex labeler, not Jev, and so could not separate "the report
didn't mention it" from "the regex didn't recognise how it was mentioned".

With 58 studies, rare targets have only a handful of positives, so every rate is printed
with its counts. Treat rates based on fewer than ~5 positives as anecdotes.

Usage:
  python scripts/label_mining/audit_jev_vs_gold.py
  python scripts/label_mining/audit_jev_vs_gold.py --jev data/processed/jev_encoded_features.csv \
      --train data/raw/train.csv --threshold 0.5 --out data/processed/jev_vs_gold_audit.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def audit(jev_df: pd.DataFrame, gold_df: pd.DataFrame, threshold: float = 0.5,
          unsure_band: float = 0.2) -> pd.DataFrame:
    """Per-target agreement between Jev probabilities and gold 0/1 labels.

    jev_df:  StudyInstanceUID + "<target>_jev_prob" columns.
    gold_df: StudyInstanceUID + target columns (0/1) for the gold studies.
    """
    jev = jev_df.assign(StudyInstanceUID=jev_df["StudyInstanceUID"].astype(str)).set_index("StudyInstanceUID")
    gold = gold_df.assign(StudyInstanceUID=gold_df["StudyInstanceUID"].astype(str)).set_index("StudyInstanceUID")
    common = gold.index.intersection(jev.index)
    rows = []
    for t in TARGET_COLS:
        g = gold.loc[common, t].astype(float).to_numpy()
        p = jev.loc[common, f"{t}_jev_prob"].astype(float).to_numpy()
        ok = ~np.isnan(g) & ~np.isnan(p)
        g, p = g[ok], p[ok]
        jev_pos = p >= threshold
        pos, neg = g == 1.0, g == 0.0
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        missed = int((pos & ~jev_pos).sum())         # gold positive, Jev says absent
        false_alarm = int((neg & jev_pos).sum())     # gold negative, Jev says present
        jev_neg_total = int((~jev_pos).sum())
        jev_neg_wrong = missed                       # Jev negatives that are gold positive
        unsure = int((np.abs(p - 0.5) < unsure_band).sum())
        rows.append({
            "Target": t,
            "Gold pos": n_pos,
            "Gold neg": n_neg,
            "Jev missed (pos->neg)": f"{missed}/{n_pos}",
            "Miss rate": missed / n_pos if n_pos else np.nan,
            "Jev false alarms (neg->pos)": f"{false_alarm}/{n_neg}",
            "False alarm rate": false_alarm / n_neg if n_neg else np.nan,
            # Of the studies Jev labels negative, the share that are truly negative.
            # This is what decides whether Jev's negative labels deserve full loss weight.
            "Jev-negative precision (NPV)": (jev_neg_total - jev_neg_wrong) / jev_neg_total if jev_neg_total else np.nan,
            "Jev unsure (|p-0.5|<band)": unsure,
            "Few positives": n_pos < 5,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jev", default="data/processed/jev_encoded_features.csv")
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="data/processed/jev_vs_gold_audit.csv")
    args = parser.parse_args()

    jev_df = pd.read_csv(args.jev)
    train = pd.read_csv(args.train)
    gold_df = train[train["ACL"].notna()]
    print(f"Gold studies: {len(gold_df)} | Jev rows: {len(jev_df)} | threshold: {args.threshold}")
    missing = set(gold_df["StudyInstanceUID"].astype(str)) - set(jev_df["StudyInstanceUID"].astype(str))
    if missing:
        print(f"[!] {len(missing)} gold studies have no Jev output; they are left out.", file=sys.stderr)

    result = audit(jev_df, gold_df, threshold=args.threshold)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(result.to_string(index=False))
    result.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print("Read 'Miss rate' and 'Jev-negative precision' together, and ignore rows flagged 'Few positives'.")


if __name__ == "__main__":
    main()
