"""Discrepancy Auditor & Confident Learning Diagnostic Engine.

Audits:
1. Silver Mined Labels vs. 58 Expert Gold-Standard Labels (Error Analysis & Language Breakdown).
2. Vision Model Out-of-Fold (OOF) Predictions vs. Silver Mined Labels (Confident Learning / Silent Finding Detection).
3. Confidence-based sample weight generation for noise-robust training.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Ensure scripts directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.label_mining.regex_labeler import TARGET_COLS

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DiscrepancyAuditor:
    def __init__(self, raw_train_path: str = "data/raw/train.csv", silver_labels_path: str = "data/processed/train_labels.csv"):
        self.raw_train_path = raw_train_path
        self.silver_labels_path = silver_labels_path
        self.raw_df = pd.read_csv(raw_train_path) if os.path.exists(raw_train_path) else None
        self.silver_df = pd.read_csv(silver_labels_path) if os.path.exists(silver_labels_path) else None
        
        # 58 Gold Standard subset
        if self.raw_df is not None:
            self.gold_df = self.raw_df.dropna(subset=["ACL"]).copy()
        else:
            self.gold_df = None

    def audit_raw_extractor_vs_gold(self, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compares raw un-merged extractor predictions (e.g. raw regex/LLM output) against the 58 gold cases."""
        if self.gold_df is None:
            raise FileNotFoundError("Missing raw train.csv containing 58 gold cases.")

        merged = pd.merge(
            self.gold_df[["StudyInstanceUID", "Report"] + TARGET_COLS],
            pred_df[["StudyInstanceUID"] + TARGET_COLS],
            on="StudyInstanceUID",
            suffixes=("_gold", "_pred")
        )

        rows = []
        for col in TARGET_COLS:
            y_gold = merged[f"{col}_gold"].values.astype(float)
            y_pred = merged[f"{col}_pred"].values.astype(float)

            y_pred_bin = (y_pred >= 0.5).astype(int)
            y_gold_bin = (y_gold >= 0.5).astype(int)

            acc = accuracy_score(y_gold_bin, y_pred_bin)
            prec, rec, f1, _ = precision_recall_fscore_support(y_gold_bin, y_pred_bin, average="binary", zero_division=0)

            try:
                auc = roc_auc_score(y_gold_bin, y_pred)
            except Exception:
                auc = np.nan

            fn_count = int(np.sum((y_gold_bin == 1) & (y_pred_bin == 0)))
            fp_count = int(np.sum((y_gold_bin == 0) & (y_pred_bin == 1)))
            uncertain_count = int(np.sum((y_pred > 0.0) & (y_pred < 1.0)))

            rows.append({
                "Target": col,
                "Gold_Positives": int(y_gold_bin.sum()),
                "Accuracy": acc,
                "Precision": prec,
                "Recall (Sens)": rec,
                "F1": f1,
                "AUC": auc,
                "FN (Missed)": fn_count,
                "FP (Overcalled)": fp_count,
                "Uncertain": uncertain_count
            })

        return pd.DataFrame(rows), merged

    def audit_silver_vs_gold(self) -> pd.DataFrame:
        """Compares mined silver labels against 58 gold ground truth studies."""
        if self.gold_df is None or self.silver_df is None:
            raise FileNotFoundError("Missing raw train.csv or silver train_labels.csv")

        merged = pd.merge(
            self.gold_df[["StudyInstanceUID", "Report"] + TARGET_COLS],
            self.silver_df[["StudyInstanceUID"] + TARGET_COLS + [c for c in ["label_source"] if c in self.silver_df.columns]],
            on="StudyInstanceUID",
            suffixes=("_gold", "_silver")
        )

        rows = []
        for col in TARGET_COLS:
            y_gold = merged[f"{col}_gold"].values.astype(float)
            y_silver = merged[f"{col}_silver"].values.astype(float)

            # Binary metrics using threshold 0.5
            y_silver_bin = (y_silver >= 0.5).astype(int)
            y_gold_bin = (y_gold >= 0.5).astype(int)

            acc = accuracy_score(y_gold_bin, y_silver_bin)
            prec, rec, f1, _ = precision_recall_fscore_support(y_gold_bin, y_silver_bin, average="binary", zero_division=0)
            
            try:
                auc = roc_auc_score(y_gold_bin, y_silver)
            except Exception:
                auc = np.nan

            fn_count = int(np.sum((y_gold_bin == 1) & (y_silver_bin == 0)))
            fp_count = int(np.sum((y_gold_bin == 0) & (y_silver_bin == 1)))
            uncertain_count = int(np.sum((y_silver > 0.0) & (y_silver < 1.0)))

            rows.append({
                "Target": col,
                "Gold_Positives": int(y_gold_bin.sum()),
                "Accuracy": acc,
                "Precision": prec,
                "Recall (Sensitivity)": rec,
                "F1": f1,
                "AUC": auc,
                "False_Negatives (Missed)": fn_count,
                "False_Positives (Over-called)": fp_count,
                "Uncertain_Predictions": uncertain_count
            })

        summary_df = pd.DataFrame(rows)
        return summary_df, merged

    def find_high_discrepancy_cases(self, merged_df: pd.DataFrame, target: str, error_type: str = "FN", top_n: int = 5) -> pd.DataFrame:
        """Extracts specific clinical study reports where silver labels failed."""
        gold_col = f"{target}_gold"
        silver_col = f"{target}_silver"

        if error_type.upper() == "FN":
            # True positive missed
            subset = merged_df[(merged_df[gold_col] >= 0.5) & (merged_df[silver_col] < 0.5)].copy()
        elif error_type.upper() == "FP":
            # True negative over-called
            subset = merged_df[(merged_df[gold_col] < 0.5) & (merged_df[silver_col] >= 0.5)].copy()
        else:
            # All disagreements
            subset = merged_df[((merged_df[gold_col] >= 0.5) != (merged_df[silver_col] >= 0.5))].copy()

        return subset[["StudyInstanceUID", "Report", gold_col, silver_col]].head(top_n)

    def audit_model_oof_vs_silver(self, oof_preds_df: pd.DataFrame, loss_threshold: float = 0.70) -> pd.DataFrame:
        """Finds potential label noise and silent findings in silver dataset using Vision OOF predictions.
        
        Args:
            oof_preds_df: DataFrame with StudyInstanceUID and columns for each target probability [0.0, 1.0].
            loss_threshold: Cross-entropy error threshold to flag severe discrepancy.
        """
        merged = pd.merge(
            self.raw_df[["StudyInstanceUID", "Report"]],
            self.silver_df[["StudyInstanceUID"] + TARGET_COLS],
            on="StudyInstanceUID",
            suffixes=("", "_silver")
        )
        merged = pd.merge(
            merged,
            oof_preds_df[["StudyInstanceUID"] + [f"{c}_pred" for c in TARGET_COLS if f"{c}_pred" in oof_preds_df.columns or c in oof_preds_df.columns]],
            on="StudyInstanceUID"
        )

        discrepancies = []
        for col in TARGET_COLS:
            pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
            silver_col = col

            y_silver = merged[silver_col].values.astype(float)
            y_pred = np.clip(merged[pred_col].values.astype(float), 1e-6, 1.0 - 1e-6)

            # Binary cross entropy per sample
            bce_loss = -(y_silver * np.log(y_pred) + (1.0 - y_silver) * np.log(1.0 - y_pred))

            # Silent finding: Model is very confident positive (>= 0.85), Silver label says 0
            silent_mask = (y_pred >= 0.85) & (y_silver <= 0.20)
            # Phantom finding: Model is very confident negative (<= 0.15), Silver label says 1
            phantom_mask = (y_pred <= 0.15) & (y_silver >= 0.80)

            for idx in np.where(silent_mask)[0]:
                discrepancies.append({
                    "StudyInstanceUID": merged.iloc[idx]["StudyInstanceUID"],
                    "Target": col,
                    "Discrepancy_Type": "SILENT_FINDING (Model Pos, Silver Neg)",
                    "Model_Prob": float(y_pred[idx]),
                    "Silver_Label": float(y_silver[idx]),
                    "BCE_Loss": float(bce_loss[idx]),
                    "Report_Snippet": str(merged.iloc[idx]["Report"])[:250]
                })

            for idx in np.where(phantom_mask)[0]:
                discrepancies.append({
                    "StudyInstanceUID": merged.iloc[idx]["StudyInstanceUID"],
                    "Target": col,
                    "Discrepancy_Type": "PHANTOM_FINDING (Model Neg, Silver Pos)",
                    "Model_Prob": float(y_pred[idx]),
                    "Silver_Label": float(y_silver[idx]),
                    "BCE_Loss": float(bce_loss[idx]),
                    "Report_Snippet": str(merged.iloc[idx]["Report"])[:250]
                })

        return pd.DataFrame(discrepancies).sort_values(by="BCE_Loss", ascending=False)

    def generate_confidence_sample_weights(self) -> pd.DataFrame:
        """Generates per-study sample weights based on label confidence and agreement."""
        if self.silver_df is None:
            raise FileNotFoundError("Missing silver labels dataframe.")

        df = self.silver_df.copy()
        
        # 1. Gold standard studies get maximum sample weight 1.0
        is_gold = df["StudyInstanceUID"].isin(self.gold_df["StudyInstanceUID"] if self.gold_df is not None else [])
        
        # 2. Compute uncertainty penalty: count of 0.5 (unknown) targets
        prob_matrix = df[TARGET_COLS].values
        uncertain_count = np.sum((prob_matrix > 0.3) & (prob_matrix < 0.7), axis=1)
        
        # Base weight starts at 0.9 for clean regex/LLM extracted studies
        weights = 0.9 - (uncertain_count * 0.05)
        weights = np.clip(weights, 0.3, 0.9)
        weights[is_gold] = 1.0

        df["sample_weight"] = weights
        return df


def main():
    auditor = DiscrepancyAuditor()
    print("=" * 80)
    print("CREAKY DISCREPANCY AUDIT: SILVER vs. GOLD BENCHMARK")
    print("=" * 80)

    summary_df, merged_df = auditor.audit_silver_vs_gold()
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("SAMPLE FALSE NEGATIVE DIAGNOSTIC (Medial Meniscus):")
    print("=" * 80)
    fn_samples = auditor.find_high_discrepancy_cases(merged_df, target="Medial Meniscus", error_type="FN", top_n=2)
    for _, row in fn_samples.iterrows():
        print(f"\n[Study UID: {row['StudyInstanceUID']}]")
        print(f"Gold: {row['Medial Meniscus_gold']} | Silver: {row['Medial Meniscus_silver']}")
        print(f"Report:\n{row['Report'][:350]}...\n")


if __name__ == "__main__":
    main()
