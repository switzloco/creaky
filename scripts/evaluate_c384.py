"""Phase 5 Evaluator: Compare C384 folds against Track N4 folds on Gold & Silver OOF."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def compute_macro_and_per_target(df: pd.DataFrame, pred_prefix: str = "pred_"):
    per_target = {}
    valid = []
    for t in TARGETS:
        yt = df[t].to_numpy(dtype=np.float32)
        yp = df[f"{pred_prefix}{t}"].to_numpy(dtype=np.float32)
        mask = (yt != 0.5) & (~np.isnan(yt)) & (~np.isnan(yp))
        yt_m = (yt[mask] >= 0.5).astype(int)
        yp_m = yp[mask]
        if len(np.unique(yt_m)) == 2:
            auc = float(roc_auc_score(yt_m, yp_m))
            per_target[t] = auc
            valid.append(auc)
        else:
            per_target[t] = float("nan")
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_target


def rank_blend_df(df_a: pd.DataFrame, df_b: pd.DataFrame, weight_b: float = 0.5) -> pd.DataFrame:
    merged = df_a[["StudyInstanceUID", "is_gold"] + TARGETS].copy()
    for t in TARGETS:
        pa = df_a[f"pred_{t}"].to_numpy(dtype=np.float64)
        pb = df_b[f"pred_{t}"].to_numpy(dtype=np.float64)
        ra = rankdata(pa) / len(pa)
        rb = rankdata(pb) / len(pb)
        merged[f"pred_{t}"] = (1.0 - weight_b) * ra + weight_b * rb
    return merged


def evaluate_fold(fold: int, c384_dir: Path, n4_dir: Path):
    c384_p = c384_dir / f"val_preds_fold_{fold}.csv"
    n4_p = n4_dir / f"val_preds_fold_{fold}.csv"
    if not c384_p.exists():
        print(f"[Fold {fold}] No C384 predictions found at {c384_p}; skipping.")
        return

    df_c = pd.read_csv(c384_p)
    print("=" * 88)
    print(f"  PHASE 5 EVALUATION — FOLD {fold} ({len(df_c)} total validation rows)")
    print("=" * 88)

    has_n4 = n4_p.exists()
    if has_n4:
        df_n = pd.read_csv(n4_p)
        # Align by StudyInstanceUID
        df_c = df_c.set_index("StudyInstanceUID")
        df_n = df_n.set_index("StudyInstanceUID")
        common = df_c.index.intersection(df_n.index)
        df_c = df_c.loc[common].reset_index()
        df_n = df_n.loc[common].reset_index()
        df_blend50 = rank_blend_df(df_n, df_c, weight_b=0.50)
        df_blend25 = rank_blend_df(df_n, df_c, weight_b=0.25)

    for subset_name, is_gold_flag in [("SILVER OOF (~870 rows, primary signal)", False), ("GOLD BENCHMARK (58 rows, noisy on rare targets)", True)]:
        sub_c = df_c[df_c["is_gold"] == is_gold_flag].reset_index(drop=True)
        macro_c, pt_c = compute_macro_and_per_target(sub_c)

        print(f"\n--- {subset_name} ---")
        if has_n4:
            sub_n = df_n[df_n["is_gold"] == is_gold_flag].reset_index(drop=True)
            sub_b50 = df_blend50[df_blend50["is_gold"] == is_gold_flag].reset_index(drop=True)
            sub_b25 = df_blend25[df_blend25["is_gold"] == is_gold_flag].reset_index(drop=True)
            macro_n, pt_n = compute_macro_and_per_target(sub_n)
            macro_b50, pt_b50 = compute_macro_and_per_target(sub_b50)
            macro_b25, pt_b25 = compute_macro_and_per_target(sub_b25)

            rows = []
            for t in TARGETS:
                rows.append({
                    "Target": t,
                    "N4 (256px)": pt_n[t],
                    "C384 (384px)": pt_c[t],
                    "Blend 75/25": pt_b25[t],
                    "Blend 50/50": pt_b50[t],
                    "Lift (50/50 vs N4)": pt_b50[t] - pt_n[t],
                })
            res_df = pd.DataFrame(rows)
            print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("-" * 88)
            print(f"Macro AUC | N4 Alone   : {macro_n:.4f}")
            print(f"Macro AUC | C384 Alone : {macro_c:.4f}  ({macro_c - macro_n:+.4f} vs N4)")
            print(f"Macro AUC | Blend 75/25: {macro_b25:.4f}  ({macro_b25 - macro_n:+.4f} vs N4)")
            print(f"Macro AUC | Blend 50/50: {macro_b50:.4f}  ({macro_b50 - macro_n:+.4f} vs N4)")
        else:
            print(f"Macro AUC | C384 Alone : {macro_c:.4f}")
            for t in TARGETS:
                print(f"  {t:18s}: {pt_c[t]:.4f}")


def main():
    c384_dir = Path("checkpoints/c384_cleanlab")
    n4_dir = Path("checkpoints/n4_cleanlab_trained")
    for fold in [0, 1, 2]:
        evaluate_fold(fold, c384_dir, n4_dir)


if __name__ == "__main__":
    main()
