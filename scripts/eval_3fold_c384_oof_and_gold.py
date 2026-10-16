"""Evaluates the complete 6-Model Ensemble (3-Fold C384 384px CoAtNet + 3-Fold N4 256px ConvNeXt) across:
1. Fold 2 Silver OOF (870 studies)
2. Combined 3-Fold Silver OOF (2,610 studies: Fold 0 + Fold 1 + Fold 2)
3. Out-of-Sample 58-Study Gold Benchmark
4. Updates Stage 4 C384_ADAPTIVE_WEIGHTS in creaky-apex-cleanlab-blend.ipynb & submission_notebook.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

C384_DIR = "checkpoints/c384_cleanlab"
N4_DIR = "checkpoints/n4_cleanlab_trained"
APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
SUB_PY = "scripts/submission/submission_notebook.py"
SUB_NB = "notebooks/submission-notebook/submission-notebook.ipynb"


def score_target(yt: np.ndarray, yp: np.ndarray) -> float:
    mask = ~np.isnan(yt) & (yt != 0.5)
    if mask.sum() < 2:
        return float("nan")
    yb = (yt[mask] >= 0.5).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return float(roc_auc_score(yb, yp[mask]))


def compute_macro_auc(df: pd.DataFrame, pred_prefix: str = "pred_") -> tuple[float, dict]:
    per_class = {}
    for t in TARGET_COLS:
        auc = score_target(df[t].values, df[f"{pred_prefix}{t}"].values)
        if not np.isnan(auc):
            per_class[t] = auc
    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return macro, per_class


def update_stage4_weights(weights: dict) -> None:
    lines = ["        C384_ADAPTIVE_WEIGHTS = {"]
    for t in TARGET_COLS:
        lines.append(f'            "{t}": {weights[t]:.3f},')
    lines.append("        }")
    new_block = "\n".join(lines)

    pattern = re.compile(r"        C384_ADAPTIVE_WEIGHTS = \{[\s\S]*?        \}")

    with open(APEX_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)
    c11 = "".join(nb["cells"][11]["source"])
    c11_new, n_sub = pattern.subn(new_block, c11, count=1)
    assert n_sub == 1, "Could not replace C384_ADAPTIVE_WEIGHTS in apex notebook"
    nb["cells"][11]["source"] = [l for l in c11_new.splitlines(keepends=True)]
    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    with open(SUB_PY, "r", encoding="utf-8") as f:
        sub_src = f.read()
    sub_new, n_sub2 = pattern.subn(new_block, sub_src, count=1)
    assert n_sub2 == 1, "Could not replace C384_ADAPTIVE_WEIGHTS in submission_notebook.py"
    with open(SUB_PY, "w", encoding="utf-8") as f:
        f.write(sub_new)

    if os.path.exists(SUB_NB):
        with open(SUB_NB, "r", encoding="utf-8") as f:
            snb = json.load(f)
        for cell in snb["cells"]:
            if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
                cell["source"] = [l for l in sub_new.splitlines(keepends=True)]
        with open(SUB_NB, "w", encoding="utf-8") as f:
            json.dump(snb, f, indent=1, ensure_ascii=False)
    print("Updated C384_ADAPTIVE_WEIGHTS across all submission notebooks!")


def main():
    c_dfs, n_dfs = [], []
    for f in [0, 1, 2]:
        cp = os.path.join(C384_DIR, f"val_preds_fold_{f}.csv")
        np_path = os.path.join(N4_DIR, f"val_preds_fold_{f}.csv")
        if not os.path.exists(cp):
            print(f"[Waiting] {cp} not found yet.")
            return
        c_dfs.append(pd.read_csv(cp))
        n_dfs.append(pd.read_csv(np_path))

    c_silvers, n_silvers = [], []
    for f in [0, 1, 2]:
        cs = c_dfs[f][c_dfs[f]["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
        ns = n_dfs[f][n_dfs[f]["is_gold"] == False].sort_values("StudyInstanceUID").reset_index(drop=True)
        assert (cs["StudyInstanceUID"].values == ns["StudyInstanceUID"].values).all()
        c_silvers.append(cs)
        n_silvers.append(ns)
        nm, _ = compute_macro_auc(ns)
        cm, _ = compute_macro_auc(cs)
        print(f"Fold {f} Silver OOF (870 studies): N4={nm:.4f} | C384={cm:.4f}")

    c_oof = pd.concat(c_silvers, ignore_index=True)
    n_oof = pd.concat(n_silvers, ignore_index=True)
    n_oof_macro, n_oof_pc = compute_macro_auc(n_oof)
    c_oof_macro, c_oof_pc = compute_macro_auc(c_oof)

    opt_w_3fold = {}
    b_oof_opt = n_oof.copy()
    for t in TARGET_COLS:
        yt = n_oof[t].values
        pn = n_oof[f"pred_{t}"].values
        pc = c_oof[f"pred_{t}"].values
        best_w, best_a = 0.25, -1.0
        for w in np.linspace(0.10, 0.50, 41):
            a = score_target(yt, (1.0 - w) * pn + w * pc)
            if a > best_a:
                best_a = a
                best_w = float(round(w, 3))
        opt_w_3fold[t] = best_w
        b_oof_opt[f"pred_{t}"] = (1.0 - best_w) * pn + best_w * pc

    b_oof_macro, b_oof_pc = compute_macro_auc(b_oof_opt)
    print("\n=================================================================")
    print("COMBINED 3-FOLD SILVER OOF (2,610 Held-Out Studies: Folds 0, 1, 2)")
    print("=================================================================")
    print(f"  N4 3-Fold OOF (256px ConvNeXt-S)  : {n_oof_macro:.4f}")
    print(f"  C384 3-Fold OOF (384px CoAtNet)   : {c_oof_macro:.4f}")
    print(f"  Adaptive 6-Model Blend OOF        : {b_oof_macro:.4f} ({b_oof_macro - n_oof_macro:+.4f})")
    for t in TARGET_COLS:
        print(f"    {t:18s}: N4={n_oof_pc[t]:.4f} | C384={c_oof_pc[t]:.4f} | Blend={b_oof_pc[t]:.4f} ({b_oof_pc[t] - n_oof_pc[t]:+.4f}) [w_c384={opt_w_3fold[t]:.2f}]")

    # Gold Benchmark (58 studies)
    c_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in c_dfs]
    n_golds = [df[df["is_gold"] == True].sort_values("StudyInstanceUID").reset_index(drop=True) for df in n_dfs]
    n3_gold = n_golds[0].copy()
    c3_gold = c_golds[0].copy()
    blend_gold = n_golds[0].copy()
    for t in TARGET_COLS:
        n3_gold[f"pred_{t}"] = np.mean([g[f"pred_{t}"].values for g in n_golds], axis=0)
        c3_gold[f"pred_{t}"] = np.mean([g[f"pred_{t}"].values for g in c_golds], axis=0)
        w = opt_w_3fold[t]
        blend_gold[f"pred_{t}"] = (1.0 - w) * n3_gold[f"pred_{t}"].values + w * c3_gold[f"pred_{t}"].values

    n3_gm, _ = compute_macro_auc(n3_gold)
    c3_gm, _ = compute_macro_auc(c3_gold)
    bg_m, _ = compute_macro_auc(blend_gold)
    print("\n=================================================================")
    print("OUT-OF-SAMPLE 58-STUDY GOLD BENCHMARK (6 MODELS: 3x N4 + 3x C384)")
    print("=================================================================")
    print(f"  3-Fold N4 Solo (256px ConvNeXt-S) : {n3_gm:.4f}")
    print(f"  3-Fold C384 Solo (384px CoAtNet)  : {c3_gm:.4f}")
    print(f"  6-Model Blend (3x N4 + 3x C384)   : {bg_m:.4f} ({bg_m - n3_gm:+.4f})")

    with open(os.path.join(C384_DIR, "optimal_c384_blend_weights.json"), "w", encoding="utf-8") as f:
        json.dump(opt_w_3fold, f, indent=2)
    update_stage4_weights(opt_w_3fold)


if __name__ == "__main__":
    main()
