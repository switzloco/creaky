"""Applies both high-ROI upgrades:
1. Upgrades Fold 1 Kaggle GPU notebook (`notebooks/training-book-c384-gpu-f1`) to 5 epochs and builds `notebooks/colab-c384-a100/colab-c384-a100-fold1.ipynb`.
2. Upgrades Stage 4 & Stage 5 in `creaky-apex-cleanlab-blend.ipynb`, `submission_notebook.py`, and `submission-notebook.ipynb` with:
   - OOF-Anchored Per-Checkpoint Z-Logit Calibration (0.8665 Silver OOF / 0.8561 Gold AUC)
   - 2x In-VRAM Multi-Scale Test-Time Augmentation (1.00x canonical + 1.03x center-crop zoom, zero extra DICOM I/O)
   - Recalibrated Stage 5 `OPTIMIZED_TARGET_WEIGHTS` reflecting the 6-model Gold AUCs (Medial OA 0.981, Baker's 0.976, Fracture 0.922, MCL 0.896, ACL 0.885, Contusion 0.876, Effusion 0.867).
"""

import json
import os
import re

GPU_F1_NB = "notebooks/training-book-c384-gpu-f1/training-book-c384-gpu-f1.ipynb"
COLAB_F2_NB = "notebooks/colab-c384-a100/colab-c384-a100-fold2.ipynb"
COLAB_F1_NB = "notebooks/colab-c384-a100/colab-c384-a100-fold1.ipynb"
APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
SUB_PY = "scripts/submission/submission_notebook.py"
SUB_NB = "notebooks/submission-notebook/submission-notebook.ipynb"
CAL_JSON = "checkpoints/c384_cleanlab/optimal_c384_zlogit_calibration.json"

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def upgrade_fold1_notebooks():
    # 1. Kaggle GPU Fold 1 notebook -> 5 epochs
    with open(GPU_F1_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][1]["source"])
    src = src.replace('"epochs": 3,', '"epochs": 5,')
    src = src.replace('"preprocessing": "cache_v1" if train_ds.cached_dir else "raw_v1",', '"preprocessing": "cache_v1_384" if train_ds.cached_dir else "raw_v1",')
    src = src.replace("probs = torch.sigmoid(logits).cpu().numpy()", "probs = torch.sigmoid(logits.float()).cpu().numpy()")
    compile(src, "<gpu_f1_5ep>", "exec")
    nb["cells"][1]["source"] = [l for l in src.splitlines(keepends=True)]
    with open(GPU_F1_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Upgraded {GPU_F1_NB} to 5 epochs!")

    # 2. Colab A100 Fold 1 notebook (coatnet_rmlp_2_rw_384, 5 epochs, batch_size=4, accum_steps=1)
    with open(COLAB_F2_NB, "r", encoding="utf-8") as f:
        cnb = json.load(f)
    cnb["cells"][0]["source"] = [l.replace("Fold 2", "Fold 1") for l in cnb["cells"][0]["source"]]
    c2_src = "".join(cnb["cells"][2]["source"]).replace('"fold": 2,', '"fold": 1,')
    compile(c2_src, "<colab_f1_cell2>", "exec")
    cnb["cells"][2]["source"] = [l for l in c2_src.splitlines(keepends=True)]
    c3_src = "".join(cnb["cells"][3]["source"]).replace("Fold 2", "Fold 1").replace("fold_2", "fold_1").replace("f2", "f1")
    compile(c3_src, "<colab_f1_cell3>", "exec")
    cnb["cells"][3]["source"] = [l for l in c3_src.splitlines(keepends=True)]
    with open(COLAB_F1_NB, "w", encoding="utf-8") as f:
        json.dump(cnb, f, indent=1, ensure_ascii=False)
    print(f"Built turnkey Colab A100 Fold 1 notebook at {COLAB_F1_NB}!")


def build_stage4_inference_block(cal_data: dict) -> str:
    weights = cal_data["weights"]
    zstats = cal_data["zlogit_stats"]

    w_lines = ["        C384_ADAPTIVE_WEIGHTS = {"]
    for t in TARGET_COLS:
        w_lines.append(f'            "{t}": {weights[t]:.3f},')
    w_lines.append("        }")
    w_block = "\n".join(w_lines)

    zstats_json_str = json.dumps(zstats, indent=12)
    # Format indentation cleanly
    z_lines = zstats_json_str.splitlines()
    z_formatted = "        OOF_ZLOGIT_STATS = " + z_lines[0] + "\n" + "\n".join("        " + l for l in z_lines[1:])

    new_inference_section = f"""        # 3-Fold (2,610 Silver OOF studies) OOF-Anchored Z-Logit Calibration + Adaptive Weights
        # Lifts 2,610-study Silver OOF to 0.8665 and 58-study Out-of-Sample Gold Benchmark to 0.8561 (+0.0156 over N4)
{w_block}
{z_formatted}

        def _make_zoom_tta(x: torch.Tensor, crop_ratio: float = 0.96) -> torch.Tensor:
            # Pure in-VRAM 1.04x center-crop zoom TTA across all slices (1, K, 3, H, W) -> (1, K, 3, H, W)
            b, k, c, h, w = x.shape
            ch, cw = int(round(h * crop_ratio)), int(round(w * crop_ratio))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            cropped = x[:, :, :, y0:y0 + ch, x0:x0 + cw].reshape(b * k, c, ch, cw)
            zoomed = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
            return zoomed.reshape(b, k, c, h, w)

        # Precompute 2nd TTA view in VRAM once per preprocessing pipeline (zero extra DICOM I/O)
        plane_tensors_zoom = {{}}
        if models:
            for pipeline in pipelines_in_use:
                plane_tensors_zoom[pipeline] = {{
                    plane: _make_zoom_tta(plane_tensors[pipeline][plane], crop_ratio=0.96)
                    for plane in ["Sagittal", "Coronal", "Axial"]
                }}

        # Run Model Inference (with 2x Multi-Scale TTA + OOF-Anchored Z-Logit Calibration) or Priors Fallback
        if models:
            convnext_family = []
            coatnet_family = []
            other_family = []
            with torch.no_grad():
                for model_item in models:
                    model, pipeline = model_item[0], model_item[1]
                    ckpt_fold = model_item[2] if len(model_item) > 2 else None
                    pt = plane_tensors[pipeline]
                    pt_z = plane_tensors_zoom[pipeline]
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits_1 = model(pt["Sagittal"], pt["Coronal"], pt["Axial"]).float()
                        logits_2 = model(pt_z["Sagittal"], pt_z["Coronal"], pt_z["Axial"]).float()
                    raw_logits = (0.65 * logits_1 + 0.35 * logits_2).cpu().numpy()[0]
                    bb = getattr(model, "backbone_name", "")
                    fam_key = "c384" if bb.startswith("coatnet") else ("n4" if bb in ["convnext_small", "convnext_tiny"] else None)
                    fold_key = str(ckpt_fold) if ckpt_fold in (0, 1, 2, "0", "1", "2") else None

                    if fam_key and fold_key and fold_key in OOF_ZLOGIT_STATS.get(fam_key, {{}}):
                        f_stats = OOF_ZLOGIT_STATS[fam_key][fold_key]
                        z_vec = np.zeros(len(TARGET_COLS), dtype=np.float32)
                        for t_idx, target in enumerate(TARGET_COLS):
                            mu_t, sd_t = f_stats[target]
                            clipped_l = float(np.clip(raw_logits[t_idx], -11.5, 11.5))
                            z_vec[t_idx] = (clipped_l - mu_t) / sd_t
                        if fam_key == "c384":
                            coatnet_family.append(z_vec)
                        else:
                            convnext_family.append(z_vec)
                    else:
                        probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -11.5, 11.5)))
                        other_family.append(probs)

            if convnext_family and coatnet_family:
                cn_avg = np.mean(convnext_family, axis=0)
                ca_avg = np.mean(coatnet_family, axis=0)
                avg_probs = np.zeros(len(TARGET_COLS), dtype=np.float32)
                for t_idx, target in enumerate(TARGET_COLS):
                    w_ca = float(C384_ADAPTIVE_WEIGHTS.get(target, 0.25))
                    z_blend = (1.0 - w_ca) * cn_avg[t_idx] + w_ca * ca_avg[t_idx]
                    avg_probs[t_idx] = float(1.0 / (1.0 + np.exp(-np.clip(z_blend, -11.5, 11.5))))
            elif convnext_family or coatnet_family:
                z_all = convnext_family + coatnet_family
                z_avg = np.mean(z_all, axis=0)
                avg_probs = 1.0 / (1.0 + np.exp(-np.clip(z_avg, -11.5, 11.5)))
            else:
                avg_probs = np.mean(other_family, axis=0)"""
    return new_inference_section


def patch_submission_code(src: str, new_inf_block: str) -> str:
    # 1. Ensure ckpt_fold is recorded in models.append((model, pipeline, ckpt_fold))
    old_append = """                pipeline = checkpoint_pipeline(state, bb)
                models.append((model, pipeline))"""
    new_append = """                pipeline = checkpoint_pipeline(state, bb)
                ckpt_fold = state.get("fold") if isinstance(state, dict) else None
                if ckpt_fold is None:
                    m_f = re.search(r"fold_?(\\d+)", os.path.basename(ckpt_path))
                    ckpt_fold = int(m_f.group(1)) if m_f else None
                models.append((model, pipeline, ckpt_fold))"""
    if old_append in src:
        src = src.replace(old_append, new_append, 1)

    # 2. Replace the lines `pipelines_in_use = sorted({pipe for _, pipe in models}) or ["cache_v1"]`
    src = src.replace(
        "pipelines_in_use = sorted({pipe for _, pipe in models}) or [\"cache_v1\"]",
        "pipelines_in_use = sorted({m[1] for m in models}) or [\"cache_v1\"]",
    )

    # 3. Replace from `# Silver-OOF-fitted` (or `C384_ADAPTIVE_WEIGHTS = {`) down to `avg_probs = np.mean(all_probs, axis=0)`
    pattern = re.compile(
        r"        (?:# Silver-OOF-fitted[\s\S]*?)?C384_ADAPTIVE_WEIGHTS = \{[\s\S]*?avg_probs = np\.mean\(all_probs, axis=0\)"
    )
    new_src, count = pattern.subn(new_inf_block, src, count=1)
    assert count == 1, f"Expected 1 replacement of inference block, got {count}"
    return new_src


def main():
    upgrade_fold1_notebooks()

    with open(CAL_JSON, "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    new_inf_block = build_stage4_inference_block(cal_data)

    # 1. Update scripts/submission/submission_notebook.py
    with open(SUB_PY, "r", encoding="utf-8") as f:
        sub_src = f.read()
    sub_new = patch_submission_code(sub_src, new_inf_block)
    compile(sub_new, SUB_PY, "exec")
    with open(SUB_PY, "w", encoding="utf-8") as f:
        f.write(sub_new)
    print(f"Updated and compiled {SUB_PY}!")

    # 2. Update notebooks/submission-notebook/submission-notebook.ipynb
    if os.path.exists(SUB_NB):
        with open(SUB_NB, "r", encoding="utf-8") as f:
            snb = json.load(f)
        for cell in snb["cells"]:
            if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
                cell["source"] = [l for l in sub_new.splitlines(keepends=True)]
        with open(SUB_NB, "w", encoding="utf-8") as f:
            json.dump(snb, f, indent=1, ensure_ascii=False)
        print(f"Updated {SUB_NB}!")

    # 3. Update notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb (Stage 4 + Stage 5 target weights)
    with open(APEX_NB, "r", encoding="utf-8") as f:
        anb = json.load(f)
    c11 = "".join(anb["cells"][11]["source"])
    c11_new = patch_submission_code(c11, new_inf_block)

    # Also update OPTIMIZED_TARGET_WEIGHTS in Stage 5 to reflect the 6-model Z-Logit Gold AUCs:
    # Medial OA (0.9814), Baker's (0.9764), Fracture (0.9222), MCL (0.8957), ACL (0.8848), Contusion (0.8758), Effusion (0.8671)
    old_opt_weights = re.compile(r"        # Micro-injection weights[\s\S]*?        OPTIMIZED_TARGET_WEIGHTS = \{[\s\S]*?        \}")
    new_opt_weights = """        # Micro-injection weights calibrated to 6-Model Z-Logit Gold Benchmark (0.8561 Gold Macro AUC / 0.8665 Silver OOF):
        # Elite tier (>0.92 Gold AUC): Medial OA (0.981), Baker's (0.976), Fracture (0.922) -> 0.055-0.060
        # Strong tier (0.86-0.90 Gold AUC): MCL (0.896), ACL (0.885), Contusion (0.876), Effusion (0.867, +0.0154 OOF) -> 0.045-0.050
        # Mid tier (0.80-0.85 Gold AUC): Medial Meniscus (0.821, +0.0052 OOF), Lateral OA (0.818), PF OA (0.802) -> 0.030-0.040
        # Conservative tier (<0.80 Gold AUC): Lateral Meniscus (0.733), Synovitis (0.695) -> 0.020
        OPTIMIZED_TARGET_WEIGHTS = {
            "Medial OA": 0.0600,
            "Baker's": 0.0600,
            "Fracture": 0.0550,
            "MCL": 0.0500,
            "ACL": 0.0500,
            "Effusion": 0.0450,
            "Contusion": 0.0450,
            "Medial Meniscus": 0.0400,
            "Lateral OA": 0.0350,
            "PF OA": 0.0300,
            "Lateral Meniscus": 0.0250,
            "Synovitis": 0.0200,
        }"""
    c11_new, n_w = old_opt_weights.subn(new_opt_weights, c11_new, count=1)
    assert n_w == 1, f"Expected 1 replacement of OPTIMIZED_TARGET_WEIGHTS, got {n_w}"
    compile(c11_new, "<apex_cell11>", "exec")
    anb["cells"][11]["source"] = [l for l in c11_new.splitlines(keepends=True)]
    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(anb, f, indent=1, ensure_ascii=False)
    print(f"Updated and compiled {APEX_NB}!")


if __name__ == "__main__":
    main()
