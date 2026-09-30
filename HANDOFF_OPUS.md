# Handoff Briefing for Opus 5.5

> **Date:** September 29, 2026  
> **Current Best Public LB:** 0.885 (ConvNeXt-Small MoE + ResNet-34 0.7/0.3 ensemble with `cache_v1` preprocessing parity)  
> **Active Kaggle Run:** Experiment E04 (`nswitzer/training-book`, Version 5) — Phase 2 Fold 0 Training  
> **Repository Branch:** `main` (clean, fully synced)

---

## 1. Executive Summary & Progress Since Last Handoff

### Milestone 1 — E03 Public LB Improvement (0.882 → 0.885)
- **Action:** Implemented strict preprocessing parity between training and inference. ConvNeXt received native `cache_v1` (InstanceNumber slice ordering, central 12–88% band, sampled-neighbor 2.5D channels) while ResNet-34 received `raw_v1`.
- **Result:** Public LB rose from **0.882 → 0.885** (+0.003) with zero retraining.
- **Key Takeaway:** Confirms hypothesis that visual input distribution alignment directly lifts leaderboard ranking.

### Milestone 2 — Phase 3 Domain & Label Quality Audit
- Audited regex labeler and clinical text against all 58 consensus Gold studies (`scripts/audit_gold_disagreements.py`).
- **Critical Domain Finding:** 30–60% of true anatomical abnormalities are selectively omitted by radiologists from text reports (e.g. Synovitis text coverage is only 22.4%, Medial OA text coverage is 69.6%). Full details recorded in `DOMAIN_NOTES.md`.

### Milestone 3 — Experiment E02 Autopsy (Gold AUC 0.8561 Progression)
- `training-book` completed all 6 epochs of ConvNeXt-Small MoE on Kaggle T4.
- **Model Learning Curve:** Gold validation macro AUC climbed steadily:
  - Epoch 1: `0.7283`
  - Epoch 2: `0.7719`
  - Epoch 3: `0.8035`
  - Epoch 4: `0.8319`
  - Epoch 5: `0.8427`
  - Epoch 6: `0.8561`
- **Metric Bug Discovered:** Silver AUC evaluated to `NaN` every epoch because `compute_competition_metric` had `mask = (yt == 0.0) | (yt == 1.0)`. In `train_labels.csv`, Jev output continuous soft probabilities (e.g. 0.02, 0.98), so almost all Silver validation studies were filtered out. Because checkpoint selection was configured to use Silver AUC, no `.pt` checkpoint was written to `/kaggle/working`.

### Milestone 4 — Bug Fix & Phase 2 Pipeline Implementation
1. **Metric Fix:** Binarized continuous targets at 0.5 threshold (`(yt >= 0.5).astype(int)`), while properly excluding unannotated exact 0.5 soft-unknowns. Tested and confirmed in `tests/test_phase2.py` (37 tests passing).
2. **Checkpoint Selection Guardrail:** Added automatic fallback to Gold AUC if Silver AUC is ever `NaN`.
3. **Volume-Consistent Augmentations:** Vectorized `TF.affine` across all $K=16$ slices in a plane ($\pm 7^\circ$ rotation, $\pm 5\%$ translation, $0.95–1.05\times$ zoom, contrast $\pm 10\%$, brightness $\pm 8\%$). **Horizontal flip is strictly disabled** to preserve medial vs. lateral knee anatomy.
4. **Multilabel Stratification:** 5-fold iterative stratification (`assign_multilabel_folds`) on Silver studies, preserving the 58 Gold studies as fixed reference validation.

---

## 2. Completed Run: Experiment E04 Results (Phase 2 Fold 0)

- **Kernel:** [`nswitzer/training-book` (Version 5)](https://www.kaggle.com/code/nswitzer/training-book) — **COMPLETE** (2h 56m runtime on Kaggle T4).
- **Architecture:** ConvNeXt-Small Anatomical MoE with `VolumeConsistentAugmenter` (horizontal flip strictly OFF) and 5-fold iterative multilabel stratification.
- **Results Across Epochs:**
  - Epoch 1: Train Loss `0.5581` | Gold AUC `0.6975` | Silver AUC `0.7470`
  - Epoch 2: Train Loss `0.5131` | Gold AUC `0.7785` | Silver AUC `0.8053`
  - Epoch 3: Train Loss `0.4742` | Gold AUC `0.8265` | Silver AUC `0.8324`
  - Epoch 4: Train Loss `0.4404` | Gold AUC `0.8509` | Silver AUC `0.8494`
  - Epoch 5: Train Loss `0.4076` | Gold AUC `0.8478` | **Silver AUC: 0.8541** $\rightarrow$ **SAVED BEST CHECKPOINT**
  - Epoch 6: Train Loss `0.3829` | **Gold AUC: 0.8581** | Silver AUC `0.8538`
- **Saved Artifacts Downloaded Locally:**
  - `checkpoints/e04_trained/best_model_fold_0.pt` (200.82 MB, records `"preprocessing": "cache_v1"`, best epoch 5, val AUC 0.8541)
  - `checkpoints/e04_trained/val_preds_fold_0.csv` (0.19 MB, 928 validation studies: 58 Gold + 870 Silver OOF)
- **Per-Class Silver AUC Breakdown (at best checkpoint):**
  - Baker's: `0.9090` | Medial OA: `0.8928` | Medial Meniscus: `0.8717` | Fracture: `0.8690`
  - ACL: `0.8629` | Lateral OA: `0.8617` | Effusion: `0.8574` | PF OA: `0.8523`
  - Synovitis: `0.8367` | Contusion: `0.8322` | MCL: `0.8294` | Lateral Meniscus: `0.7740`


---

## 3. Open Strategic Questions for Opus 5.5

When E04 results land, we would love your guidance on the following four decisions:

### Question 1: Ensemble Strategy with E04 Checkpoint
When `best_model_fold_0.pt` is generated:
- Should we build a 3-way ensemble blending our existing 0.885 models (ConvNeXt-S pre-Phase 2 + ResNet-34) with the new Phase 2 Fold 0 model?
- Or should we evaluate Fold 0 in isolation on the Public LB first to establish a clean ablation against 0.885?

### Question 2: Optimizing Head & Model Weights (Phase 2 Step 3)
E04 will output `val_preds_fold_0.csv` containing raw prediction probabilities alongside ground truth for all validation studies.
- Would you recommend fitting per-target weights using scipy `minimize` (Nelder-Mead / SLSQP optimizing macro-AUC) across the models, or sticking to global temperature / Platt scaling?
- How do we best guard against overfitting the ensemble weights on the validation slice?

### Question 3: Folds 1–4 vs. Backbone Diversity
Training Fold 0 takes ~2.5 hours on Kaggle T4.
- Path A: Complete all 5 folds of ConvNeXt-Small MoE (~12.5h GPU time total) for a complete 5-fold out-of-fold blended model.
- Path B: Train Fold 0 of a diverse second backbone (e.g. `efficientnet_v2_s` or `resnet50`) to maximize architectural diversity before training all folds.
- Which sequence gives the best risk-adjusted return on our weekly GPU quota?

### Question 4: Label Loss Asymmetry for High-Omission Targets
Our Gold audit showed that radiologist text reports omit true positive findings at high rates:
- Synovitis (77.6% omitted in text)
- Effusion (44.8% omitted in text)
- Meniscal tears (35% omitted in text)
Currently, `WeightedBCEWithLogitsLoss` treats false positives and false negatives symmetrically on Silver labels. Should we implement an asymmetric loss (e.g. Asymmetric Loss for Multi-Label Learning / focal discounting of negatives on high-omission targets) to stop the model from penalizing true image findings that text reports missed?
