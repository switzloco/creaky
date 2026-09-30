# Next Runs — Handoff

Steps for running the latest changes on Kaggle. Context is in `PLAN.md` (Phases 0 and 1a); results go in `EXPERIMENTS.md`.

The Kaggle notebooks in `notebooks/*/` are already regenerated from the scripts. If you edit `scripts/training/train_kaggle_notebook.py` or `scripts/submission/submission_notebook.py`, rebuild them first:

```
python scripts/build_notebooks.py
python scripts/sync_to_kaggle_folders.py
python -m unittest tests/test_preprocessing_parity.py   # needs numpy, pandas, pydicom, opencv
```

# Next Runs — Execution & Monitoring

Context is in `PLAN.md` and `HANDOFF_OPUS.md`; results go in `EXPERIMENTS.md`.

---

## Completed Runs

- **E03 (Public LB: 0.885)**: Preprocessing parity fix verified on Kaggle. ConvNeXt received native `cache_v1` while ResNet received `raw_v1`. Score increased +0.003 without retraining.
- **E02 (Autopsy)**: Ran 6 epochs on Kaggle T4. Gold AUC climbed from 0.7283 → 0.8561. Metric bug resolved.
- **E04 (Phase 2 Fold 0)**: Finished in 2h 56m on Kaggle T4. Silver AUC reached **0.8541** at Ep 5; Gold AUC reached **0.8581** at Ep 6. Downloaded `best_model_fold_0.pt` (200.8MB) and `val_preds_fold_0.csv`.

---

## Immediate Next Actions (Assuming Opus Thumbs Up)

### 1. Upload E04 Checkpoint to Kaggle Models / Dataset
Upload `checkpoints/e04_trained/best_model_fold_0.pt` as a Kaggle dataset or model source so the submission kernel can access it.

### 2. Run E05: Leaderboard Submission
- **Option A (Ablation)**: Single-model Fold 0 submission to benchmark Phase 2 in isolation against 0.885.
- **Option B (3-Way Blend)**: Ensemble Phase 2 Fold 0 + pre-Phase 2 ConvNeXt-S (0.885 anchor) + pre-Phase 2 ResNet-34.

### 3. Fit Ensemble Weights on `val_preds_fold_0.csv`
Use `scipy.optimize` to fit per-model and per-target blending weights on the 928 validation studies instead of guessing.

### 4. Launch Next Training Run: Fold 1 or Backbone Diversity
- **Fold 1 Training**: Set `"fold": 1` in `CONFIG` in `scripts/training/train_kaggle_notebook.py`, rebuild notebooks, and push `training-book` to train Fold 1 (~2.5h).
- **Backbone Diversity**: Or set `"backbone": "efficientnet_v2_s"` for Fold 0 to produce a diverse architecture for the ensemble.


