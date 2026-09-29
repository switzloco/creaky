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
- **E02 (Autopsy)**: Ran 6 epochs on Kaggle T4. Gold AUC climbed from 0.7283 → 0.8561. Discovered metric bug on continuous Silver labels (NaN) which prevented checkpoint saving. Bug permanently fixed in Phase 2 pipeline.

---

## Active Run — E04: Phase 2 Fold 0 Training (Augmentation + 5-Fold Stratification)

Kernel currently running on Kaggle: `nswitzer/training-book` (Version 5).

### What's Running:
- **Architecture**: ConvNeXt-Small Anatomical MoE.
- **Augmentation**: VolumeConsistentAugmenter (affine ±7°, translation ±5%, zoom 0.95–1.05x, contrast/brightness jitter; **horizontal flip strictly disabled**).
- **CV**: Fold 0 of 5-fold iterative multilabel stratification on Silver labels. Fixed 58 Gold reference validation.
- **Metric**: Binarized continuous Silver labels at $\ge 0.5$ + checkpoint fallback guardrail.
- **Outputs**: `best_model_fold_0.pt` and `val_preds_fold_0.csv`.

### How to Check & Fetch Results:
```cmd
uv run python scripts/fetch_kernel_output.py --kernel nswitzer/training-book --dir checkpoints/e04_training
```

### When E04 Finishes:
1. Fetch `best_model_fold_0.pt` and `val_preds_fold_0.csv`.
2. Extract final Gold AUC and Silver AUC from the log and update `EXPERIMENTS.md`.
3. Create Kaggle dataset `nswitzer/creaky-e04-checkpoint` or attach to submission notebook.
4. Review strategic options with Opus 5.5 as detailed in `HANDOFF_OPUS.md`.

