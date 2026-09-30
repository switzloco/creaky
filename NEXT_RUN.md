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

## Immediate Next Actions

### 1. Wait for E05 (Solo Submission) to Score
- Pushed E04 checkpoint to a Kaggle dataset (`nswitzer/creaky-e04-convnext-fold0`).
- Submitted E05 as a **solo** model run using only the E04 checkpoint to provide a clean ablation against our previous 0.885 LB score. 
- Wait for the LB score to finish computing.

### 2. Wait for E06 (Fold 1 Training) to Finish
- Launched Fold 1 training (`nswitzer/training-book`).
- Once finished, download `best_model_fold_1.pt` and `val_preds_fold_1.csv` and log results.
- **If E05 scores higher than 0.885**, proceed with training folds 2, 3, and 4 to complete the 5-fold ensemble.
- **If E05 scores lower than 0.885**, halt fold training and investigate why Phase 2 harmed LB performance (overfitting? augmentations too aggressive? metric disparity?).

### 3. Consider Ensemble Weight Fitting (Postponed)
- Fitting optimal weights requires predictions on the *same* validation set from all models in the ensemble.
- Since we do not have local predictions from the older models on the current validation folds, we cannot run this offline.
- Options for later: write a kernel to output predictions from all models on the validation set, or simply use equal weighting for a 5-fold ensemble of the same architecture.


