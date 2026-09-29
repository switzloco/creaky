# Next Runs — Handoff

Steps for running the latest changes on Kaggle. Context is in `PLAN.md` (Phases 0 and 1a); results go in `EXPERIMENTS.md`.

The Kaggle notebooks in `notebooks/*/` are already regenerated from the scripts. If you edit `scripts/training/train_kaggle_notebook.py` or `scripts/submission/submission_notebook.py`, rebuild them first:

```
python scripts/build_notebooks.py
python scripts/sync_to_kaggle_folders.py
python -m unittest tests/test_preprocessing_parity.py   # needs numpy, pandas, pydicom, opencv
```

## Run 1 — E03: submission with the preprocessing fix (no training needed)

```
kaggle kernels push -p notebooks/submission-notebook
```

Same attached checkpoints as the 0.882 submission. Only the ConvNeXt's input preprocessing changed.

**Check in the log before submitting:**
- Each checkpoint prints a `Loaded model from: ... Backbone: ..., Preprocessing: ...` line. Expect `convnext_small → cache_v1` and `resnet34 → raw_v1`.
- If it fails with `Failed to load checkpoint` or `Could not build backbone`: that's the new strict loading catching an architecture mismatch (before, it would have silently run with random weights). Report which checkpoint failed; don't switch back to `strict=False`.
- `[OK] Predictions vary dynamically across test cases.`

Then submit to the competition and record the LB score as **E03** in `EXPERIMENTS.md` (compare with 0.882 in E01).

## Run 2 — E02: training with the new validation report (independent of Run 1)

```
kaggle kernels push -p notebooks/training-book
```

Same model config as before. What's new is the reporting:
- **Gold** and **Silver** AUC per epoch, with per-target positive/negative counts (targets with <5 gold positives are flagged as noisy).
- The best checkpoint is picked on silver AUC.
- Outputs are `best_model_fold_0.pt` (which now records its `preprocessing`) and `val_preds_fold_0.csv`.

Record the best epoch's gold and silver AUC as **E02** in `EXPERIMENTS.md`.

**Don't submit this checkpoint in the same run as E03**, so each score change has a single cause.
