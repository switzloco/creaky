# Next Runs — Handoff

Context is in `PLAN.md` and `HANDOFF_OPUS.md`; results go in `EXPERIMENTS.md`.

The Kaggle notebooks in `notebooks/*/` are regenerated from the scripts. If you edit `scripts/training/train_kaggle_notebook.py` or `scripts/submission/submission_notebook.py`, rebuild them first:

```
python scripts/build_notebooks.py
python scripts/sync_to_kaggle_folders.py
python -m unittest tests/test_preprocessing_parity.py   # needs numpy, pandas, pydicom, opencv
```

---

## Completed Runs

- **E03 (Public LB: 0.885)**: Preprocessing parity fix verified on Kaggle. ConvNeXt received native `cache_v1` while ResNet received `raw_v1`. Score increased +0.003 without retraining.
- **E02 (Autopsy)**: Ran 6 epochs on Kaggle T4. Gold AUC climbed from 0.7283 → 0.8561. Metric bug resolved.
- **E04 (Phase 2 Fold 0)**: Finished in 2h 56m on Kaggle T4. Silver AUC reached **0.8541** at Ep 5; Gold AUC reached **0.8581** at Ep 6. Downloaded `best_model_fold_0.pt` (200.8MB) and `val_preds_fold_0.csv`.

---

## Immediate Next Actions

Updated after the Opus review of the E04/E05 handoff. Do these in order; each one changes a single thing.

### 1. Check that E05 really was solo
The submission loads **every** `.pt` under `/kaggle/input`, and its `kernel-metadata.json` still attaches the `nswitzer/training-book` kernel output next to the E04 dataset. Open the E05 log and read the `Loaded model from: ...` lines.
- One line (the E04 dataset) → E05 is a clean solo result.
- More than one → E05 was an ensemble; log it as such in `EXPERIMENTS.md`.

From now on, set `CHECKPOINT_FILTER` at the top of `scripts/submission/submission_notebook.py` for every ablation (e.g. `["creaky-e04-convnext-fold0"]`). The log then prints which checkpoints were used and which were skipped, and stops if the filter matches nothing. Rebuild the notebooks after editing it (commands at the top of this file).

### 2. E07 — old ConvNeXt solo (submission only, no training)
E05 (one model) vs 0.885 (two-model ensemble) doesn't answer whether Phase 2 helped. Submit the **pre-Phase-2 ConvNeXt alone**, with its `cache_v1` preprocessing:
- Upload the old checkpoint (from the 0.882 run, e.g. `checkpoints/convnext_trained/best_model_fold_0.pt`) as its own dataset if it isn't one already, attach it, and set `CHECKPOINT_FILTER` to that dataset name.
- **E07 vs E05** = effect of Phase 2 (augmentation + folds), same architecture, both solo.
- **E07 vs 0.885 (E03)** = what the ResNet adds to the ensemble.

### 3. E06 (Fold 1) — let it finish, keep it
Download `best_model_fold_1.pt` and `val_preds_fold_1.csv` and log gold/silver AUC. It's a valid ensemble member whatever E05/E07 show.

**Do not use the old rule "if E05 < 0.885, halt fold training."** A single model scoring below a two-model ensemble is expected. Folds earn their keep when averaged together.

### 4. E08 — Fold 0 with 10 epochs instead of 6 (training; start after E06 finishes)
Gold AUC was still rising at epoch 6 in both E02 and E04, so the model is probably undertrained. The training script is already set to `fold: 0, epochs: 10`; nothing else changed, so E08 compares directly with E04 (gold 0.8581 / silver 0.8541 best).
- Expect ~5 hours on a T4.
- Upload its checkpoint under a **new** dataset name (the file is again `best_model_fold_0.pt`).
- There's no fixed random seed, so differences under ~0.005 in gold/silver AUC are noise.
- If E08 is clearly better: train folds 1–4 with 10 epochs. If not: keep 6 epochs and train folds 2–4.
- A second backbone (EfficientNetV2-S) comes after the folds.

### 5. Jev vs gold label audit (CPU, no GPU)
```
python scripts/label_mining/audit_jev_vs_gold.py
```
Needs `data/raw/train.csv` and `data/processed/jev_encoded_features.csv`. Per target, it prints how many gold positives Jev labels negative and how often a Jev negative is truly negative. Paste the table into `DOMAIN_NOTES.md`. This, not the regex audit, decides whether to down-weight Jev's negative labels for any target (the loss already takes per-target weights; no new loss function needed).

### Ensemble weights
Don't fit weights on `val_preds_fold_*.csv`:
- Only the new folds have validation predictions.
- The old models trained on a random 90% split that overlaps these validation folds, so their predictions there would be leaked.
- Only the 58 gold studies are clean, which is too few for per-target weights.

Average folds of the same model equally. Mix old and new models with coarse weights (e.g. 50/50) checked on the leaderboard.
