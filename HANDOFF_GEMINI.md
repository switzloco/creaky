# Handoff for Gemini 3.8: Oct 5 → Oct 22 Run Plan

> **Written by:** Opus 5.5, 2026-10-05
> **Current best:** 0.940 public LB (E10: 85% public CoAtNet + 15% our E04 ConvNeXt, rank blend)
> **Best single model:** E11, report-supervised ConvNeXt-S, fold 0, silver AUC 0.8571
> **Deadline:** 2026-10-22. Submissions: 4/day, ~30–40 left. GPU: Kaggle 30 h/week (T4), 9 h/run.

Read this whole file before doing anything. Then work through the tasks **in order**. Each task says when it's done and when to **STOP and ask the user**.

---

## 0. Strategy in one paragraph

The CoAtNet blend is our **floor**. Many teams will blend the same public models and land near 0.94, so blending more public models won't improve our rank. We win on signal other teams don't have: Jev labels, report supervision, and checking assumptions others take for granted. Our model has only 15% of the blend weight, so even a big improvement in our model moves the blend LB by **thousandths**. That's expected. Don't oversell small gains.

---

## 1. Hard rules (read every session)

1. **Shell:** `cmd.exe` only. PowerShell is disabled. Python only via `uv run python ...`. Never call `python`/`pip` directly. No global installs.
2. **Write the hypothesis first:** add the `EXPERIMENTS.md` row with Hypothesis + Change **before** launching a run. Fill in the results after.
3. **One change per experiment.** E01 and E09 changed several things at once, and we still don't know what helped or hurt. If you're tempted to change two things, make it two experiments.
4. **One Kaggle kernel per concurrent run.** Pushing a new version to a running kernel kills it (that's how E08 died). Use a new kernel slug per fold or experiment (e.g. `nswitzer/training-book-f1`).
5. **Rebuild after editing scripts.** The notebooks are generated. After editing `scripts/training/train_kaggle_notebook.py` or `scripts/submission/submission_notebook.py`:
   ```
   uv run python scripts/build_notebooks.py
   uv run python scripts/sync_to_kaggle_folders.py
   uv run python -m unittest tests/test_preprocessing_parity.py
   ```
6. **Explicit checkpoints:** every submission sets `CHECKPOINT_FILTER`. After every submission run, read the `Loaded model from:` lines in the log and write down exactly which checkpoints were used.
7. **Don't fit blend weights** on `val_preds_fold_*.csv` or on the 58 gold studies. Weights are coarse (e.g. 85/15, 80/20) and checked on the LB.
8. **Guard against LB overfitting:** keep a blend change only if it gains **> 0.002** LB. At most **3** hand-tuned blend-weight variants in total, ever.
9. **Noise floor:** the fold spread is ~0.009 silver AUC (E04 fold 0 0.854 vs fold 1 0.845), and there is no fixed seed. A difference smaller than ~0.005 silver AUC is **noise**. Write "inconclusive", not "improved".
10. **Report honestly.** Report exact numbers from logs. Never "extraordinarily strong" or "free boost". If a result is ambiguous, say so.
11. **Explain commands in caveman** whenever you ask the user for permission (user's rule).
12. **Keep the quiz current:** after each finished experiment that teaches something, add one question to `quiz.html` (same format as Q11–Q20; the total updates itself). Check it with:
    ```
    node "C:\Users\nswitzer\.gemini\antigravity\brain\55bbb1e3-8631-424a-b0c1-265d1c577dc4\scratch\check_quiz.js" quiz.html
    ```

---

## 2. Task list

### T1. Submit E12 and log it (no GPU)
- E12 = Version 3 of `nswitzer/creaky-0-94-ensemble` (CoAtNet 85% + E11 fold 0 15%). Submit it once the kernel run has finished.
- Also pull that notebook into the repo so blend changes are version-controlled:
  ```
  kaggle kernels pull nswitzer/creaky-0-94-ensemble -p notebooks/ensemble -m
  ```
- Log the LB score in `EXPERIMENTS.md` (row E12).
- **Decision:**
  - **E12 ≥ 0.940** → E11 settings are the base. Continue to T2.
  - **E12 < 0.940** → **STOP and ask the user** whether to use E04 settings as the base instead.

### T2. Launch E11 folds 1 and 2 (E13, E14), ~6 GPU-h total
- In `scripts/training/train_kaggle_notebook.py` `CONFIG`, change **only** `"fold"` (line ~62). All other settings stay as in E11.
- Two separate kernels, `training-book-f1` and `training-book-f2`. Copy `notebooks/training-book/kernel-metadata.json` and change `id` and `title`. Keep `kernel_sources: ["nswitzer/creaky-caching-dataset"]`, GPU on, internet on.
- When each run finishes:
  - download `best_model_fold_N.pt` and `val_preds_fold_N.csv` into `checkpoints/e1X_trained/`
  - upload the checkpoint as dataset `creaky-e11-reportaux-foldN`
  - log gold AUC, silver AUC and per-class silver AUC in `EXPERIMENTS.md`
- **Do not** start folds 3–4 yet. They wait for T4/T5.

### T3. N2 check: left/right knee mix (CPU, ~30 min; run while T2 trains)
**Question:** does the data contain both left and right knees? If yes, horizontal flip is anatomically valid. A flipped right knee is a valid left knee, because medial/lateral is defined by anatomy, not by image side.
- Write `scripts/inspect_laterality.py`. For a sample of ≥ 300 training studies (use `data/raw/train_series.csv` plus the local DICOMs available under `data/`), read headers only (`pydicom.dcmread(..., stop_before_pixels=True)`) and tally:
  - `Laterality`, `ImageLaterality`
  - `SeriesDescription` / `ProtocolName` containing L/R/left/right words in the dataset's languages: izq/der (ES), links/rechts (NL/DE), gauche/droit (FR), sol/sağ (TR), αριστερ/δεξ (EL), ляв/десн (BG)
  - for coronal series, the sign of the first component of `ImageOrientationPatient` (gives the patient-left direction on screen)
- Print a table of counts, plus how many studies could not be determined.
- If only a few local DICOMs are available, run the same script as a small CPU Kaggle kernel against the competition data.
- Write the result into `DOMAIN_NOTES.md` under a "Laterality" heading.
- **Decision:**
  - **Both sides present, each ≥ ~20% of studies** → N2 is a go: do T5.
  - **Almost all one side, or undeterminable** → skip T5 and log why.

### T4. N1: report supervision v2 with a multilingual sentence embedding
**Hypothesis:** E11's report target is char n-gram TF-IDF + SVD (`build_report_fingerprints`, line ~895), which captures surface spelling, not meaning ("no tear" ≈ "tear"). A multilingual sentence encoder captures negation and meaning across ES/NL/FR/TR/EL/BG/EN. A better target should give better image features.

**The single change:** where the 64-d report target comes from. Everything else, including dim 64, weight 0.5 and cosine loss, stays as in E11.

Steps:
1. **Offline embedding (local CPU).** Create `scripts/label_mining/embed_reports.py`:
   - Read `data/raw/train.csv`, column `Report`, keyed by `StudyInstanceUID`.
   - Encode with `intfloat/multilingual-e5-small` (sentence-transformers; prefix each text with `"passage: "`; `normalize_embeddings=True`).
   - Reduce 384 → **64** dims with PCA so the dimension matches E11 (that keeps the change to one thing), then L2-normalise.
   - Save `data/processed/report_embed_e5_64.csv` (columns: `StudyInstanceUID, e0..e63`).
   - Install into the project venv only: `uv pip install sentence-transformers` (ask the user first, with a caveman explanation).
   - Fitting the PCA on all reports is fine. It's unsupervised, and validation fingerprints are never used as targets (same as E11).
2. **Sanity check before any GPU time.** Pick 5 report pairs: "ACL tear" vs "no ACL tear" in the same language. Print their cosine similarity under the old TF-IDF fingerprint and under the new embedding. The new one should separate negated pairs better. **If it doesn't, STOP and report.**
3. Upload the CSV as Kaggle dataset `creaky-report-embed-e5`.
4. **Training code.** Add `CONFIG["report_source"]` with values `"tfidf"` (default, keeps E11 behaviour) and `"e5"`. With `"e5"`, load vectors from the attached dataset instead of calling `build_report_fingerprints`. Leave all other code paths untouched. Add the dataset to that kernel's `dataset_sources`.
5. **E15 run:** fold 0, `report_source: "e5"`, in kernel `training-book-n1`.
6. **Compare with E11 fold 0** (silver 0.8571, gold 0.8353):
   - silver ≥ 0.862 (≥ +0.005) → **win**
   - within ±0.005 → **inconclusive**
   - worse → **loss**
   - Also log the final cosine similarity (E11 reached 0.89).

### T5. N2: mirror-knee flip (only if T3 says go)
**The single change:** in `VolumeConsistentAugmenter` (line ~480), add an optional horizontal flip with p=0.5. Apply it identically to all slices and all three planes of a study, and do **not** swap any labels. Controlled by `CONFIG["hflip"]` (default `False`). Update the class docstring and the `"augment"` comment to explain *why* it's now allowed (anatomical laterality, T3 evidence). Don't just delete the "CRITICAL" warning.
- **E16 run:** fold 0, `report_source: "tfidf"`, `hflip: True`, in kernel `training-book-n2`. Compare with E11 fold 0 using the same thresholds as T4.
- Watch the per-class AUCs for **Medial vs Lateral** targets especially. If they drop while others rise, the anatomy argument failed: log it and keep flip off.
- Add a test in `tests/` checking that the flip is applied identically across slices and planes.

### T6. Decide the final configuration, then the remaining folds
- **STOP and show the user** a table: E11 fold 0 vs E15 (N1) vs E16 (N2), with silver, gold and per-class AUC.
- The user picks the final configuration:
  - **One idea wins** → train that configuration on folds 1–4 (the E11 folds 1–2 from T2 stay as the backup ensemble).
  - **Both win** → one more run combining both (fold 0) before committing to folds.
  - **Neither wins** → train E11 folds 3–4.
- Same procedure as T2: separate kernels, new dataset per fold.

### T7. Blend submissions (≤ 3 weight variants total)
1. **E-blend-A:** CoAtNet 85% + the equal mean of all folds of the final configuration 15%. Before submitting, time one full run against the competition's runtime limit (5 ConvNeXt-S checkpoints plus CoAtNet at 384).
2. **E-blend-B:** same, 80/20. A 5-fold average is more reliable than one fold, so it may deserve more weight.
3. Keep B only if it beats A by > 0.002.
4. **Per-class weights:** only if you find **published per-class CV AUC for the public CoAtNet** (check the `rsna-knee-speedy-raptors-coatnet-d4-0943` notebook and its discussion). Then at most 2–3 class groups, one submission. If that isn't available, skip it.

### T8. N3: lateral meniscus deep-dive (weekend session with the user)
Lateral Meniscus is our weakest target (E04 silver 0.774, E11 ~0.79). Prepare material for the user; don't draw conclusions alone:
- Use `data/processed/jev_vs_gold_audit.csv` and `scripts/label_mining/audit_jev_vs_gold.py`. List the gold studies where Jev disagrees on Lateral Meniscus, with report excerpts.
- For those studies, find which sagittal slice indices contain the lateral meniscus, and whether they fall **outside** the cache's 12–88% slice band (`cache_fast_slices.py`). The lateral meniscus sits off-center in the sagittal stack.
- Put the findings in `DOMAIN_NOTES.md`. **STOP and review with the user** before any code change.

### T9 (optional, only if GPU budget remains after T6). 384 px pilot
Follow "Step 3" in the Opus plan. **First** measure the native `Rows`/`Columns`/`PixelSpacing` distribution. If most series are ≤ 320 px natively, don't build the cache. Note that 256 px is already ~0.6 mm/pixel. The ~3 mm figure is slice **spacing**, which in-plane resolution doesn't change.

---

## 3. GPU budget

| Task | GPU-h | Week |
|---|---|---|
| T2 folds 1–2 | 6 | this week |
| T4 E15, T5 E16 | 6 | this week |
| T6 folds 1–4 of the final config | 12 | next week (quota resets) |
| T9 384 pilot (optional) | ~11 | only if budget remains |

TPU is a fallback only. Our code is CUDA/AMP PyTorch and would need a `torch_xla` port, so test the port before relying on it.

---

## 4. When to stop and ask the user (summary)

- E12 < 0.940 (T1)
- The negation sanity check fails (T4.2)
- Before installing any package
- Before choosing the final configuration (T6)
- Before any change from the N3 findings (T8)
- Anything that touches files outside `C:\Users\nswitzer\Antigrav Proj\creaky`
- Anything that would delete checkpoints or datasets
