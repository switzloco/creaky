# RSNA Knee Abnormality Detection — Plan (Sep 29 → Oct 22)

> **Last updated:** 2026-09-29
> **Current best:** 0.882 public LB (ConvNeXt-Small anatomical MoE + ResNet, 0.7/0.3 weighted ensemble)
> **Deadline:** 2026-10-22 (~3.5 weeks)
> **Previous plan:** [`docs/archive/PLAN-2026-08-original.md`](docs/archive/PLAN-2026-08-original.md). It is kept for reference: its competition overview, data facts and gotchas still apply, but its architecture and timeline are out of date.

---

## 1. Goals

There are two goals, and the plan is built so that each step works toward both.

### Goal A: Place as high as possible
- **Main leaderboard:** macro AUC across the 12 targets.
- **Efficiency track:** runtime divided by squared normalized score gain (see the original plan, section 9).
- The final submissions are chosen on the evidence in the experiment log, not on public LB alone. The public LB is a sample of the test set and can mislead.

### Goal B: Understand the solution well
The end state: **you can explain every part of the final pipeline and back each claim with a number.** In practice:
- Every change is an **experiment with a written hypothesis**, recorded in `EXPERIMENTS.md` (format in section 4) *before* the result comes in.
- Change **one thing at a time** where the GPU budget allows, so every score movement has a known cause. (The 0.882 commit changed backbone, batch size and ensemble weighting at once and listed a co-occurrence head that was never wired in, so we can't say what caused the gain.)
- Each phase below ends with **"questions you should be able to answer"**. If you can't answer them, the phase isn't done, even if the score went up.
- The project ends with a **writeup** (a Kaggle writeup and/or README) that explains the solution, what worked, what didn't, and why.

### When the goals conflict
- A change that raises the score for an unknown reason is **kept, but flagged** for an ablation before the final submission.
- Moonshots from `IDEAS.md` are learning-only. They get time only after Phase 3 is done, or as a clearly bounded side project.
- In the last 5 days (Oct 17–22), Goal A takes priority: no new ideas, only confirming, selecting and writing up.

---

## 2. Where things stand

**What exists:**
- **Labels:** 58 gold studies plus Jev report-derived ("silver") labels, embedded in the training notebook; the regex labeler is the fallback.
- **Cache:** a 256px cache with 16 slices per plane and a 0.12–0.88 slice band (`scripts/preprocessing/cache_fast_slices.py`).
- **Model:** `KneeAnatomicalMoEClassifier`, a shared backbone with gated attention pooling per plane and plane-specific heads (sagittal for ACL/menisci, coronal for MCL, axial for PF OA/effusion/synovitis, a joint head for the rest).
- **Training:** a single split. All 58 gold studies plus 10% of silver are used for validation. 6 epochs, no augmentation.
- **Submission:** reads raw DICOMs and runs the hardcoded 0.7/0.3 ensemble.

**Known issues (from the Sep 29 review):**

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 1 | **Train/inference preprocessing mismatch.** Slice order (InstanceNumber vs physical position), slice range (central 76% vs full), 2.5D neighbours (sampled slices ~1/16 volume apart vs truly adjacent), rescale slope/intercept (not applied vs applied) | `cache_fast_slices.py` vs `submission_notebook.py:290-330` | The model is scored on inputs that look different from what it trained on. Probably the biggest lever available. |
| 2 | Co-occurrence head exists only as a standalone test | `scripts/models/test_cooccurrence.py` | The 0.882 is being credited to the wrong change. |
| 3 | `load_state_dict(strict=False)` and a silent fallback to resnet34 | `submission_notebook.py:597`, `:403` | A broken checkpoint still produces a submission, with random weights. |
| 4 | Validation mixes gold and silver. Unscorable targets count as 0.5 in the average | `train_kaggle_notebook.py:724-748, 761-766` | Checkpoint choice and experiment verdicts are driven by noise. |
| 5 | Ensemble weights fixed by backbone name | `submission_notebook.py:686` | Guessed rather than fitted. |
| 6 | No augmentation, one fold, gold never used for training | training script | Leaves score behind. |

---

## 3. Phases

Each phase lists **Do**, **Done when**, **Win** (what it does for Goal A), **Learn** (what it does for Goal B), and the questions you should be able to answer at the end.

### Phase 0 — Make measurements trustworthy (Sep 29 – Oct 1)
Nothing learned after this point means much unless this is done first.

**Do**
1. Split the validation metric into **gold AUC** (58 studies) and **silver AUC** (held-out Jev labels), reported separately each epoch. Drop targets that have fewer than 2 classes from the mean and print how many were dropped, instead of scoring them as 0.5.
2. Print per-target positive counts in the gold set, so you know which gold AUCs are based on 1–3 positives and aren't trustworthy.
3. Submission: `strict=True` loading (or fail loudly on missing/unexpected keys), and no silent backbone fallback.
4. Create `EXPERIMENTS.md` and backfill the rows you can reconstruct (0.803 run, 0.882 run).
5. Save out-of-fold (OOF) predictions to disk with every training run, which later phases need.

**Done when:** a training log shows gold and silver AUC side by side with the per-target counts, and a deliberately broken checkpoint makes the submission fail.

**Win:** the scores you'll base decisions on become ones you can trust. **Learn:** how noisy a 58-sample AUC is.

**Questions you should be able to answer:**
- With *n* gold positives for Fracture, roughly how much could its AUC swing by chance?
- Why can gold AUC and LB disagree?
- Why is AUC unaffected by calibration, and when does that stop being true (averaging across models)?

### Phase 1 — Fix the preprocessing mismatch (Oct 1 – Oct 5)

**Do**
1. Move slice selection and preprocessing into **one function** that the cache builder, the training fallback and the submission all call: physical-position ordering, one banding rule, rescale applied, 2.5D neighbours defined the same way everywhere.
2. Choose the 2.5D neighbour definition on purpose. Either **adjacent slices** (cache stores idx±1 for each sampled slice, 48 slices per plane) or **sampled neighbours** (what training effectively does now). Adjacent is closer to the usual 2.5D setup; test it if the cache size allows.
3. Add a **parity test**: run one training study through the cache path and the inference path and assert the tensors match.
4. Rebuild the cache, retrain the current ConvNeXt config **with nothing else changed**, and submit.
5. Side benefit for the efficiency track: inference then reads 16 (or 48) DICOMs per plane instead of all of them.

**Done when:** the parity test passes and the new LB number is recorded in `EXPERIMENTS.md` next to 0.882.

**Win:** possibly the largest single gain available. **Learn:** a controlled experiment on how much train/test skew costs.

**Questions you should be able to answer:**
- Which of the four differences mattered most? (Ablate one, if the GPU budget allows.)
- Why does sorting by `InstanceNumber` sometimes disagree with physical position?
- What does the model "see" in the neighbour channels?

### Phase 2 — Standard gains: augmentation and folds (Oct 5 – Oct 12)

**Do**
1. Add light augmentation, applied the same way to all 16 slices of a plane: small shift/scale/rotate, brightness/contrast. **No horizontal flip** until laterality handling is understood (flipping swaps medial and lateral anatomy in coronal/axial views, which affects the medial/lateral targets).
2. One run with augmentation versus Phase 1's run without it, same split, and log the result.
3. Move to **K-fold** (3–5 depending on GPU hours; multilabel stratified on the silver labels). Check your remaining weekly Kaggle GPU quota before committing to 5.
4. Fit the ensemble weights on OOF predictions (per target if they're stable, otherwise global), replacing the 0.7/0.3 guess.
5. Final-candidate retrain includes the gold studies in training (use OOF or a silver hold-out for monitoring).

**Done when:** a fold ensemble is submitted, and the ensemble weights come from OOF, not a guess.

**Win:** augmentation and fold ensembling are the most reliable gains in Kaggle vision competitions. **Learn:** variance between folds, which is the error bar on every other experiment.

**Questions you should be able to answer:**
- How much does val AUC vary between folds? Is the gap between any two past experiments larger than that?
- Why would a horizontal flip hurt Medial OA vs Lateral OA?
- What does OOF mean, and why is fitting ensemble weights on it valid?

### Phase 3 — Label quality (Oct 8 – Oct 15, overlaps Phase 2's GPU runs)
The original plan called labels "the biggest differentiator". This is CPU/LLM work, so it runs while Phase 2 is training.

**Do**
1. Per-target agreement between Jev labels and the 58 gold labels: sensitivity, specificity, kappa. Rank the targets by how bad they are.
2. For the worst 2–3 targets, read 10 disagreements by hand and sort them into causes (negation, language, hedging, missed synonym).
3. Fix the top cause, re-measure against gold, retrain **one** fold, and compare.
4. Try downweighting silver labels where Jev is uncertain (the loss already takes per-target weights), so the model trusts them less.

**Done when:** you have a table of label agreement per target, and at least one label fix was tested end to end.

**Win:** better labels for the targets where they're worst. **Learn:** how label noise turns into model error — the core lesson of weak supervision.

**Questions you should be able to answer:**
- Which targets are limited by the labels, and which by the images?
- If silver labels are 85% accurate on a target, what's the rough ceiling on model AUC for that target?

### Phase 4 — Architecture experiments (Oct 12 – Oct 17)
These run only on top of the Phase 2 baseline, one change at a time.

**Candidates, in priority order:**
1. **Co-occurrence head** (`test_cooccurrence.py`): wire it in behind a config flag and ablate. First check the co-occurrence of positives in the silver labels; if the targets are mostly independent, expect little gain.
2. **Resolution / slice count:** 256→320 px, or 16→24 slices, weighed against the time budget.
3. **Second backbone for diversity** (EfficientNetV2-S already supported). It's worth keeping only if it raises the OOF ensemble score, not just its own score.

**Done when:** each experiment you ran has an `EXPERIMENTS.md` row with a verdict. Negative results count.

**Learn:** whether cross-target reasoning helps when the backbone already sees all planes. **Questions you should be able to answer:** Why would a diverse weak model improve an ensemble more than a strong but similar one?

### Phase 5 — Efficiency submission (Oct 15 – Oct 19)

**Do**
- Single best fold, fp16, reading only the sampled slices (from Phase 1), optionally `torch.compile`.
- Measure runtime per study and project the total over the test set.
- Compute the efficiency score formula for 2–3 variants and pick the best on paper before submitting.

**Learn:** the tradeoff between accuracy and speed. Because the score term is squared, a small AUC loss can outweigh a large speedup. Work out where the break-even point is.

### Phase 6 — Final selection and writeup (Oct 19 – Oct 22)

**Do**
- Freeze the code on Oct 19. No new ideas after that.
- Choose the final submissions from OOF and gold evidence, **not** just public LB: one "best evidence" submission and one "best public LB", if they differ. Check the competition rules for how many final submissions you can select.
- Run the ablation for any gain still flagged "unknown cause" (see section 1), if time allows.
- Write the writeup from `EXPERIMENTS.md`: pipeline diagram, what worked, what didn't, and what you'd do next.

**Questions you should be able to answer:** everything above, in writing.

---

## 4. Experiment log format (`EXPERIMENTS.md`)

One row per training run or submission. Fill in the hypothesis **before** the run.

| ID | Date | Hypothesis | Change (one thing) | Gold AUC | Silver AUC | Fold spread | LB | Verdict | Lesson |
|---|---|---|---|---|---|---|---|---|---|
| E01 | 09-28 | ConvNeXt-S > ResNet backbone | backbone, bs, weighting (confounded) | ? | ? | – | 0.882 | kept, cause unclear | commit message credited a head that wasn't used |

---

## 5. Calendar

| Dates | Main work | GPU runs |
|---|---|---|
| Sep 29 – Oct 1 | Phase 0: metrics, strict loading, log | none (code only) |
| Oct 1 – Oct 5 | Phase 1: shared preprocessing, parity test | cache rebuild, 1 retrain, 1 submit |
| Oct 5 – Oct 12 | Phase 2: augmentation, K-fold, OOF weights | aug ablation, 3–5 folds |
| Oct 8 – Oct 15 | Phase 3: label audit and fixes (CPU) | 1 fold per label fix |
| Oct 12 – Oct 17 | Phase 4: co-occurrence, resolution, diversity | 1 run per experiment |
| Oct 15 – Oct 19 | Phase 5: efficiency variant | timing runs |
| Oct 19 – Oct 22 | Phase 6: freeze, select, write up | final retrain only |

## 6. Parked (not this month unless Phase 3 is done early)
Everything in `IDEAS.md`: VLM report generation, 3D meshes, arthroscopy pretraining, DICOM metadata models. These are good for learning but unlikely to pay off before Oct 22. Revisit after the deadline, or take one on as a bounded side project if it's the thing you most want to learn.
