# RSNA Knee Abnormality Detection — Plan

> **Current best:** 0.885 public LB (ConvNeXt-Small anatomical MoE + ResNet, 0.7/0.3 weighted ensemble with preprocessing parity fix)
> **Deadline:** 2026-10-22
> **Previous plan:** [`docs/archive/PLAN-2026-08-original.md`](docs/archive/PLAN-2026-08-original.md). It is kept for reference: its competition overview, data facts and gotchas still apply, but its architecture and timeline are out of date.

---

## 1. Goals and constraints

### Priority: place as high as possible
- **Main leaderboard:** macro AUC across the 12 targets. The efficiency track is a bonus if there's time.
- Set a concrete target (e.g. top X%) once you've checked your current rank with `scripts/find_rank.py`.

### Constraints this plan is built around
- **Solo, with a day job, kids and sports.** Time at the keyboard is short and broken up. GPU time is not the scarce resource; *your attention* is.
- **First time in a competition this hard.**
- **Everyone else has AI coding tools too.** Writing code fast is no longer an advantage.
- **Compute options (Kaggle + free external credits):** Kaggle provides 30h/week of free T4/P100 (9h per run limit). If we need to run something Kaggle "can't handle" for free — like high-res 320–512px slices, heavy multi-fold backbone sweeps, or parallel multi-GPU training — we have access to **free Nebius AI Cloud credits** and/or **AMD Cloud credits** to burst compute externally for free and upload the checkpoint weights into Kaggle Models for inference.

### Where a solo competitor can still win
Since everyone can generate code, the advantage comes from what AI tools don't do well by themselves:
1. **Correctness nobody checks.** The code that got 0.882 has a train/inference mismatch (section 2). Plausible code that's subtly wrong is common in AI-assisted entries. Catching these is a cheap gain.
2. **Trustworthy validation.** Many teams overfit the public LB and drop when the private LB is revealed. Choosing final submissions on solid evidence protects against that.
3. **Label quality.** Reading actual reports and disagreements is slow, human work that most teams skip.

**Learning is how you get there, not a separate goal.** You can only direct an AI tool well, and spot when it's confidently wrong, if you understand what the pipeline does. So:
- Every change gets a one-line hypothesis in `EXPERIMENTS.md` before it runs, and the result afterwards.
- Change one thing at a time when possible, so you know what moved the score.
- Each phase lists a few **questions you should be able to answer**. They're short on purpose.

### Value beyond the prize: sports medicine and med devices
The prize is one payoff. The other is lasting knowledge and a portfolio in sports medicine and medical-device AI. **Whenever a task can also teach real domain knowledge, take that opportunity.** In practice:
- **Choose the version of a task that teaches domain knowledge.** The Phase 3 label audit means reading real knee MRI reports: how radiologists describe ACL and meniscal tears, bone bruises, effusion. Per-target error analysis shows which injuries are hard to see on which plane, and why. That is sports-medicine imaging knowledge, not just Kaggle knowledge.
- **Keep a short `DOMAIN_NOTES.md`** for things learned along the way: anatomy, injury patterns, what each MRI plane and sequence shows, how reports are phrased across languages. A few lines at a time is enough.
- **Think like a med-device builder when evaluating.** Imaging AI products are judged on per-finding sensitivity/specificity, performance across sites and scanners, and failure modes — not one leaderboard number. Reporting results that way (per target, per site if metadata allows) is better science and the language that clinical and med-device people use.
- **The writeup is a portfolio piece.** A clear public writeup and code (not the data — the competition license forbids redistributing it) is what shows sports-medicine or med-device teams you can do this work.
- **After the deadline**, the parked `IDEAS.md` items with the most domain value (3D knee morphometrics, arthroscopy correlation) are good candidates to pursue for learning and portfolio reasons.

### Working rhythm (fits around a busy week)
- **Short session (15–30 min):** read the last run's log, fill in the `EXPERIMENTS.md` row, launch the next run. Every run is set up so it can go unattended overnight or during the workday.
- **Longer session (weekend):** anything that needs thinking — reviewing code changes, the label audit, choosing final submissions.
- **Claude does the code changes between sessions**, each one small, reviewed and tested, so your time goes to decisions rather than typing.

### Rules for the tradeoffs
- A change that raises the score for an unknown reason is kept, but flagged for an ablation before the final submission.
- `IDEAS.md` moonshots stay parked until after the deadline.
- Near the end: no new ideas, only confirming, selecting and writing up.
- If time runs short, drop phases from the bottom of the list. Phases 0–2 are the ones that matter.

---

## 2. Where things stand

**What exists:**
- **Labels:** 58 gold studies plus Jev report-derived ("silver") labels, embedded in the training notebook; the regex labeler is the fallback.
- **Cache:** a 256px cache with 16 slices per plane and a 0.12–0.88 slice band (`scripts/preprocessing/cache_fast_slices.py`).
- **Model:** `KneeAnatomicalMoEClassifier`, a shared backbone with gated attention pooling per plane and plane-specific heads (sagittal for ACL/menisci, coronal for MCL, axial for PF OA/effusion/synovitis, a joint head for the rest).
- **Training:** a single split. All 58 gold studies plus 10% of silver are used for validation. 6 epochs, no augmentation.
- **Submission:** reads raw DICOMs and runs the hardcoded 0.7/0.3 ensemble.

**Known issues (from the first review):**

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 1 | **Train/inference preprocessing mismatch for the ConvNeXt model** (fixed on the inference side, Phase 1a; the ResNet was trained without the cache and already matched). Slice order (InstanceNumber vs physical position), slice range (central 76% vs full), 2.5D neighbours (sampled slices ~1/16 volume apart vs truly adjacent), rescale slope/intercept (not applied vs applied) | `cache_fast_slices.py` vs `submission_notebook.py` `load_and_preprocess_series` | The model is scored on inputs that look different from what it trained on. Probably the biggest lever available. |
| 2 | Co-occurrence head exists only as a standalone test | `scripts/models/test_cooccurrence.py` | The 0.882 is being credited to the wrong change. |
| 3 | `load_state_dict(strict=False)` and a silent fallback to resnet34 | submission notebook | A broken checkpoint still produces a submission, with random weights. |
| 4 | Validation mixes gold and silver. Unscorable targets count as 0.5 in the average | `compute_competition_metric`, `run_training` | Checkpoint choice and experiment verdicts are driven by noise. |
| 5 | Ensemble weights fixed by backbone name | submission notebook | Guessed rather than fitted. |
| 6 | No augmentation, one fold, gold never used for training | training script | Leaves score behind. |

---

## 3. Phases, in order

### Phase 0 — Make measurements trustworthy
*Code only, no GPU. Claude does it; you review it in one session.*

1. Validation reports **gold AUC** (58 studies) and **silver AUC** (held-out Jev labels) separately. Targets with only one class are left out of the average (and the count printed) instead of being scored as 0.5.
2. Per-target positive counts in the gold set are printed, so you can see which gold AUCs rest on 1–3 positives.
3. Checkpoint selection uses silver AUC. It has far more studies, so it's less noisy; gold AUC is reported alongside it.
4. Validation predictions of the best epoch are saved to disk, which later phases need for fitting ensemble weights.
5. The submission loads checkpoints strictly and fails loudly on a mismatched or unknown backbone, instead of silently running random weights.
6. `EXPERIMENTS.md` is created and backfilled.

**Status:** code done (items 1–6). Remaining: one training run to produce the E02 baseline in `EXPERIMENTS.md`.

**Done when:** the next training log shows gold and silver AUC side by side.
**Questions you should be able to answer:** with only a handful of gold positives for a target, how far could its AUC swing by chance? Why can gold AUC and LB disagree?

### Phase 1 — Fix the preprocessing mismatch

**1a. Make inference match training (done in code; needs one submission, no retrain).**
The two models in the 0.882 ensemble were trained on *different* preprocessing:
- **ResNet34 (0.803 run):** trained before the cache existed, on raw DICOMs (`raw_v1`: physical-position order, full series, truly adjacent neighbours, rescale applied). The old submission code matched this.
- **ConvNeXt-Small (0.882 run):** trained on the cache (`cache_v1`: `InstanceNumber` order, 12–88% band, neighbours = adjacent *sampled* slices, no rescale, crop margin 4, last matching series per plane). The old submission code did **not** match this: the stronger model with 70% of the weight was being scored on unfamiliar input.

The submission now runs each checkpoint with its own pipeline. New checkpoints record theirs (`"preprocessing"` key); for the two existing ones it's inferred from the backbone and printed in the log.
- `tests/test_preprocessing_parity.py` builds synthetic DICOMs and checks that the submission reproduces both training pipelines exactly, including series selection. Run `python -m unittest tests/test_preprocessing_parity.py` after any preprocessing change.
- **Your step:** save and submit the submission notebook with the same checkpoints and log the result as E03. Only the ConvNeXt's input changed, so any LB change is due to the fix.

**1b. Better preprocessing for training and inference (optional experiment, later).**
Now that both sides match, improving the preprocessing itself is a normal experiment: physical-position ordering, truly adjacent neighbour slices (standard 2.5D), rescale applied. It needs a cache rebuild and a retrain, so it's one change to test against the Phase 2 baseline rather than a bug fix.

**Questions you should be able to answer:** what were the four differences, and why would each confuse the model? Why can `InstanceNumber` order differ from physical position? Why is the attention pooling not affected by slice order, while the 2.5D neighbour channels are? Why did the ensemble's two models need different preprocessing?

### Phase 2 — Standard gains: augmentation and folds
*Several unattended runs; about one short session each.*

1. Light augmentation, applied the same way to all slices of a plane: small shift/scale/rotate, brightness/contrast. **No horizontal flip** yet (flipping swaps medial and lateral anatomy, which affects the medial/lateral targets). One run with it versus one without, and log the result.
2. **K-fold** (3–5 depending on your remaining weekly Kaggle GPU quota), multilabel-stratified on the silver labels. Each fold is a separate run you can launch and leave.
3. Fit the ensemble weights on the saved validation predictions, replacing the 0.7/0.3 guess.
4. Retrain the final candidate with the gold studies included in training.

**Questions you should be able to answer:** how much does val AUC vary between folds, and was any earlier "improvement" smaller than that?

### Phase 3 — Label quality
*CPU work that runs while Phase 2 is training; a good weekend session.*

1. Per-target agreement between Jev labels and the 58 gold labels. Rank the targets from worst to best.
2. For the worst 2–3 targets, read ~10 disagreements and sort them by cause (negation, language, hedging, missed synonym).
3. Fix the top cause, re-measure against gold, retrain one fold, compare.

**Questions you should be able to answer:** which targets are limited by the labels, and which by the images?

### Phase 4 — Architecture experiments (only if time allows)
One at a time, on top of the Phase 2 baseline:
1. **Co-occurrence head:** wire it in behind a config flag and ablate. First check whether the targets actually co-occur in the silver labels.
2. **Resolution or slice count** (256→320 px, or 16→24 slices), weighed against runtime (can leverage Nebius/AMD free cloud credits if Kaggle T4 VRAM or 9h limit is exceeded).
3. **Second backbone** (EfficientNetV2-S is already supported), kept only if it improves the ensemble.

### Phase 5 — Efficiency submission (only if time allows)
Single best fold, fp16, sampled slices only. Measure runtime per study, compute the efficiency score for 2–3 variants on paper, submit the best.

### Phase 6 — Final selection and writeup
- Freeze the code a few days before the deadline.
- Choose final submissions on validation evidence, **not** just public LB: one "best evidence", one "best public LB" if they differ. Check the competition rules for how many you can select.
- Run any pending ablations for gains flagged "unknown cause", if time allows.
- A short writeup from `EXPERIMENTS.md`: what worked, what didn't, and why.

---

## 4. Order of work at a glance

| Step | Main work | Your time | GPU runs |
|---|---|---|---|
| 0 | Metrics, strict loading, experiment log | 1 review session | none |
| 1 | Inference matches training (done), one submission; optional better preprocessing later | 1 short session | none for 1a; cache rebuild + retrain for 1b |
| 2 | Augmentation, K-fold, fitted ensemble weights | 1 short session per run | aug ablation + 3–5 folds |
| 3 | Label audit and fixes (overlaps step 2) | 1 weekend session | 1 fold per fix |
| 4 | Co-occurrence, resolution, second backbone (optional) | 1 short session per run | 1 run per experiment |
| 5 | Efficiency variant (optional) | 1 short session | timing runs |
| 6 | Freeze, select, write up | 1 weekend session | final retrain only |

## 5. Parked
Everything in `IDEAS.md`: VLM report generation, 3D meshes, arthroscopy pretraining, DICOM metadata models. Good for learning, unlikely to pay off before the deadline. Revisit afterwards, starting with the ones that teach the most sports medicine (3D morphometrics, arthroscopy correlation).
