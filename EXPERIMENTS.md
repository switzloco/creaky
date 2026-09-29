# Experiment Log

One row per training run or submission. Write the **hypothesis before** launching the run; fill in the results after.

- **Gold AUC:** the 58 expert-labelled studies. Few positives on rare targets, so it's noisy; the training log flags targets with fewer than 5 positives.
- **Silver AUC:** the held-out 10% of Jev report-derived labels. Used to pick the best checkpoint (from E02 onward).
- **Fold spread:** the range of val AUC across folds, once K-fold exists (Phase 2). Differences smaller than this are noise.

| ID | Date | Hypothesis | Change (ideally one thing) | Gold AUC | Silver AUC | Fold spread | LB | Verdict | Lesson |
|---|---|---|---|---|---|---|---|---|---|
| E00 | 2026-09-28 | MoE heads + cached slices beat the earlier baseline | MoE routing, fast slice cache, ResNet backbone (several changes) | ? (mixed val only) | – | – | 0.803 | kept | – |
| E01 | 2026-09-28 | ConvNeXt-Small beats ResNet as the backbone | Backbone → ConvNeXt-S, batch 4→2 with 2-step accumulation, 0.7/0.3 ConvNeXt/ResNet ensemble | ? (mixed val only) | – | – | 0.882 | kept, **cause unclear** | Three changes at once. The commit message credits a co-occurrence head that isn't used anywhere. Needs an ablation (ConvNeXt alone vs the ensemble). Also: at inference the ConvNeXt got raw-DICOM preprocessing, not the cache preprocessing it was trained on (see E03). |
| E02 | | Fixing the metric changes nothing about the model; this run establishes the gold/silver baseline | Phase 0 metric + checkpoint selection on silver AUC | | | – | | | |
| E03 | | ConvNeXt scores better when its test input is preprocessed like its (cached) training input | Submission only: ConvNeXt now gets `cache_v1` preprocessing; ResNet unchanged (`raw_v1`). Same checkpoints and weights | – | – | – | | | |
