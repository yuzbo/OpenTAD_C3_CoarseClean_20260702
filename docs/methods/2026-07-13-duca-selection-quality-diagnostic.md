# DUCA Selection-Quality Diagnostic

## Verdict

This is a diagnostic of the realized selector, not paper evidence. The joint
coarse action-state model is only moderately discriminative and poorly
calibrated. The learned transition scorer ranks GT boundary neighborhoods worse
than the raw actionness-change signal. At matched fixed budget, the learned
positions gain a small number of exact hits but lose substantial radius-1 and
two-endpoint coverage versus exact uniform sampling.

## Audited object

- Repository commit: `8bfc0e549434591b9bf1a9cd5563deb0da388f92`.
- Configuration:
  `configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py`.
- Training run: legacy beta=0 Job `1159416`.
- Checkpoint: EMA epoch 89, SHA-256
  `8ea8e5c7a53ba159285ec244ecd34b7e79f86cec89acca965351f7a4bd869749`.
- Best detector evaluation for this run was at epoch 91 with Avg-mAP 64.34;
  an epoch-91 checkpoint was not retained.
- Analysis Job `1161079`: completed with exit code 0 in 13m06s.
- Validation exposure: 211 videos, 487 overlapping windows, 355,592 valid
  temporal observations.
- GT is used only after selection for evaluation. No GT, teacher, raw detector
  prediction, or cache is passed to the selector. The detector backbone is not
  executed by the exporter.
- Confidence intervals use 2,000 bootstrap draws over original-video clusters.

The analyzed checkpoint was trained with the invalidated alpha=0 uniform
homotopy from commit `8bfc0e5`. It therefore cannot validate or refute the
corrected `0ea4e15` method. Beta=0 also has no detector-gradient bridge.

## Coarse action-state quality

| Metric | Value |
|---|---:|
| pooled AUROC | 0.6214 |
| pooled AUPRC | 0.4111 |
| action prevalence | 0.3250 |
| AUPRC lift over prevalence | 1.265x |
| macro-window AUROC, video-cluster CI | 0.6307 [0.5959, 0.6653] |
| macro-window AUPRC, video-cluster CI | 0.4568 [0.4142, 0.5010] |
| Brier / ECE | 0.2401 / 0.1710 |
| balanced accuracy / F1 at 0.5 | 0.5866 / 0.4930 |

Action observations have mean/median p(action) 0.5268/0.5417. Background
observations have 0.4811/0.4940. At threshold 0.5 the model predicts action on
53.62% of observations although true prevalence is 32.50%; TPR is 65.32%, TNR
is 52.01%, and precision is 39.59%.

A historical standalone official-ASFormer checkpoint reached AP 0.4569 and
AUROC 0.6494 at epoch 90. Its 928-window exposure is protocol-unmatched and is
background evidence only.

## Indirect transition ranking

| GT boundary radius | prevalence | learned AP / AUROC | raw actionness-change AP / AUROC |
|---|---:|---:|---:|
| r0 | 0.0051 | 0.0075 / 0.5775 | 0.0082 / 0.6079 |
| r1 | 0.0430 | 0.0619 / 0.5841 | 0.0654 / 0.6069 |
| r2 | 0.0791 | 0.1107 / 0.5785 | 0.1178 / 0.6070 |
| r4 | 0.1462 | 0.1916 / 0.5711 | 0.2019 / 0.5982 |
| r8 | 0.2569 | 0.3038 / 0.5542 | 0.3145 / 0.5770 |

The learned scorer is above chance but trails raw actionness change at every
radius. This checkpoint does not demonstrate that its learned transition head
adds useful indirect localization beyond `abs(delta p_action)`.

## Selected-position quality

Mean selected count is 378.05 because short windows use `K=min(384,T)`.

| Policy | exact recall r0 | recall r1 | both endpoints r1 | endpoint distance | mean max hole | action enrichment |
|---|---:|---:|---:|---:|---:|---:|
| learned structured | 0.1568 | 0.8437 | 0.7108 | 0.6755 | 11.86 | 1.0436x |
| exact uniform | 0.1415 | 0.9991 | 0.9982 | 0.4800 | 1.83 | 0.9988x |
| stratified random | 0.1336 | 0.9082 | 0.8280 | 0.5481 | 1.86 | 0.9990x |
| learned utility top-k, diagnostic | 0.1570 | 0.8438 | 0.7110 | 0.6814 | 15.33 | 1.0445x |
| raw-change top-k, diagnostic | 0.1811 | 0.8377 | 0.7059 | 0.8559 | 28.91 | 0.9953x |

Paired learned-minus-uniform effects:

- exact-r0 recall: +0.01535, 95% cluster CI [0.00497, 0.02544];
- radius-1 recall: -0.15546, CI [-0.17040, -0.13949];
- endpoint distance: +0.19549 frames, CI [0.15910, 0.23254];
- mean maximum-hole increase: +10.03 frames; observed learned maximum is 15.

The learned policy improves endpoint distance in 113/487 windows (23.2%), ties
in 66 (13.6%), and is worse in 308 (63.2%). Even its top coarse-AUROC quartile
has mean gain -0.088 frames and only 34.4% improved windows. Per-window coarse
AUROC versus selection gain has Pearson correlation 0.182; transition AP@r4
versus gain has correlation 0.139.

Learned decoded positions overlap unconstrained learned-utility top-k by 99.80%
on average and are exactly identical in 264/487 windows. Max-gap enforcement
reduces the worst unconstrained hole from 60 to 15, but changes too few
positions to alter boundary metrics materially.

## Result anchors that must not be conflated

| Anchor | Auditable meaning | Not established |
|---|---|---|
| N16R4 direct GT-boundary oracle, Job `1001959`, 76.67 | verified privileged-information upper bound; raw Slurm/stdout was recovered after this selector audit | uses train/val/test GT boundaries and a historical Adapter/ActionFormer protocol, not the current DUCA protocol |
| historical direct GT-boundary oracle, 77.62 | partial historical anchor; complete vector is recorded as 84.42 / 82.41 / 79.69 / 74.67 / 66.91 | original AutoDL stdout was not re-read in the 2026-07-13 provenance pass |
| Job `1150842`, 65.696 | historical grid-aware exact-uniform reference | not same commit/training protocol as current DUCA |
| Job `1150701`, 64.352 | historical native-stride exact-uniform reference | not same detector geometry as all current variants |
| Job `1159416`, 64.34 | legacy learned beta=0 diagnostic | homotopy start was invalid; no matched uniform conclusion |
| Job `1159417`, 63.55 | legacy learned beta=0.25 diagnostic | detector-gradient benefit; it trails beta=0 by 0.79 |
| historical separated lattice, 63.18 | older separated-training reference | not a matched joint-training comparison |

The N16R4 raw artifact was recovered after this selector audit. Job `1001959`
completed with exit code 0 and final Avg-mAP 76.67; its complete
mAP@0.3/0.4/0.5/0.6/0.7 vector is 83.63 / 81.54 / 78.92 / 73.42 / 65.83.
The sampler used `oracle_boundary_subsample`, K=384 from T=768, radius 2, and
train/validation/test GT segments. This verifies the mechanism-level statement
that direct GT-boundary allocation has a roughly 76--77 Avg-mAP upper bound in
the historical Adapter/ActionFormer protocol. It does not make the oracle,
uniform, and learned numbers comparable. Commit, split, detector geometry,
training exposure, checkpoint policy, evaluator, and inference-visible signals
still need normalization before any causal gap decomposition.

## Reproduction

Exporter:

```bash
python tools/bata/export_duca_selection_quality.py \
  --config <config.py> \
  --checkpoint <epoch_89.pth> \
  --output-jsonl <val.records.jsonl>
```

Analyzer:

```bash
python tools/bata/analyze_duca_selection_quality.py \
  --records-jsonl <val.records.jsonl> \
  --output-dir <analysis_dir> \
  --bootstrap-samples 2000 \
  --random-seed 20260713
```

Focused validation:

```bash
python -m pytest tests/test_duca_selection_quality_analysis.py -q
python -m py_compile \
  tools/bata/export_duca_selection_quality.py \
  tools/bata/analyze_duca_selection_quality.py \
  tests/test_duca_selection_quality_analysis.py
git diff --check
```

Raw JSONL records, CSV, and generated PDF/PNG figures are intentionally not
committed. Their local root and checksums are recorded in the coordination
research wiki.
