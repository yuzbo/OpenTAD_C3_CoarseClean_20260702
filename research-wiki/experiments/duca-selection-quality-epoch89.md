---
type: experiment_diagnostic
node_id: exp:duca-selection-quality-epoch89
title: "DUCA coarse and selection quality diagnostic at legacy epoch 89"
status: tested
outcome: coarse_moderate_selection_not_supported
added: 2026-07-13
---

# DUCA coarse and selection quality diagnostic

## Scope and admissibility

This diagnostic evaluates the EMA `epoch_89.pth` checkpoint from legacy Job
`1159416` (`transition beta=0`). The corresponding detector run reached its
best Avg-mAP 64.34 at epoch 91, but an epoch-91 checkpoint was not retained.
The analyzed checkpoint and training protocol are commit
`8bfc0e549434591b9bf1a9cd5563deb0da388f92`.

This is not paper evidence. The checkpoint was trained with the invalidated
uniform-homotopy start documented in `duca-transition-only-fixed384.md`, and
the beta=0 configuration has no detector-gradient bridge. The result diagnoses
the realized coarse probe, transition scorer, structured decoder, and selected
positions; it neither validates nor refutes corrected commit `0ea4e15`.

## Provenance and protocol

- Remote analysis Job: `1161079`, completed with exit code 0 in 13m06s.
- Validation exposure: 211 videos, 487 sliding windows, 355,592 valid temporal
  observations.
- Checkpoint state: `state_dict_ema`, epoch 89.
- Checkpoint SHA-256:
  `8ea8e5c7a53ba159285ec244ecd34b7e79f86cec89acca965351f7a4bd869749`.
- GT is evaluation-only. No GT, teacher, raw detector prediction, or cache was
  passed to the selector. The detector backbone was not executed.
- Learned, exact-uniform, and one-per-stratum random policies use identical
  per-window valid length and budget. Exact uniform is
  `round(linspace(0,T-1,K))`.
- Confidence intervals use 2,000 bootstrap draws over 211 original-video
  clusters, rather than treating overlapping windows as independent.
- Representative plots use an automatic best/median/worst stratification by
  `uniform endpoint distance - learned endpoint distance`; no samples were
  manually selected.

## Coarse action-state quality

| Metric | Result |
|---|---:|
| pooled AUROC | 0.6214 |
| pooled AUPRC | 0.4111 |
| action prevalence | 0.3250 |
| AUPRC lift over prevalence | 1.265x |
| macro window AUROC | 0.6307 [0.5959, 0.6653] |
| macro window AUPRC | 0.4568 [0.4142, 0.5010] |
| Brier / ECE | 0.2401 / 0.1710 |
| balanced accuracy / F1 at 0.5 | 0.5866 / 0.4930 |

Action observations have mean/median p(action) 0.5268/0.5417, while background
observations have 0.4811/0.4940. The interquartile ranges overlap heavily:
action [0.4675, 0.5981], background [0.4051, 0.5666]. At threshold 0.5 the
model predicts action for 53.62% of observations although prevalence is
32.50%; TPR is 65.32%, TNR 52.01%, and precision 39.59%. The classifier is
therefore weak-to-moderate and substantially over-confident, not a clean
action/background state estimator.

Historical standalone official-ASFormer epoch 90 reached AP 0.4569 and AUROC
0.6494, with delta-p boundary support@1 0.8453. That run used 928 windows and a
different protocol, so it is context only, not a matched superiority result.

## Indirect transition localization

Important audit correction: the exporter stored
`transition_score = abs_delta_p_action + uncertainty_peak` under the key
`raw_transition_scores`. The following comparison is therefore against a
compound hand transition proxy, not pure `abs(delta p_action)`. Pure delta was
exported separately but was not analyzed in this run.

| GT boundary band | prevalence | learned AP / AUROC | compound proxy AP / AUROC |
|---|---:|---:|---:|
| r0 | 0.0051 | 0.0075 / 0.5775 | 0.0082 / 0.6079 |
| r1 | 0.0430 | 0.0619 / 0.5841 | 0.0654 / 0.6069 |
| r2 | 0.0791 | 0.1107 / 0.5785 | 0.1178 / 0.6070 |
| r4 | 0.1462 | 0.1916 / 0.5711 | 0.2019 / 0.5982 |
| r8 | 0.2569 | 0.3038 / 0.5542 | 0.3145 / 0.5770 |

The learned transition policy is above chance, but it is worse than the
`abs_delta + uncertainty_peak` compound proxy at every radius. Its utility curve is visibly noisy
and does not consistently peak at GT state transitions. This checkpoint did
not turn coarse state evidence into a stronger indirect boundary ranker.

## Selected-position quality

The mean selected count is 378.05 because short valid windows use
`K=min(384,T)`. Core macro-window results are:

| Policy | exact recall r0 | recall r1 | both endpoints r1 | endpoint distance | mean max hole | action enrichment |
|---|---:|---:|---:|---:|---:|---:|
| learned structured | 0.1568 | 0.8437 | 0.7108 | 0.6755 | 11.86 | 1.0436x |
| exact uniform | 0.1415 | 0.9991 | 0.9982 | 0.4800 | 1.83 | 0.9988x |
| stratified random | 0.1336 | 0.9082 | 0.8280 | 0.5481 | 1.86 | 0.9990x |
| utility top-k, diagnostic | 0.1570 | 0.8438 | 0.7110 | 0.6814 | 15.33 | 1.0445x |
| compound-proxy top-k, diagnostic | 0.1811 | 0.8377 | 0.7059 | 0.8559 | 28.91 | 0.9953x |

Learned minus exact-uniform paired effects are:

- exact r0 recall: +0.01535, 95% cluster CI [0.00497, 0.02544];
- r1 recall: -0.15546, CI [-0.17040, -0.13949];
- uniform-minus-learned endpoint distance: -0.19549 frames, CI
  [-0.23254, -0.15910], meaning learned is farther from boundaries;
- mean max-hole increase: +10.03 frames; observed learned maximum is 15.

The learned policy is better than uniform on endpoint distance in only 113/487
windows (23.2%), tied in 66 (13.6%), and worse in 308 (63.2%). Even in the top
coarse-AUROC quartile, mean gain remains -0.088 frames and only 34.4% of windows
improve. Per-window coarse AUROC versus selection gain has only Pearson
correlation 0.182; transition AP@r4 versus gain is 0.139. Coarse errors matter,
but they do not fully explain the selection failure.

The learned decoder overlaps unconstrained learned-utility top-k by 99.80% on
average and is exactly identical in 264/487 windows. Max-gap repair reduces the
worst top-k hole from 60 to 15, but changes too few positions to alter boundary
metrics materially. The dominant failure is the learned utility ranking and
the weak coverage/utility tradeoff, not an absent max-gap fail-closed check.

## Verdict

1. The joint coarse classifier is not well separated or calibrated enough to
   be called a strong state estimator.
2. Some transition signal exists, but the learned transition scorer degrades
   the audited compound hand proxy. Its relation to pure absolute actionness
   change remains unmeasured.
3. The selected positions do not satisfy the coverage-first research goal.
   A small exact-hit gain is bought with a large and statistically clear loss
   in radius-1 and both-endpoint coverage.
4. The structured decoder honors the hard max-hole bound, but at K near T/2 a
   max gap of 15 is far weaker than the natural one-to-two-frame spacing of
   exact uniform sampling.
5. Before any new full train, corrected-code diagnostics must separately gate
   coarse AUROC/AUPRC/ECE, separate pure-delta/compound/learned transition
   ranking, and
   learned-vs-uniform endpoint coverage. Detector mAP alone cannot certify that
   the selector learned the intended mechanism.

## Artifacts

Local root:
`E:/DeskTop/TAD/analysis_outputs/duca_selection_quality_20260713/output`.

- `val.export.json`: SHA-256
  `16B3A750010B2E6EA542EC1EDBDEC5006F4EAAC26DD1CBC0FE47D0D91A40BE27`
- `val.records.jsonl`: SHA-256
  `450C793FBA8F12BF3EDF3712C748CC6BEF8B02F7379335D7947E016311F5C450`
- corrected local `selection_quality_summary.json`: SHA-256
  `5D462129D7052EC76B19C8C72EE106A2DDC4F28DEE5EC8D053FC3C197A1D321C`
- vector overview/sample figures and their PNG renderings are under
  `val_analysis/`.

Analysis implementation was committed at
`1fc7037358e1141f7555ad87d1edd9128ce2e6a5`:
`tools/bata/export_duca_selection_quality.py`,
`tools/bata/analyze_duca_selection_quality.py`, and
`tests/test_duca_selection_quality_analysis.py`. Focused verification is
11 passed; py_compile and `git diff --check` pass.
