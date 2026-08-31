# ActionFormer SparseHead DCSR G1 Negative-Result Analysis

Date: 2026-07-30

Status: exact G1 rejection is `empirically_supported`; current route terminated

Paper status: `paper_ready=false`

## Executive verdict

DCSR G1 failed its preregistered internal non-inferiority gate on every seed and
every tIoU threshold. Mean DCSR-minus-dense Avg-mAP is `-7.556202 pp`, while
mAP@0.6 and mAP@0.7 fall `-11.043134/-11.019821 pp`. The route therefore stops
at G1: no G2--G4, no five-seed official test study and no efficiency claim.

The failure is not best explained by random variation, selected-only
supervision or a dead zero-initialized residual. The strongest observed
explanation is that the one-layer cheap scaffold plus the learned residual
decomposition does not reproduce the official three-layer dense head. Fixed
K384 residual support adds a smaller, high-IoU-sensitive penalty.

This conclusion is narrow. It rejects this exact one-layer-scaffold,
three-layer-residual, uniform-K384 G1 contract. It does not prove that all
sparse heads, all conditional residual computation or all proposal-preserving
designs fail.

## Evidence identity and boundary

- Training source commit/tree:
  `bf0df83d7400c89fc61f38d169d68085420a2263` /
  `2f9346fcfd2bfb7fc5a76a86ef65545030a67469`.
- Frozen 160/40 official-validation manifest SHA-256:
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`.
- Formal G1 aggregate SHA-256:
  `b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
- Diagnostic source commit/tree:
  `8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b` /
  `1ac5a68c6b8d0b1c9028ea3154765ae20e87622a`.
- Diagnostic completion/prediction/checkpoint SHA-256:
  `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53` /
  `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36` /
  `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.
- Diagnostics read 40 validation-holdout videos and 611 GT instances. They use
  no test subset, perform no training and are post-hoc/counterfactual.

These are valid internal method-selection and mechanism-diagnostic results.
They are not official THUMOS test results and are forbidden from paper
performance tables. In particular, their absolute values must not be compared
with historical `63.xx`, released `66.833392`, or official S0 `66.583013`.
Only paired within-holdout deltas are causal for G1.

## Formal G1 raw results

Each row is terminal epoch-35 EMA, paired by seed. Values after Avg are
mAP@0.3/0.4/0.5/0.6/0.7.

| seed | arm | Avg | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026073001 | dense | 0.5769068582 | 0.7454479392 | 0.6703238925 | 0.6051488293 | 0.5176847285 | 0.3459289013 |
| 2026073001 | DCSR K384 | 0.5032135335 | 0.6797973676 | 0.6419292664 | 0.5495437746 | 0.3998147540 | 0.2449825048 |
| 2026073002 | dense | 0.5619997727 | 0.7067480229 | 0.6607374491 | 0.5967370890 | 0.4976686917 | 0.3481076106 |
| 2026073002 | DCSR K384 | 0.4881934096 | 0.6624068015 | 0.6337282212 | 0.5320643209 | 0.3792375708 | 0.2335301335 |
| 2026073003 | dense | 0.5653126303 | 0.7208504949 | 0.6639986345 | 0.5905823946 | 0.4930057634 | 0.3581258642 |
| 2026073003 | DCSR K384 | 0.4861262566 | 0.6592440185 | 0.6141737120 | 0.5161456200 | 0.3980128293 | 0.2430551031 |

Three-seed aggregate:

| arm | Avg | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.5680730871 | 0.7243488190 | 0.6650199920 | 0.5974894377 | 0.5027863946 | 0.3507207920 |
| DCSR K384 | 0.4925110665 | 0.6671493959 | 0.6299437332 | 0.5325845718 | 0.3923550514 | 0.2405225804 |
| delta, pp | -7.556202 | -5.719942 | -3.507626 | -6.490487 | -11.043134 | -11.019821 |

Avg-delta sample standard deviation is `0.3139 pp`; every seed and threshold is
negative. Seed uncertainty is therefore not a plausible primary explanation.

## Counterfactual decomposition

The same frozen checkpoints and evaluator were replayed without training:

| inference arm | Avg | delta vs dense | @0.6 | @0.7 |
|---|---:|---:|---:|---:|
| dense official head | 0.568073 | 0.000000 pp | 0.502786 | 0.350721 |
| one-layer scaffold only | 0.493892 | -7.418076 pp | 0.396587 | 0.237621 |
| scaffold + residual on all valid queries | 0.504906 | -6.316665 pp | 0.416457 | 0.264033 |
| scaffold + residual on K384 | 0.492511 | -7.556202 pp | 0.392355 | 0.240523 |

Key contrasts:

- all-query residual value over scaffold:
  `+1.101411 pp` Avg and `+1.9870/+2.6412 pp` at 0.6/0.7;
- trained K384 residual value over scaffold:
  `-0.138126 pp` Avg and `-0.4232/+0.2901 pp` at 0.6/0.7;
- K384 support penalty relative to all-query residual:
  `-1.239537 pp` Avg and `-2.4102/-2.3511 pp` at 0.6/0.7.

Thus most of the G1 gap already exists in the cheap scaffold/decomposition.
Sparse residual support is real but secondary, with disproportionate damage at
high IoU. Full-grid targets and the official positive normalizer were retained,
so selected-only supervision is not the primary cause in G1.

## Localization, recall, duration, class and score evidence

At class-aware recall@200 and tIoU 0.7, dense/all-query/K384/scaffold are
`62.25/53.03/49.86/49.70%`. Class-agnostic values are
`68.19/61.92/62.41/62.25%`. The much larger class-aware gap for K384 indicates
that label/score ranking contributes in addition to geometry.

Mean best same-label tIoU is `0.6897` for dense, `0.6500` for all-query,
`0.6470` for K384 and `0.6465` for scaffold. Median start/end error normalized
by GT duration is `0.1371/0.1156` for dense versus
`0.1803/0.1549` for K384. Localization quality is therefore degraded before
the strict AP thresholds, not merely by one NMS threshold.

Class-aware recall@200 at tIoU 0.7 by duration:

| duration | dense | all-query | K384 | scaffold |
|---|---:|---:|---:|---:|
| <2 s | 32.38% | 27.78% | 29.68% | 29.21% |
| 2--4 s | 71.37% | 57.44% | 49.91% | 49.34% |
| 4--8 s | 85.23% | 71.94% | 66.24% | 66.24% |
| 8--16 s | 80.61% | 80.61% | 79.39% | 81.21% |
| 16--32 s | 63.64% | 54.55% | 51.52% | 51.52% |

The dominant losses occur in 2--8 second and 16--32 second actions; the
40-video holdout has no >=32 second GT, so that stratum is unidentified.

Worst mean K384-minus-dense class AP deltas include:

- @0.6: SoccerPenalty `-33.29 pp`, Shotput `-31.43 pp`,
  BaseballPitch `-26.03 pp`, Billiards `-22.09 pp`;
- @0.7: SoccerPenalty `-36.51 pp`, LongJump `-29.80 pp`,
  Diving `-25.36 pp`, HammerThrow `-22.26 pp`.

Some classes improve at @0.7, so this is not a universal class-wise theorem:
BaseballPitch `+7.25 pp`, CricketShot `+5.05 pp`, CricketBowling `+1.69 pp`.

Scores are compressed: dense/K384 mean scores are `0.0885/0.0803`, p95 is
`0.3086/0.2299`, and dense has about 35 predictions per seed in score
`[0.6,0.8)` versus one for K384. These are descriptive TP-rate bins, not a
probability-calibration metric. Post-NMS outputs cannot identify suppressed
pre-NMS proposals or prove NMS causality.

## Checkpoint dynamics

The zero-initialized final residual projections are not dead. Across seeds,
classification-final L2 is already `0.2520` at epoch 5 and `0.4069` at epoch
35; regression-final L2 is `0.0545` and `0.1278`. Exact-zero fractions are
approximately zero. Residual hidden and scaffold updates shrink toward epoch
35, consistent with a settled optimization trajectory.

This weakens the explanation “zero initialization prevented residual learning
for the whole run.” It does not prove that the learned gradients were useful,
that the objective was aligned, or that the optimizer found the best
decomposition. Checkpoints are five epochs apart and gradient norms were not
logged; first-step dynamics and optimization causality remain unidentified.

## Competing explanations

### 1. Weak scaffold / mismatched decomposition — leading

Support:

- scaffold-only is already `-7.4181 pp` below dense;
- all-query residual restores only `+1.1014 pp` and remains `-6.3167 pp`;
- high-IoU boundary and class-aware recall degrade even without K384 support.

Counterevidence:

- all-query residual does recover some high-IoU AP, so the residual is not
  wholly useless;
- current diagnostics cannot separate scaffold capacity from joint-training
  objective/optimization.

Falsifiable prediction:

- under a new preregistered pilot, a dense-capacity-matched scaffold or an
  official dense proposal floor should recover most of the `-6.32 pp`
  all-query gap before sparsity is introduced.

Minimal decisive experiment:

- new validation-only 2x2: one-layer versus official-depth scaffold crossed
  with residual disabled versus all-query residual, paired from scratch with
  first-step gradient logs. This is not authorized under the terminated G1.

### 2. Residual support bottleneck — secondary

Support:

- K384 loses `1.2395 pp` to all-query residual;
- the penalty doubles at high IoU to roughly `2.4 pp`;
- K384 class-aware recall@0.7 is `49.86%` versus `53.03%` all-query.

Counterevidence:

- support explains only a minority of the `-7.56 pp` total gap;
- K384 residual is nearly neutral relative to scaffold, so residual utility
  and support cannot be cleanly separated after joint training.

Falsifiable prediction:

- a preregistered support-density sweep should recover high-IoU AP
  monotonically before reaching all-query execution if K384 coverage is causal.

Minimal decisive experiment:

- inference-only residual support sweep on a future independently trained
  checkpoint, with pre-NMS decoder-input recall captured before threshold/NMS.

### 3. Optimization / zero-init transient — plausible but not primary

Support:

- final residual projections begin as no-ops, so hidden residual layers receive
  zero gradient through them on the first update;
- learned K384 residual adds almost no Avg-mAP over scaffold.

Counterevidence:

- final projections are clearly nonzero by epoch 5 and continue updating;
- 35 epochs and shrinking late updates argue against a branch that stayed dead.

Falsifiable prediction:

- first-step gradient logging would show a one-step hidden delay but subsequent
  nonzero hidden gradients; an init-only change should not close a 7.56 pp gap
  if scaffold capacity is dominant.

Minimal decisive experiment:

- a new preregistered short optimization probe with per-group gradients,
  activation scales and residual-to-scaffold output ratios. Current norms alone
  cannot authorize a causal optimization claim.

### 4. Seed or evaluator noise — unlikely primary

Support:

- the holdout contains only 40 videos, so absolute generalization remains
  uncertain.

Counterevidence:

- paired Avg deltas are negative for all seeds with `0.3139 pp` SD;
- G0 exact equivalence, independent recomputation, raw-artifact receipts and
  diagnostic validation all pass.

Prediction:

- additional seeds might alter magnitude but are unlikely to move the mean
  across the frozen `-0.5 pp` gate.

Decision:

- more seeds are not justified after the preregistered kill and would not turn
  an internal result into an official one.

## Consistency with prior evidence

There is no contradiction with the earlier PhysTime decode-cross gain.
PhysTime changed how already retained frozen proposals map to physical time.
Hard K384 and DCSR alter head representation or residual support before/at
proposal production. A decoder can improve surviving proposals but cannot
recreate missing or poorly represented boundaries.

There is also no contradiction between S0 and G1. S0 hard deletion loses
`22.6633 pp` on an official same-run test screen. Preserving a dense scaffold
recovers much of that catastrophic loss, but the cheap one-layer scaffold
still fails the paired internal non-inferiority test by `7.5562 pp`.

## Route decision and next plan

The current DCSR/SparseHead route is terminated exactly as preregistered.
No silent threshold, depth, budget, selector, NMS, checkpoint or seed change is
allowed, and no G2--G4 or official five-seed/cost run will be launched.

If sparse conditional computation is revisited, it must be a separately named
and preregistered route. The strongest scientifically motivated successor is
not another selector over a weak scaffold; it is an official-quality dense
proposal floor with selectively gated residual computation, first tested for
representation equivalence before any efficiency claim. That successor remains
`discussed`, not `designed`, `implemented` or authorized for training.

## Claim boundary

Supported:

1. The exact G1 architecture is rejected on its frozen three-seed internal
   validation gate.
2. The one-layer scaffold/decomposition is the dominant observed source of the
   G1 gap; K384 residual support is a smaller high-IoU-sensitive factor.
3. Zero initialization did not leave residual final layers identically zero
   through training.

Not supported:

- official THUMOS test performance or comparison with `63.xx/66.xx`;
- a paper main-table DCSR row;
- speedup, FLOPs, energy or end-to-end efficiency;
- probability calibration or pre-NMS proposal causality;
- universal failure of sparse heads or conditional computation;
- a unique causal separation of representation capacity and optimization.
