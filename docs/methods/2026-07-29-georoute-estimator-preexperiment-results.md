# GeoRoute estimator/representation D/K/M results

Date: 2026-07-29

Status: `tested_complete_go_pilot_design_only`

Decision: `GO_PILOT_DESIGN_ONLY`

This is a development-only diagnostic result. It authorizes freezing the
separate six-arm exploratory pilot, but it does not authorize full CER-TAD,
P2/P3, official test, an efficiency claim, or a paper claim.

## Immutable evidence

- Runtime source:
  `0c20f2e89e6af8bac0e3612776e03f80c0a9f3fb`
- Source experiment:
  `7be8363ea6e26b320bffafeb03f0e82d8b660779`
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_repiso_0c20f2e8_dkm_20260729_1643`
- KAT Job `1203105`; decode Job `1203106`; Phase-M Jobs
  `1203107`--`1203112`; finalizer Job `1203113`.
- Every job completed `0:0`.
- Finalization SHA-256:
  `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
- KAT receipt SHA-256:
  `046ac67f5a44f37d4529f5b178f181623e9b207366fc7d06454894e52e756dd0`
- Decode receipt SHA-256:
  `e7f05bad7a05d51443f34abe5c196920029d5d74c59572e9290c7a3166b9c141`

## D: exact-index decode census

The exact development sliding-window dataset contains 136 items. Two complete
passes produced 272/272 successful item retrievals and zero failures. The
population SHA-256 is
`ece8106f492be3bf69c6894c6ac37bb2cb9cc1cd49763119d609f56ac7c638f8`.

This closes the immediate question of whether the old ROI failure implied a
population-wide deterministic decode defect. It does not prove future long-run
training I/O reliability, so each pilot leaf still fails on its first decode
error and is never resumed.

## K: estimator and representation known-answer tests

| Check | Result |
| --- | ---: |
| PL manual log-probability error | `0` |
| PL ordered probability normalization error | `0` |
| PL selected / unselected min absolute gradient | `0.705795 / 0.295805` |
| exact risk-gradient max error / cosine | `6.94e-18 / 1.0` |
| ST selected max absolute gradient | `0.357143` |
| ST unselected nonzero gradient count | `0` |
| PL selected / unselected min absolute gradient in matched reachability case | `0.745799 / 0.032871` |
| representation-off geometry/coordinate perturbation | bitwise invariant |
| representation-off geometry/coordinate gradients | absent |
| legacy all-enabled maximum error | `0` |

All three representation channels were independently effective in the frozen
known-answer case: absolute-coordinate, ROI-relative, and geometry-projection
maximum output deltas were `1.02583`, `1.15240`, and `1.55424`.

The result proves mathematical reachability and implementation isolation only.
It does not establish lower variance, better optimization, or higher mAP for
PL.

## M: prediction-neutral replay

All six leaves replayed the same 136-window population, reproduced their source
prediction JSON SHA-256 exactly, and emitted population SHA-256
`8acf7c49d562ec457768edf5665375447e05d92c5edb1b9f479bcc4e043e9cf4`.

### Diagnostic execution profile

The numbers below exclude the evaluator and are not paper-grade end-to-end
latency or energy. Loader wait is highly variable, so model/postprocess and
same-process window wall time are reported separately.

| Arm | model+post p50/p95 ms | loader p50/p95 ms | window p50/p95 ms | peak MB |
| --- | ---: | ---: | ---: | ---: |
| dense | `221.48 / 343.79` | `762.74 / 4307.32` | `971.99 / 4531.38` | `2681.35` |
| fixed | `129.36 / 260.21` | `1223.79 / 4285.44` | `1370.01 / 4424.91` | `1816.72` |
| fixed+geometry | `141.14 / 282.38` | `493.89 / 4376.51` | `664.11 / 4501.48` | `1818.08` |
| random | `127.82 / 226.47` | `1223.50 / 4264.00` | `1350.56 / 4400.01` | `1817.18` |
| free | `130.59 / 244.09` | `949.69 / 4181.87` | `1103.68 / 4359.44` | `1818.16` |
| hybrid | `129.06 / 265.28` | `609.06 / 4340.95` | `838.63 / 4474.97` | `1818.08` |

### Route telemetry

| Arm | adjacent Jaccard | lineage retention | x/y span | geometry area | selected route hashes |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | `1.000` | `1.000` | `0.950 / 0.909` | `1.000` | `1` |
| fixed | `1.000` | `1.000` | `0.950 / 0.909` | `1.000` | `1` |
| fixed+geometry | `1.000` | `1.000` | `0.950 / 0.909` | `0.299` | `1` |
| random | `0.152` | `0.263` | `0.947 / 0.909` | `1.000` | `1` |
| free | `0.830 ± 0.093` | `0.896` | `0.802 / 0.747` | `1.000` | `136` |
| hybrid | `0.795 ± 0.090` | `0.877` | `0.950 / 0.909` | `0.249` | `136` |

Free is selective and temporally sticky: selected/unselected residual means
are `3.157/1.302`, selected/unselected surrogate means are `0.765/0.171`, and
hard-soft L1 is `0.189`. This is nevertheless detector-misaligned in the old
completed run, whose diagnostic Avg-mAP was only `10.03`.

Hybrid has selected/unselected residual means `-0.939/-1.154`,
selected/unselected ROI means `-0.973/-1.513`, surrogate means
`0.551/0.506`, and hard-soft L1 `0.490`. Its weak surrogate separation and
large hard-soft mismatch motivate estimator/representation isolation before
any dynamic complementary router is built.

Jaccard, span, and area are not directly comparable across cardinalities or
route geometries. The old mAP values are source-run metadata preserved by
prediction parity; Phase M did not create new accuracy evidence.

## Research implication

The implementation can now cleanly answer whether:

1. PL changes detector utility relative to ST with residual support and all
   new representation channels off;
2. geometry representation helps under fixed support;
3. geometry representation helps under learned ROI support; and
4. learned ROI support helps relative to residual support when both use PL and
   representation-off.

Those four contrasts define the independent single-seed exploratory pilot.
Full context/geometry/residual allocation, a critic, boundary supervision,
coverage, and temporal-stability losses remain `discussed`.
