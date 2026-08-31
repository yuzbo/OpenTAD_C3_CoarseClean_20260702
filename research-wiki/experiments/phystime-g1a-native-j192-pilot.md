---
type: experiment
node_id: exp:phystime-g1a-native-j192-pilot
title: "PhysTime G1a native-J192 matched selected-axis vs physical-metric pilot"
idea: idea:phystime
verdict: partial
confidence: medium
commit: "623a376700c5781a3a54e3c6622ceb2ebc5ffc8e"
jobs: "1162048-1162050"
updated: 2026-07-15
---

# PhysTime G1a native-J192 matched pilot

## Contract

- Snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_623a376_20260713_checkpointfix`.
- Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800`.
- Gate job `1162048` completed with exit code 0.
- Selected-axis pilot job `1162049` completed with exit code 0.
- Physical-metric pilot job `1162050` completed with exit code 0.
- Gate evidence: `K_raw_observations=384`, `J_native_tubelet_tokens=192`,
  `Q0_base_candidates=192`, `Q_total_candidates=378`,
  `feature_interpolation=false`, AMP contract verified, final-only checkpoint
  writing active.

## Raw pilot results

| Variant | Status | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---|---:|---:|---:|---:|---:|---:|
| selected-axis | COMPLETED / validation_pass=true | 10.26 | 23.78 | 15.20 | 7.94 | 3.28 | 1.09 |
| physical-metric | COMPLETED / validation_pass=true | 10.56 | 24.64 | 15.84 | 8.08 | 3.18 | 1.04 |

`physical-metric` is +0.30 Avg-mAP over selected-axis in this six-epoch pilot,
mostly from low-IoU thresholds. It does not improve high-IoU localization:
mAP@0.6 and mAP@0.7 are slightly lower than selected-axis.

## Verdict

This pilot is an engineering and stability pass, not paper evidence. It shows
that the native-J192 physical-metric path can run through gate, training,
validation, and final checkpoint validation without NaN/OOM/traceback, but the
accuracy signal is too weak and too low-epoch to support a method claim.

The next valid step is not to expand seeds or write claims. The next step is a
detector-head diagnosis that explains why physical time helps low IoU slightly
but fails to improve high-IoU localization, and why the earlier full K384
physical-grid / PhysTime variants still trail selected-axis.

## 2026-07-15 geometry diagnosis

Commit `f2725f5` on `codex/phystime-performance-diagnosis-20260712` added
`tools/bata/analyze_phystime_g1a_geometry.py` and
`tests/test_phystime_g1a_geometry_diagnostic.py`. Local pure-Python tests passed
(`4 passed`); Windows torch-dependent tests remain blocked by the known
`c10.dll` initialization issue. Remote Linux validation passed the same test
and produced:

- Validation diagnostic:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800/diagnostics/g1a_geometry_val.json`.
- Train Monte-Carlo diagnostic:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800/diagnostics/g1a_geometry_train_s5.json`.

Validation geometry over 487 windows / 3665 GT shows physical-time seconds has
fewer positives than uniform-rank seconds: 10765 vs 11030 positive locations,
eligible locations per GT 3.155 vs 3.235, and GT without any eligible location
0.682% vs 0.191%. The short-action gap is larger: for `<1s` GT, no-eligible
fraction rises from 2.87% to 7.79%.

Training geometry over 1000 sampled windows / 7484 GT repeats the same pattern:
20437 vs 21043 positive locations, eligible locations per GT 2.946 vs 3.039,
and GT without any eligible location 1.523% vs 0.454%. For `<1s` GT,
no-eligible fraction rises from 3.62% to 11.49%.

This supports a concrete failure mode: the current physical-metric seconds
axis does not merely expose true timing; under the inherited ActionFormer
center-sampling and regression-range geometry it makes short-action assignment
less friendly. The weak low-IoU pilot gain is therefore not enough. The next
implementation should test physical-time-aware assignment/range design before
any larger training suite.

## 2026-07-15 rank-assignment diagnostic

Commit-under-test in the PhysTime worktree added a diagnostic
`physical_time_rank_assignment` mode: decode/regression centers remain on
`phystime_g1a_axis_positions_sec`, while center-sampling and range eligibility
can read `phystime_uniform_rank_timestamps_sec`. A hard guard keeps the
physical center inside GT, so the diagnostic cannot create positive samples
whose physical regression target would require negative distances.

Focused remote tests on Linux passed: `20 passed in 44.77s`. The real-data
diagnostic outputs are:

- `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800/diagnostics_rankassign_20260715/g1a_rankassign_geometry_val.json`.
- `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800/diagnostics_rankassign_20260715/g1a_rankassign_geometry_train_s5.json`.

The result is negative. On validation, `physical_time_rank_assignment` has only
7745 positives and 18.96% GT with no eligible location, much worse than both
physical-time seconds (10765 positives / 0.682% no eligible) and uniform-rank
seconds (11030 positives / 0.191% no eligible). For `<1s` GT, no-eligible rises
to 65.16%. On train-s5, it has 14201 positives and 25.13% no eligible, with
`<1s` no-eligible 72.02%.

This falsifies the simple hypothesis that changing assignment/range reference
alone fixes G1a. The dominant issue is that physical-time anchors themselves
can be clustered or shifted away from short GT. If the physical center is not
inside the action, a ReLU left/right distance head cannot learn that GT from
that anchor. The next valid direction is therefore not a rank-assignment pilot,
but a head redesign that separates observation support from query anchors:
e.g. uniform physical query anchors with sparse observation support features,
or explicit support-aware pooling/query features. Do not deploy the
rank-assignment variant as a paper experiment.

## Connections

Maintained only in `research-wiki/graph/edges.jsonl`.
