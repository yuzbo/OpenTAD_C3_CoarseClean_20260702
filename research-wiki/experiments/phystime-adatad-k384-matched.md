---
type: experiment
node_id: exp:phystime-adatad-k384-matched
title: "PhysTime-AdaTAD raw-video K=384 matched three-head experiment"
idea: idea:phystime
verdict: no
confidence: high
commit: "3ac93a12c299012db64513567d5bdedf0c6d5f71"
jobs: "1159491-1159495"
updated: 2026-07-12
---

# PhysTime-AdaTAD raw-video K=384 matched experiment

## Contract and gates

- Snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_phystime_adatad_3ac93a1_20260712_ampfix`.
- Jobs `1159491` (real AMP/evaluator gate) and `1159492` (two-epoch
  stability gate) completed successfully.
- The matched formal runs use the same raw-video input and the same fixed,
  non-learned, GT-free irregular K=384 sampling policy. Only the temporal
  detection geometry/head differs: selected-axis, physical-grid, or PhysTime.
- Formal jobs `1159493`, `1159494`, and `1159495` completed 60 epochs with
  exit code 0 and `TRAINING_COMPLETE` markers.

## Final raw results

Best-checkpoint validation results:

| Variant | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| selected-axis (epoch 59) | 63.61 | 79.87 | 74.15 | 66.12 | 56.02 | 41.87 |
| physical-grid (epoch 57) | 59.14 | 77.09 | 71.80 | 63.74 | 50.74 | 32.34 |
| PhysTime (epoch 59) | 57.21 | 72.70 | 68.38 | 60.94 | 49.06 | 34.96 |

Final epoch-59 loss was 0.4140 for selected-axis (classification 0.2102,
regression 0.2038), 0.4843 for physical-grid (0.2443, 0.2400), and 0.4507 for
PhysTime (classification 0.1607, regression 0.2113, endpoint 0.0787). These
losses are not directly comparable because PhysTime has a different head and
an additional endpoint objective.

Runtime summaries were 28000/29646/29401 seconds and 3959/3957/3555 MB peak
GPU memory for selected-axis/physical-grid/PhysTime. The real gate's single
sample inference timing was 32.96/40.75/43.13 ms; this is diagnostic only, not
a formal latency benchmark.

No real AMP optimizer skip, NaN, OOM, traceback, missing file, or overflow
budget failure was present. The two `amp_skip` text matches in each log are
configuration fields (`max_consecutive_amp_skips` and
`max_total_amp_skips_per_epoch`), not executed skips.

## Result-to-claim verdict

`claim_supported = no`, confidence high. Relative to selected-axis, PhysTime
loses 6.40 Avg-mAP and 6.91 points at tIoU 0.7, and trails at every threshold.
Relative to physical-grid it loses 1.93 Avg-mAP and tIoU 0.3-0.6, while gaining
2.62 points only at tIoU 0.7. This narrow strict-localization signal is not an
overall or selected-axis advantage and is not attributable from the current
three-way comparison.

The current PhysTime-AdaTAD v1 superiority claim is rejected and Phase 2 is
not unlocked. If the narrow tIoU-0.7 signal is pursued, the only justified next
experiment is a preregistered matched factorial ablation that independently
removes timestamps, support geometry, and endpoint supervision within the
same head/training setup, followed by multiple seeds. Arbitrary loss or
hyperparameter tuning is not justified.

## Provenance

Run root:
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800`.
Raw evidence is in each variant's `train.out`, checkpoints, Slurm accounting,
`real_gate/real_gate.json`, `real_gate/matched_contract.json`, and
`phystime_stability_gate/runtime_summary.json`.

## Connections

Maintained only in `research-wiki/graph/edges.jsonl`.
