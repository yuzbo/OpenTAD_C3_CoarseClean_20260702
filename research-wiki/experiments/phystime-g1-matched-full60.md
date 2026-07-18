---
type: experiment
node_id: exp:phystime-g1-matched-full60
title: "PhysTime G1 native-J192 matched two-arm 60-epoch validation"
idea: idea:phystime-tad-2
status: experiment_running
verdict: pending
confidence: no_full60_metrics_yet
metrics: "NA until both FULL_COMPLETE.json artifacts pass independent validation."
provenance: "/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800"
added: 2026-07-18T11:25:00+08:00
---

# PhysTime G1 Matched Full60

## Question

Does the physical-time metric advantage from the matched 20-epoch run persist
under a complete 60-epoch schedule when every other experimental variable is
held fixed?

## Fixed Comparison

- `selected_axis`: official ActionFormer topology with uniform-rank-derived seconds.
- `physical_metric`: the same topology with real physical seconds.
- Both arms use the same THUMOS14 raw videos, K=384 raw observations, J=192
  native tubelet tokens, seed 42, VideoMAE checkpoint, optimizer, augmentation,
  evaluator, and fixed no-GT irregular sampler.
- G1b SDPQ is excluded because the matched medium experiment did not support it.
- Feature interpolation is disabled in both arms.
- The cosine scheduler and workflow both end at 60 epochs. Validation starts at
  epoch 40 and runs every two epochs; epoch 59 must be evaluated.

## Deployment

- Code commit: `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`
- Git tree: `bddc9b9386604d00d213275a47ce7997b35d3f4c`
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1_full60_0dc5851_20260718`
- Remote focused verification: `54 passed in 61.36s`
- Gate job: `1170945`, `COMPLETED 0:0`, elapsed `00:04:47`
- Selected-axis job: `1170946`, running after gate
- Physical-metric job: `1170947`, running after gate
- Submission free space: `31,285,616 KiB`; fail-closed floor: `8,388,608 KiB`

The real gate binds commit, tree, canonical configs, the complete dataset
manifest, the pretrained checkpoint, K/J geometry, optimizer behavior, and the
official evaluator. Its top-level `gate_pass` is true.

## Artifact Contract

Each arm must produce epoch-59 predictions and metrics, a final-only lightweight
checkpoint containing finite online and EMA weights, and `FULL_COMPLETE.json`.
The independent validator recomputes mAP and rejects stale artifacts, schedule
drift, config/data/checkpoint hash drift, missing EMA, optimizer/scheduler state
inside the lightweight checkpoint, or a dirty runtime snapshot.

## Evidence Boundary

The current status is `experiment_running`. No full60 mAP exists yet. Even a
successful single-seed result can only become `full60-single-seed-supported`;
it cannot become `paper_ready` without replication, mechanism diagnostics,
cost evidence, and cross-dataset evidence.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
