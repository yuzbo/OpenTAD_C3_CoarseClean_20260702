---
type: experiment
node_id: exp:phystime-g1-matched-full60
title: "PhysTime G1 native-J192 matched two-arm 60-epoch validation"
idea: idea:phystime-tad-2
status: empirically_supported
verdict: full60_single_seed_supports_physical_metric_not_paper_ready
confidence: single_seed_single_dataset_full60_validated
metrics: "Final epoch 59: selected-axis 41.28, physical-metric 57.57 Avg-mAP, delta +16.29; both FULL_COMPLETE validators pass with replayable finite online/EMA checkpoints."
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
- Selected-axis job: `1170946`, `COMPLETED 0:0`, elapsed `07:49:31`
- Physical-metric job: `1170947`, `COMPLETED 0:0`, elapsed `07:55:13`
- Submission free space: `31,285,616 KiB`; fail-closed floor: `8,388,608 KiB`

The real gate binds commit, tree, canonical configs, the complete dataset
manifest, the pretrained checkpoint, K/J geometry, optimizer behavior, and the
official evaluator. Its top-level `gate_pass` is true.

## Results

The first matched validation after epoch 41 completed for both arms:

| Variant | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-axis | 64.54 | 54.50 | 40.78 | 26.53 | 12.87 | 39.84 |
| physical-metric | 77.57 | 71.41 | 61.44 | 48.00 | 28.17 | 57.32 |
| physical minus selected | +13.03 | +16.91 | +20.66 | +21.47 | +15.30 | +17.48 |

All ten matched validations at epochs 41/43/45/47/49/51/53/55/57/59 retain
the same ordering. Final epoch-59 Avg-mAP is selected-axis `41.28%` and
physical-metric `57.57%`, delta `+16.29`. Final mAP at tIoU 0.3:0.7 is
`64.82/56.39/42.63/27.71/14.86` versus
`77.20/70.49/62.53/49.01/28.64`. The best logged validation for both arms is
epoch 55 (`41.44/57.66%`), so the final epoch remains close to the best.

Relative to the same-seed 20-epoch run, selected-axis improves from `30.42`
to `41.28`, physical-metric improves from `44.88` to `57.57`, and the method
delta grows from `+14.46` to `+16.29` Avg-mAP.

## Artifact Contract

Each arm must produce epoch-59 predictions and metrics, a final-only lightweight
checkpoint containing finite online and EMA weights, and `FULL_COMPLETE.json`.
The independent validator recomputes mAP and rejects stale artifacts, schedule
drift, config/data/checkpoint hash drift, missing EMA, optimizer/scheduler state
inside the lightweight checkpoint, or a dirty runtime snapshot.

Both arms pass this contract. Each checkpoint is 401,895,677 bytes and contains
499 finite online plus 499 finite EMA entries, with no optimizer or scheduler
state. Both completion artifacts report `validation_pass=true`,
`evaluated_weights_replayable=true`, and 422,000 predictions. Training losses
remain finite, peak logged memory is 3596 MB, and all required anomaly counts
are zero.

## Evidence Boundary

The current status is `empirically_supported`, specifically
`full60-single-seed-supported`. The complete schedule confirms that using the
real physical-time metric is substantially better than selected rank under this
matched THUMOS14 setting. It is not `paper_ready`: replication, mechanism
diagnostics, cost evidence, robustness families, and cross-dataset evidence are
still required.

## 2026-07-19 External Review Boundary

The reviewed `57.57` result remains valid. The review found no P0 issue that
invalidates training, prediction, or evaluation. The early rounding used by
cross-window NMS is shared by both arms, so it should be fixed prospectively but
does not erase the matched delta.

This experiment must not be used as a fair direct comparison with the older
`63.61` random-sampling ActionFormer or the `68.29` dense anchor. Those systems
change feature interpolation, candidate density, coordinate semantics, or raw
observation cost. The next Q-lift architecture must rerun all four Q/coordinate
arms; this experiment then becomes a historical external anchor.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
