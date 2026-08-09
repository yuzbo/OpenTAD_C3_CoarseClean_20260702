# Experiment audit: SCNR residual-centering matched training v1

Audit date: 2026-08-09

Experiment date: 2026-08-06

Scope: seed-3407 THUMOS14 Fit-to-Gate `none_control` versus
`residual_window_center` matched training and duplicate accuracy replay

Overall integrity: **PASS_WITH_SCOPE_WARNINGS**

## Verdict

The preregistered single-seed development accuracy screen is valid. Both fresh
G1 cells completed 60 epochs and exactly 9,600 successful updates, produced one
epoch-59 EMA checkpoint, and passed exact duplicate prediction, route,
population, branch, and metric replay. The centered cell improves Avg-mAP by
2.05 pp, mAP@0.6 by 2.14 pp, and mAP@0.7 by 1.16 pp. The registered finalizer
therefore correctly authorizes a separately frozen same-GPU ABBA+BAAB cost
study.

This audit does not authorize an efficiency, multi-seed, official-test,
complementarity, floor-selection, cross-detector, or final paper claim.

## Evidence identity

- model runtime:
  `16137484c5ccad422e017e67a81c1a07d1ed2fbb`;
- run root:
  `/data/run01/sczc063/yuzibo/scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`;
- Jobs: control `1223819`, centered `1223820`, finalizer `1223821`, all
  `COMPLETED 0:0`;
- shared complete-protocol SHA-256:
  `34defbdbc30e7fff10bbb05d7e6665dd29b8128f8f03cd389250bca9e3e7493c`;
- control/center stage SHA-256:
  `fbc23cf83af3a65a652c7c646471a9a50015a061af8d3adfeb0cfa5f0b792dfd` /
  `de893ebed139513ebd56dbfb5088935afed50344bdba13ebf64593e691d4836b`;
- control/center checkpoint SHA-256:
  `5350a03b1584ab8e0023b6c212fc2a3b8526c45de169b5660d827762a9dd6ff4` /
  `e45a37708c68e1ea02bec02bc14dfbe199ea8303f562088469d98a8fe45c7028`;
- finalization SHA-256:
  `2a9351a3c21c850f28aab4bd162f7b69f3ca40921a97304431a8a760d6ebbe8a`.

## Integrity dimensions

| Dimension | Verdict | Evidence |
|---|---|---|
| A. GT/provenance | PASS | Fit/Gate IDs are disjoint and hash-bound. GT is consumed by the development evaluator only. Route receipts prohibit GT, teacher, oracle, test evidence, and raw-prediction cache use. Official test remains closed. |
| B. Score handling | PASS | Official-style mAP values are parsed without prediction-dependent normalization. `high_iou_composite` is exactly the arithmetic mean of mAP@0.6 and mAP@0.7. |
| C. Result exactness | PASS | Both cells have one valid EMA checkpoint/sidecar, exact 60 epochs/9,600 updates, a common protocol hash and population, and byte-identical A/B predictions across 40 videos/80,000 candidates. Route and metric parity are exact. |
| D. Critical-path reachability | PASS | The training runner, checkpoint consumer, strict math-SDPA duplicate evaluator, branch summary, selected-role gate, and finalizer all produced and revalidated terminal artifacts. |
| E. Scope | WARN | One seed, one G1 anchor, one THUMOS14 development split. This is sufficient only for the preregistered development screen, not generalization. |
| F. Evaluation | WARN | Real-GT Gate development evaluation is valid, but there is no official-test result, independent statistical replication, matched baseline matrix, or second detector/dataset. |

## Recomputed result

| Variant | Avg-mAP | mAP@0.6 | mAP@0.7 | high-IoU |
|---|---:|---:|---:|---:|
| control | 10.52 | 8.90 | 6.98 | 7.94 |
| centered | 12.57 | 11.04 | 8.14 | 9.59 |
| delta (pp) | +2.05 | +2.14 | +1.16 | +1.65 |

All three preregistered signs pass. The terminal receipt has empty errors and
decision `RUN_SAME_GPU_ABBA_BAAB_FULL_STACK_COST`.

## Mechanism evidence

Control selected roles are context/ROI/residual `0/0/3,342,336`. Centering
changes them to `210,925/1,613,683/1,517,728`, passes the registered structural
reachability gate, and leaves a maximum absolute post-centering valid residual
mean of `2.6075e-7`. This supports role reachability and accuracy utility of the
single offset repair. It does not by itself prove causal role complementarity.

Control and centered `K_t` distributions differ despite equal mean 64 and exact
window B: observed ranges are `0..215` versus `5..206`. Therefore equal token
count cannot substitute for measured ragged/full-stack cost.

## Required next action

Run the separately frozen one-job same-GPU eight-pass cost protocol. Do not
open seeds 3408/3409 unless both primary cost-ratio 95% upper bounds are at most
1.05. Even a pass remains development evidence and requires independent cost
Jobs plus multi-seed and broader paper experiments.
