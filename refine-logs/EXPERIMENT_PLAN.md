# DUCA experiment plan

## Primary hypothesis

With the heavy VideoMAE backbone restricted to K=384 selected frames, DUCA's
train-only jointly optimized coarse evidence and learned temporal positions
improve official THUMOS14 localization over exact-uniform positions.

## Frozen Stage-A matrix

| Arm | Train videos | Evaluation videos | Seeds | Heavy input |
|---|---:|---:|---|---:|
| dense | 200 | 211 | 5801, 8123, 12011 | 768 |
| uniform_fixed_k384 | 200 | 211 | 5801, 8123, 12011 | 384 |
| uniform_mixed_train_k384_eval | 200 | 211 | 5801, 8123, 12011 | train mixed; eval 384 |
| duca_fixed_k384 | 200 | 211 | 5801, 8123, 12011 | 384 |

All cells use 60 epochs, two-process DDP, global batch size two, 100 updates per
epoch, and terminal epoch-59 EMA.  The shared augmentation identity seed is
3407 so the three optimization seeds remain paired across arms.

## Stage-B dependency

Generate full-training-set, out-of-fold per-K utility and risk targets from the
completed mixed-K pipeline.  Freeze a training-only mean-K384 budget protocol.
Then train three `duca_dynamic_mean384` cells and evaluate three corresponding
exact `uniform_same_realized_k` replays.

## Measurements

Primary model evidence is official mAP at tIoU 0.3:0.7, including the average
and per-threshold values.  Required analyses also include high-IoU localization,
short-action behavior, the realized-K distribution, full-stack latency,
throughput, and peak memory.  Inference comparisons use video-clustered paired
uncertainty over the exact 211-video set; no single-seed claim is allowed.

## Exclusions

Admission simulation, training-subset mAP, partial matrices, intermediate
epochs, H-RIME, TriDet, and K192 are outside the current execution scope.
