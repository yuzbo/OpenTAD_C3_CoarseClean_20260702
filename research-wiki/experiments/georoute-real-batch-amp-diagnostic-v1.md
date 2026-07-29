---
type: experiment
node_id: exp:georoute-real-batch-amp-diagnostic-v1
title: "GeoRoute real-batch AMP diagnostic v1"
idea: idea:geo-route-adatad
stage: designed
status: approved_design_pending_implementation
verdict: pending
confidence: high
commit: pending
jobs: pending
updated: 2026-07-29
---

# GeoRoute real-batch AMP diagnostic v1

## Question

Why does residual-PL fail the first real training batch after all eight AMP
retries even though the synthetic full-graph P0 passes, and is the failure
specific to the PL score-function path?

## Parent evidence

The immutable parent is exact source
`c822add335c38a9f6c63e609237c4bfa9b9f468d`, run root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_c822add3_20260729_2149`.
Residual-PL Job `1204309` is the sole formal stage hard failure. Closeout
`1204314` emitted `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`; no performance
artifact from that namespace is usable.

The parent did not persist exact sample indices, input tensors, sampler state
or complete pre-forward RNG state. Therefore this study cannot claim bitwise
replay of Job `1204309`. It is a deterministic same-config reproduction on the
same production data path.

## Frozen diagnostic

- matched `residual_pl_rep_off` and `residual_st_rep_off`;
- same data, seed `3407`, initialization, optimizer and first real batches;
- exact production `tools/train.py` and `train_one_epoch` path;
- optional observer records batch/input/RNG fingerprints, loss components,
  PL audit values, and gradients after scaled backward, unscale, clipping and
  scaler update;
- diagnosis-only retries continue below the historical scale `256` to identify
  the first successful scale, without changing the historical experiment;
- no checkpoint, prediction, evaluator, metric or official test;
- atomic self-hashed receipts and an all-terminal finalizer.

## Decisions

Only `ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`,
`ROOT_CAUSE_NOT_LOCALIZED_HOLD`, or
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR` may be emitted. A repair must be the smallest
change supported by the observed failing stage and parameter group.

A repaired source must then pass a fresh production-path real-data stability
gate before another six-arm study is considered.

## Paper boundary

This diagnostic can never enter a paper performance table. The historical
20-epoch single-seed pilot is also development-only. Any future paper result
requires an official AdaTAD reproduction, a matched native-source dense
control, identical optimization/evaluator/NMS contracts, disjoint multi-seed
confirmation, sealed official test, and decode-to-NMS latency/memory/energy.

Full design:
`docs/superpowers/specs/2026-07-29-georoute-real-batch-amp-diagnostic-design.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
