---
type: experiment
node_id: exp:georoute-ddp-fp16-cast-repair-gate-v1
title: "GeoRoute DDP FP16-cast no-compression repair gate v1"
idea: idea:geo-route-adatad
stage: implemented
status: local_tests_passed_pending_remote_kat
verdict: pending
confidence: high
updated: 2026-07-30
---

# GeoRoute DDP FP16-cast no-compression repair gate v1

## Research question

Does removing the uniquely implicated FP16 DDP bucket cast make the matched
64-batch residual-PL/ST official-prefix execution satisfy the already frozen
bounded dynamic-scaler rule, without changing the estimator or objective?

## Authorization

The sealed parent
`exp:georoute-gradient-decomposition-diagnostic-v1` uniquely classified all
three PL-specific failures as `DDP_FP16_CAST_OVERFLOW`. Its finalization
self/file SHA-256 are
`52d4dfd698ed0679a976e6d468fb4b0d1ede9ea630df32f808115c9f118f681e`
/
`816819086374f964264d3a8bb4810842f97ef554d5661d2ec4a6b85fd135bc9c`.
That result authorizes exactly one class-specific repair and another
no-performance gate.

## Frozen protocol

- two matched arms: residual PL and residual ST, representation off;
- independent mechanically derived seed `2307`;
- 64 real development batches at `B/T/N/K=1/384/220/64`;
- default `GradScaler`, zero retry/replay, scheduler and EMA per consumed batch;
- same data, model, objective, optimizer, clipping, temperature, weight,
  baseline and selector as the parent diagnosis;
- the sole intervention is `solver.fp16_compress=false` for both arms;
- inherited thresholds: at most two nonconsecutive skips, scale floor `16384`,
  at least 62 updates, successful final-16 tail, pairwise skip delta at most one
  and final-scale ratio at most two; and
- no metric, checkpoint, prediction, evaluator/NMS or official-test surface.

The exact design is
`docs/superpowers/specs/2026-07-30-georoute-ddp-fp16-cast-repair-gate-v1-design.md`.

## Implementation

The implementation adds a versioned AMP protocol profile, explicit
no-compression binding and validation, a matched pair classifier, exact
gradient-parent/KAT admission, a CUDA/DDP KAT proving finite `70000` FP32
reduction survives while its detached FP16 shadow overflows, and fail-closed
stage/finalizer support.

Local focused AMP/repair tests pass `32/32`; remote Linux/Torch full-suite,
same-commit KAT and Slurm gate execution remain pending. This is `implemented`,
not yet `experiment_running`.

## Claim boundary

Even a pass is numerical repair evidence only. It may authorize freezing a
separate official-comparable performance protocol, but it cannot rank PL/ST,
report mAP/cost, select a winner, open official test, start P2/P3, or support a
paper result.
