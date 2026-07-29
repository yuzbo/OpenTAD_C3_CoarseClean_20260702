---
type: experiment
node_id: exp:georoute-real-batch-amp-diagnostic-v1
title: "GeoRoute real-batch AMP diagnostic v1"
idea: idea:geo-route-adatad
stage: implemented
status: first_namespace_sealed_binder_failure_repair_pending_remote_linux
verdict: pending
confidence: high
commit: 64d991f96981a3e60b10f47d6d093d5457da9c60
jobs: [1204847, 1204848, 1204849]
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

## Implementation status

Exact implementation source is
`832caedd3713f477cb4b2f29a692acba9cd5a836`. The diagnostic observer is
strictly opt-in and the default train path does not publish or compute
diagnostic telemetry. The two held GPU leaves run in parallel and feed one
`afterany` finalizer. Capacity, storage, and all three `sbatch --test-only`
admission checks precede namespace creation. The deployment explicitly binds
the failed parent runtime `c822add335c38a9f6c63e609237c4bfa9b9f468d` and
parent finalization file hash; stage results and wrapper failures bind arm,
runtime, Slurm Job ID, self-hash, rendezvous and zero-performance-artifact
guards.

Local pure contract/finalizer/train-engine checks pass `50/50`; required C3
regressions pass `20/20`; Python compilation, Bash syntax and
`git diff --check` pass. Windows cannot load the local Torch `c10.dll`, so the
model/observer path is not marked remotely tested. Clean N16R4 Linux/CUDA
replay is mandatory before either diagnostic job may be submitted.

The exact `832caedd` clean N16R4 snapshot subsequently passed the combined
Linux/Torch suite `98/98`. Its first no-metric namespace is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_real_batch_amp_diag_832caedd_20260730_0040`.
PL `1204847` and ST `1204848` both failed in six seconds before config
publication, observer construction, data loading or model forward because
`mmengine.Config` does not implement `__delitem__`. Afterany finalizer
`1204849` completed and sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR`, empty arms/metrics and all performance/test
guards false; finalization self-hash is
`feda83e084ece379faa07e828a88e017e5bb698eba7c78d1c4866c8cd09c77da`.
This is a common binder infrastructure failure, not PL/ST evidence.

Exact repair `64d991f96981a3e60b10f47d6d093d5457da9c60` replaces only
`del cfg[key]` with the already supported `Config.pop` and adds a real
`mmengine.Config` binder regression. Local diagnostic/pilot/train-engine/C3
checks pass `71/71`. A clean proxy-synced snapshot, complete remote Linux suite
and a wholly new namespace are required; the failed namespace is not resumed.

## Paper boundary

This diagnostic can never enter a paper performance table. The historical
20-epoch single-seed pilot is also development-only. Any future paper result
requires an official AdaTAD reproduction, a matched native-source dense
control, identical optimization/evaluator/NMS contracts, disjoint multi-seed
confirmation, sealed official test, and decode-to-NMS latency/memory/energy.
The current development config differs from the official recipe in data
population, batch size, warmup, EMA/optimizer surface, checkpoint/evaluation
workflow and post-processing; therefore even a successful diagnostic or
replacement pilot is not an official-comparable result.

Full design:
`docs/superpowers/specs/2026-07-29-georoute-real-batch-amp-diagnostic-design.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
