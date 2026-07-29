---
type: experiment
node_id: exp:georoute-real-batch-amp-diagnostic-v1
title: "GeoRoute real-batch AMP diagnostic v1"
idea: idea:geo-route-adatad
stage: tested
status: matched_diagnosis_complete_localized_repair_authorized
verdict: root_cause_localized_repair_authorized
confidence: high
commit: 861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58
jobs: [1204847, 1204848, 1204849, 1204864, 1204865, 1204866, 1204908, 1204909, 1204910, 1204944, 1204945, 1204946]
updated: 2026-07-30
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
checks pass `71/71`.

The exact clean `64d991f9` snapshot passed the remote suite `99/99`, then
created root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_real_batch_amp_diag_64d991f9_20260730_0050`
with PL/ST/finalizer Jobs `1204864/1204865/1204866`. Both leaves stopped before
observer construction because the binding incorrectly treated
`SlidingWindowDataset.block_list` as an inclusion list. The historical pilot
actually blocks Gate for train and Fit for val/test, yielding Fit-train and
Gate-development populations. Finalizer `1204866` sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR`; internal/file SHA-256 values are
`3de84a8b5260485ca2b583be6a99f4994e92b20378dd3aaf1879de056803acd0`
/
`5a7d69afda7745d442de6dde1261123ea7bb771d1fe8dbcbbda748f76f64f37f`.
No observer receipt or numerical conclusion exists.

Population-contract source
`047f643f4f78f5a954364d4f9b8e694c93f16079` upgrades the binding schema,
records included and blocked populations separately, and passed the exact
remote suite `149/149`. Fresh root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_real_batch_amp_diag_047f643f_20260730_0059`
ran PL/ST/finalizer Jobs `1204908/1204909/1204910`. The leaves reached the same
real batch with identical data/CPU/CUDA RNG SHA-256 values
`1248e7d69aebe0ceed796b00d2c5564eef5c07eba6187147e69dbc1bf7766ee5`,
`a1426d06adb3fc7defd9067b40b71fb7005b3ba3693e71e9f719caffc5cf10c0`,
and
`a8d5ab7f2d377467715101488b9d063ccf88eb2eb1cbd9179935e78c7cd3ec7d`.
Both forwards had finite losses, but both stopped before the first optimizer
attempt because the diagnostic classified itself as a formal binding and
therefore enabled strict deterministic error mode, unlike the historical
pilot's deterministic warn-only seed policy. Finalizer `1204910` sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR`; internal/file SHA-256 values are
`7755f777d4dbecb3c5024100f0752c3147dc70f81a4d099ba9e77ece6ae6deac`
/
`1aea037cda2504f3a4a3a7c57d2628c7242829189722a8c8d1e78a0af838c19f`.

Exact candidate `861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` binds
`deterministic_algorithms_enabled=true` and `deterministic_warn_only=true`,
uses that receipt field in `tools/train.py`, and fail-closes if the runtime
seed policy differs. This matches the historical pilot and changes no data,
model, estimator, seed, K, optimizer, or frozen classifier. Local combined
checks pass `71/71`.

The clean matched namespace
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_real_batch_amp_diag_861e9b1e_20260730_0107`
ran PL/ST/finalizer Jobs `1204944/1204945/1204946`, all `COMPLETED 0:0`.
PL recorded nine failed scaled attempts from `65536` through `256`, always and
only in `scout_score_function`, then its first successful update at `128`.
Matched ST recorded zero failures and succeeded at `65536`. The two arms had
identical data, CPU RNG and CUDA RNG SHA-256 values
`1248e7d69aebe0ceed796b00d2c5564eef5c07eba6187147e69dbc1bf7766ee5`,
`a1426d06adb3fc7defd9067b40b71fb7005b3ba3693e71e9f719caffc5cf10c0`,
and
`a8d5ab7f2d377467715101488b9d063ccf88eb2eb1cbd9179935e78c7cd3ec7d`.
Finalizer `1204946` emitted `COMPLETE_NUMERICAL_DIAGNOSTIC_ONLY /
ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`; internal/file SHA-256 values are
`3960f747f0c5de9ba9e7de3046812f01f3474c67b63661c8382e78a4647b3c4c`
/
`d725e589e315434eca3fd0e0245cffa6e01e1b3490d10b6cae27ec361620a0d0`.
There were no checkpoints, predictions, evaluators, metrics or official-test
artifacts.

This authorized only a cause-matched numerical repair. Source `768e1a30`
introduced an explicit, backward-compatible temporal reduction and selected
`mean` for the development candidate; source `86ff1dde` additionally bound
every stability input to this exact diagnosis. It did not authorize a
performance run. Stability-v1 subsequently failed its deliberately stricter
zero-skip-at-65536 rule in both arms and is recorded separately as
`exp:georoute-real-data-amp-stability-v1`.

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
