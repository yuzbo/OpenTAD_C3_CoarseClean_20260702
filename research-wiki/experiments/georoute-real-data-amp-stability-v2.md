---
type: experiment
node_id: exp:georoute-real-data-amp-stability-v2
title: "GeoRoute official-semantics real-data AMP stability v2"
idea: idea:geo-route-adatad
stage: tested
status: sealed_terminal_hold
verdict: OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD
confidence: high
updated: 2026-07-30
---

# GeoRoute official-semantics real-data AMP stability v2

## Question

Under the official AdaTAD dynamic GradScaler, zero-retry, no-replay and
scheduler/EMA-per-batch transition semantics, do repaired residual-PL and
matched residual-ST show only bounded initial scale adaptation and then a stable
tail on an independent 64-batch real-data order?

## Frozen design

- profile `stability_official_semantics_v2`;
- diagnostic-only seed and data-order seed `4417`;
- two matched arms in parallel plus an `afterany` finalizer;
- 64 consumed batches, default GradScaler, initial observed scale `65536`;
- at most two nonconsecutive skips, minimum/final scale `16384`;
- final 16 attempts all successful;
- cross-arm skip difference at most one and final scales within one halving;
- no retries, replay, checkpoint, prediction, metric, evaluator or test;
- exact parent, input, official-reference-config and runtime binding.

Full specification:
`docs/superpowers/specs/2026-07-30-georoute-official-semantics-amp-stability-v2-design.md`.

## Implementation and local verification

The separately versioned v2 profile is implemented without modifying the
sealed v1 schemas or decision rule. It binds execution/data-order seed `4417`,
uses the default GradScaler constructor, disables retry/replay and fail-on-skip,
advances scheduler and EMA per consumed batch, and records ordered scale,
skip, stable-tail, RNG, data and transition-audit telemetry. It also requires
the sealed stability-v1 HOLD, exact origin-ref parity and an exact hashed
official AdaTAD reference config before deployment.

The reference binding verifies only the official AMP, clipping, scheduler
advance cadence and EMA transition semantics. The development source still has
different scheduler hyperparameters and is explicitly marked
`official_performance_comparable=false`, `full_official_recipe_matched=false`
and `full_official_training_claimed=false`.

Local compilation and all non-Torch focused tests pass. Windows tests that
import the user-site Torch binary remain unavailable because `c10.dll` cannot
initialize.

Exact runtime source
`27fba03cb6d4932ee10cb4545b97984dff28c28c` was proxy-resolved, cloned and
verified against
`refs/remotes/origin/codex/spatial-zoom-s1-audit-fix-20260715` with a clean
tree. The remote Linux/Torch suite passed `168/168`. The fresh no-resume run is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_official_semantics_amp_stability_v2_27fba03c_20260730_0800`.
PL Job `1205588` and ST Job `1205589` were released in parallel; afterany
finalizer `1205590` sealed the terminal evidence. Deployment self-hash is
`8c9ec92f927a1f5e7902c3abc0d5422eb8029b544a7b500dd51655daa17e543e`.
The preflight recorded `2/16` preexisting submissions and
`142833188864` free bytes. No performance artifact or conclusion exists.

## Terminal evidence

PL produced nonconsecutive score-function scaler skips at zero-based batches
`11`, `20` and `29`: `65536 -> 32768 -> 16384 -> 8192`. The third skip exceeds
the frozen maximum of two and the resulting scale is below the frozen `16384`
floor. PL consumed all 64 batches, completed 61 optimizer updates, had maximum
consecutive skip count one, retained finite forward losses, and completed a
successful final-16 tail, but those later successes cannot reverse either
registered violation. Its final receipt status is
`FAIL_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION`; the emitted failure is the
fail-closed diagnostic condition, not OOM or a non-finite forward loss.

ST consumed all 64 batches and completed 62 optimizer updates. Its two
detector-group skips occurred at batches `20/29`; final and minimum scale were
`16384`, maximum consecutive skip count was one, and the final 16 updates all
succeeded. Its arm-only status is
`PASS_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION_ONLY`. Both arms advanced
scheduler and EMA 64 times, used zero retry/replay, and have identical ordered
data, CPU-RNG and CUDA-RNG hashes:

- data: `6081c33b6d4437fb6b53b5fa44e7e85d41c059de4678a5f373aff146e6b877f3`;
- CPU RNG: `ba6b26a1da2d09d8f19bad8bae486a94ced7f176f70e3cdfdab57295fdba7e04`;
- CUDA RNG: `49a3f706f246c7fe9b3460920863f5fe4d3c87c60d7e1e55f8f5c4bd408a5c72`.

Slurm terminal states were PL `FAILED 1:0`, ST `COMPLETED 0:0`, and finalizer
`COMPLETED 0:0`. Finalization schema
`georoute_real_data_amp_stability_official_semantics_finalization_v2` emitted
status `INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2`, decision
`OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`, reason
`one_or_more_arms_failed_execution`, and all protocol-freeze, performance,
paper, official-test and P2/P3 guards false.

Canonical evidence:

- finalization internal/file SHA-256:
  `ab7ea3e5fca378532b689f8dce8d3ed57631ca78eec99b91a77a96a5e8e29d56` /
  `c7f59dbcec609430bdf4aafe99cc5ef3272ef93362b7f44ba74bcbc337c85ab0`;
- PL receipt internal/file SHA-256:
  `ec2c2e5f52b5d4aa56ca8d130bf60c6fb96cbcd16bd76af05db8074430f7c9f2` /
  `6667c8f0f357984fc19dd18d9bbe466241749dc77b6d5ce434b2e106ef7da277`;
- ST receipt internal/file SHA-256:
  `9340be6c8b19b99111109faf1b9bcda09a2e86af85c0acd7c2e3cd67e6d6e17a` /
  `cdc3947bb42e63799a5c8591274d54d508c3bdbda99ecbea8d05eb7df01c555d`.

All three internal self-hashes were independently recomputed successfully.
Both stage artifact audits recorded zero checkpoint, prediction, evaluator and
temporary payloads; no `.tmp` file or performance-named output remained.
This namespace is sealed: no cancellation, resume, rerun, supplementation or
threshold reinterpretation is permitted.

## Paper boundary

This gate can only authorize freezing an official-comparable experiment. Its
losses, scales, skips and gradients are numerical provenance and never paper
performance evidence. Because the final decision is HOLD, it did not authorize
protocol freezing or any formal multi-seed experiment.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
