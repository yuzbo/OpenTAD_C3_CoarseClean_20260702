---
type: experiment
node_id: exp:georoute-real-data-amp-stability-v2
title: "GeoRoute official-semantics real-data AMP stability v2"
idea: idea:geo-route-adatad
stage: experiment_running
status: pl_threshold_exceeded_waiting_terminal_closeout
verdict: pending
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
finalizer `1205590` is dependency-held. Deployment self-hash is
`8c9ec92f927a1f5e7902c3abc0d5422eb8029b544a7b500dd51655daa17e543e`.
The preflight recorded `2/16` preexisting submissions and
`142833188864` free bytes. No performance artifact or conclusion exists.

## Irreversible live gate state

PL produced nonconsecutive score-function scaler skips at zero-based batches
`11`, `20` and `29`: `65536 -> 32768 -> 16384 -> 8192`. The third skip exceeds
the frozen maximum of two and the resulting scale is below the frozen `16384`
floor. Therefore this namespace can no longer pass, regardless of later
successful updates. ST had detector-group skips at batches `20` and `29`,
reaching scale `16384`, exactly its per-arm limit at that observation.

Both stages remain allowed to reach natural terminal state so the afterany
finalizer can seal canonical evidence. They will not be cancelled, resumed,
rerun or supplemented. The only admissible final decision is now
`OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`; no PL/ST performance or estimator
inference follows from this numerical failure.

## Paper boundary

This gate can only authorize freezing an official-comparable experiment. Its
losses, scales, skips and gradients are numerical provenance and never paper
performance evidence.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
