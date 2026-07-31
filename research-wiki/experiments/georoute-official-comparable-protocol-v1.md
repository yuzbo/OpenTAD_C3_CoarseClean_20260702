---
type: experiment
node_id: exp:georoute-official-comparable-protocol-v1
title: "GeoRoute official-comparable protocol v1"
idea: idea:geo-route-adatad
stage: implemented
status: preflight_pending
verdict: PREFLIGHT_BEFORE_F1
confidence: high
commit: pending
jobs: []
updated: 2026-07-31
---

# GeoRoute official-comparable protocol v1

## Current verdict

No further Pro discussion is required. The numerical cause, single permitted
repair, official source anchor, development matrix, selector rule and sealed
test boundary are specified. The next admissible action is remote validation
and F0 deployment from one exact clean commit.

## Parent evidence

The sealed source-`685f935e` no-compression PL/ST gate passed under Jobs
`1207554/1207555/1207556`. Its finalization self/file SHA-256 are
`ad556812454f2ff02161587979ac99c33d9a4983b5c8fcd97d26efe47a936185` /
`f8ef174c934b42ef4efb98e91f16ee4a0a79d0b9f0bbc9c3e174ad3b64bd77e3`.
It authorizes this protocol design only.

## Frozen F0

F0 runs parallel residual-ST and residual-PL 32-real-batch, single-rank
resource/numerical stress leaves plus a two-rank default-FP32-DDP KAT. Seed
2311 is disjoint from official reference seed 42 and development seeds
3407/3408/3409. No leaf may emit a checkpoint, prediction, metric, evaluator,
or official-test artifact. The after-any finalizer authorizes F1 only if all
three leaves pass.

## Frozen F1

F1 is a 5-arm x 3-seed Fit/Gate development matrix:
`dense_native`, `fixed_lattice`, `random`, `residual_st_rep_off`,
`residual_pl_rep_off`; seeds 3407/3408/3409; K=64; 60 epochs; two ranks;
config/global batch 2 and local batch 1; official scheduler 5/100; AMP, EMA and
static graph; no FP16 DDP compression; final EMA checkpoint only.

All 15 leaves are submitted together and sealed by one after-any finalizer.
Native selectors must beat fixed and random at the high-IoU composite for every
seed and cost less than dense for every seed. ST/PL selection additionally
requires strict paired-seed accuracy/cost Pareto dominance. Otherwise the
decision is HOLD. Geometry remains excluded.

## Official-comparability boundary

F1 uses only a Fit/Gate partition of the THUMOS training population and is not
a paper result. A later F2 must reproduce the pinned upstream AdaTAD release,
close current-source and no-compression bridges, run matched native arms over
at least three seeds, measure full decode-to-NMS cost, and open the official
test once after method freeze. Until then all official-test, paper-grade
efficiency, Geometry Zoom and `paper_ready` guards remain false.

## Implementation evidence

Design:
`docs/superpowers/specs/2026-07-31-georoute-official-comparable-protocol-v1-design.md`.
The combined official-protocol, AMP, GeoRoute and required C3 checks pass
`105/105`; shell launchers pass `bash -n`. A wider Windows-only GeoRoute
collection is unavailable because that host's PyTorch `c10.dll` fails to load,
so it is not counted as code evidence and the exact remote Linux suite remains
mandatory. Pre-deployment review also corrected the F0 KAT reservation from
four GPUs to its registered two-GPU allocation and restored AdaTAD's
deterministic-warn-only test semantics.

N16R4 currently reports `70 GiB` free, `MaxSubmitJobs=16`, one unrelated
running job and one unrelated `DependencyNeverSatisfied` job. This is enough
for the conservative four-job F0 (`44 GiB` requirement), but not yet for the
16-submission/15-cell F1 or its conservative `122 GiB` peak. No unrelated job
may be cancelled. F1 therefore remains result- and capacity-gated even if F0
passes. Runtime commit, remote snapshot, jobs and receipts remain pending.
