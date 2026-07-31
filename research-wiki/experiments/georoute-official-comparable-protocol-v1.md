---
type: experiment
node_id: exp:georoute-official-comparable-protocol-v1
title: "GeoRoute official-comparable protocol v1"
idea: idea:geo-route-adatad
stage: experiment_running
status: f0_pass_f1_authorized_capacity_storage_hold
verdict: F0_PASS_F1_AUTHORIZED_WAIT_CAPACITY_STORAGE
confidence: high
commit: 3d8c2b487fa983d6d6240b347177cc423a37748b
jobs: [1209309, 1209310, 1209311, 1209312]
updated: 2026-07-31
---

# GeoRoute official-comparable protocol v1

## Current verdict

No further Pro discussion is required. Exact source `3d8c2b48` passed the
remote Linux focused suite `25/25` and complete GeoRoute suite `154/154`; its
HEAD, origin ref and tree are exact/clean. The pinned upstream snapshot remains
exact/clean at `01c58b9`.

The first F0 namespace is an execution failure, not numerical or model
evidence. World-two KAT Job `1209274` failed in one second before Python/CUDA
because its outer allocation did not bind memory while the inner `srun`
requested `192000M`:
`Unable to create step ... Memory required by task is not available`.
PL/ST Jobs `1209272/1209273` both completed their own 32-batch contracts, with
one default-GradScaler skip each and matched data-order/scale telemetry. These
leaves are terminal provenance only and cannot compensate for the missing KAT.
After-any finalizer `1209275` sealed
`INCOMPLETE_OFFICIAL_COMPARABLE_PREFLIGHT /
OFFICIAL_COMPARABLE_PREFLIGHT_HOLD`; its internal/file SHA-256 are
`f6da6db381260c40e6f90a07203e1eb1c38c50182cfda5b4e3edb0f52ec55cef` /
`72a910b6ad2d79f895462cc8b0d6dc8c34e85774a810836b76713688e9387ca7`.
No resume or same-namespace replacement is allowed.

The resource-only replacement requests two GPUs once at the outer job and lets
the inner KAT step inherit N16R4's site-bound memory. The site's submit Lua
assigns 55 GB per GPU and rejects every explicit `--mem` override; test-only
probes confirmed that an omitted memory override is the accepted contract. The
replacement changes no model, data, seed, threshold or claim rule and must use
a new exact commit and namespace.

That replacement F0 is now terminally complete under run root
`georoute_official_comparable_preflight_v1_3d8c2b48_20260731_122316`.
PL/ST/KAT/finalizer Jobs `1209309/1209310/1209311/1209312` all completed
`0:0`. PL and ST each consumed 32 matched real-data batches, each recorded one
default-GradScaler skip, each finished at scale `32768`, and each passed a
16-batch stable tail. Their 32 data fingerprints and scale traces match. The
world-two KAT passed default NCCL FP32 reduction/update while its detached
FP16 shadow overflowed as required. Finalizer decision is
`PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY /
FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`; finalization internal/file SHA-256 are
`313da95faeae9e600965fe4ac5c7ad5816f652d5ff2c97cf9734f7028d888a3c` /
`22f5dcab4c19d843bc807c5dd60e5f97605378617f67d3a3f507a7a768c57679`.
No checkpoint, prediction, metric, evaluator, temporary payload or official
test exists. This authorizes F1 execution only and is not model-performance
evidence.

F1 is not submitted yet. The unchanged all-at-once admission gate observes two
unrelated active submissions, so `active=2 + required=16 > MaxSubmitJobs=16`.
Its unchanged training storage gate observes `31,646,543,872` free versus
`130,996,502,528` required bytes. No unrelated job is cancelled, the matrix is
not split, and neither gate is relaxed.

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
so it is not counted as code evidence. Exact clean source `4a03339b` closed
that gap on remote Linux with the complete GeoRoute suite at `153/153`.
Pre-deployment review also corrected the F0 KAT reservation from four GPUs to
its registered two-GPU allocation and restored AdaTAD's
deterministic-warn-only test semantics.

The generic training storage gate initially blocked replacement source
`2156811f` before namespace creation: `45,628,272,640` free versus
`47,244,640,256` required. Authorized checkpoint retention then removed ten
earlier RIME epochs (5,176,692,497 bytes) only after SHA-256 dry-run and CPU
load validation, retaining and postvalidating both epoch-59 checkpoints.
Postverify receipt SHA-256 is
`ad5396edb82e1724d89979a5d495d485a799299b1b81c63a7f9566ac87deafed`.
Because F0 contractually forbids all large artifacts and the sealed first F0
occupied only about 129 MB, source `3d8c2b48` introduced a fixed no-artifact
F0 storage contract: 512 MiB per leaf, 1 GiB fixed overhead and a 24-GiB
filesystem reserve (26 GiB total). This does not change F1's conservative
training storage contract.
