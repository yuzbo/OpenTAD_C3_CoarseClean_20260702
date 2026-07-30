---
type: experiment
node_id: exp:georoute-ddp-fp16-cast-repair-gate-v1
title: "GeoRoute DDP FP16-cast no-compression repair gate v1"
idea: idea:geo-route-adatad
stage: tested
status: complete_repair_gate_pass_matched_formal_protocol_freeze_authorized
verdict: DDP_FP16_CAST_REPAIR_GATE_PASS_MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED
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

Exact clean source
`685f935e759d5d78f94e5f208997644e07bf4654` passes local focused
AMP/repair tests `32/32` and the complete remote GeoRoute suite `145/145`.
The same-commit real CUDA/NCCL/DDP KAT ran as Slurm Job `1207542` and completed
`0:0`. It records no compression-hook registration, finite scaled FP32 gradient
maximum `70000`, a nonfinite detached FP16 shadow, finite unscaled gradient,
and a successful optimizer update. Its status is
`PASS_DDP_FP16_CAST_REPAIR_CUDA_KAT_ONLY`; self/file SHA-256 are
`257436d617b79413b4b790cda754d6dec56602d52edb07e50c03cdcd28f78b4f`
/
`d957514816f660a8eb43b922dfb3325baf36f1bbb706f398d0a54cc0a37df3ae`.
The artifact audit found no checkpoint, metric, prediction, evaluator/NMS, or
official-test output.

The fresh gate was then admitted from the same clean commit under
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_ddp_fp16_cast_repair_gate_v1_685f935e_s2307_20260730_2314`.
PL Job `1207554`, ST Job `1207555` and afterany finalizer Job `1207556` all
completed `0:0`. Deployment self/file SHA-256 are
`da8e79727ec4ce758e23d996ac2b238568bee715493dd0e6dec767342e155451`
/
`380b85e781691e2956f978b828ba071ffec4192e0df8acaa7529ada9c281f3e0`.
Admission bound the exact gradient parent, KAT, origin ref, official-reference
config, immutable inputs, `active=2 + additional=3 <= MaxSubmitJobs=16`, and
storage `100456517632` free versus `47244640256` required.

## Terminal evidence

Each arm consumed 64 batches with finite forward losses, 64 optimizer attempts,
zero retry/replay, 64 scheduler advances and 64 EMA advances. PL and ST both
skipped only batches `20/29`, made 62 successful updates, had maximum
consecutive skip `1`, reached minimum/final scale `16384`, and passed the
registered final-16 stable tail. The pair matched its full data sequence and
CPU RNG sequence, initial CUDA RNG, execution seed and immutable inputs. Its
cross-arm skip delta is `0` and final-scale ratio is `1.0`.

PL receipt self/file SHA-256 are
`ee5ffde458d334c1555fb8ae8ba6d77832ff312712a2ce4be5eefb62c9b42a9c`
/
`12d851b69068dd1cf87171ceccbcd7cf431951b5b22c594f868ae347a78be879`;
PL stage self/file SHA-256 are
`a2adf0284e756b10edd0fe785ac788f4d8a0031297791b8f221d532f4f56944a`
/
`652669c6432bd16db5e385d41817a32830cc3a17e6fb15c88af2e2232674016b`.
ST receipt self/file SHA-256 are
`847a1847b8f1599f8ee1c5a931fcb9ec8134a81a7492505e9d4e25aaf19442ab`
/
`f406bf87ed6e1a21e961981b7c2cd71ae94f70443ef4f67a9ae1d5beff8d84ef`;
ST stage self/file SHA-256 are
`efabbd7b5815837a8a21f1d3cd4f6aaab5cb155424a0c7f6d21030b16a6b42a2`
/
`c5fafe6756caeaa691d4b74360bf91533969f78c6c90f426105aa53f06203ec3`.

Finalizer status/decision are
`COMPLETE_DDP_FP16_CAST_REPAIR_GATE_ONLY` /
`DDP_FP16_CAST_REPAIR_GATE_PASS_MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED`;
its self/file SHA-256 are
`ad556812454f2ff02161587979ac99c33d9a4983b5c8fcd97d26efe47a936185`
/
`f8ef174c934b42ef4efb98e91f16ee4a0a79d0b9f0bbc9c3e174ad3b64bd77e3`.
Independent validation recomputed the receipt, stage, deployment, submission
and finalization hashes and reproduced the pair classifier. Slurm and log
audits found no Traceback, OOM, rendezvous or fatal signature. The
255,761,847-byte namespace contains no wrapper failure, `.tmp`, checkpoint,
prediction, metric, evaluator/NMS or official-test artifact.

## Claim boundary

This pass is numerical repair evidence only. It authorizes freezing a separate
matched formal performance protocol, but `official_protocol_freeze_authorized`
remains false because this gate is not an exact official AdaTAD reproduction.
It cannot rank PL/ST, report mAP/cost, select a winner, open official test,
start P2/P3, or support a paper result.
