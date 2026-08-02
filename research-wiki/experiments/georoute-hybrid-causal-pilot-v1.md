---
type: experiment
node_id: exp:georoute-hybrid-causal-pilot-v1
title: "GeoRoute Hybrid causal exploratory pilot v1"
stage: experiment_running
status: nine_arm_screen_running_no_performance_result
tags: ["georoute", "hybrid", "causal-ablation", "plackett-luce", "development-only"]
added: 2026-08-02
updated: 2026-08-02
---

# GeoRoute Hybrid Causal Exploratory Pilot v1

## Question

Under matched support-only representation, exact K64, one packed heavy forward,
and one independent exploration seed, does fixed context + ROI + residual
routing show a strict increment over its component families, estimator control,
and geometry-misalignment control?

## Frozen protocol

- study ID: `georoute_hybrid_causal_pilot_v1`;
- seed: 5227, derived by SHA-256 first-eight-hex modulo 10000;
- 20 epochs, development-only Fit/Gate protocol;
- world size 2, global batch 2, local batch 1;
- AMP with scout/likelihood FP32 and default FP32 DDP reduction;
- `fp16_compress=false`, final checkpoint only, no resume;
- exact source 180x320, T384, N220, K64;
- pretrained VideoMAE absolute position on;
- external absolute coordinate, ROI-relative coordinate, geometry projection,
  geometry side channel, geometry regularization, and weighted pooling off;
- detector policy risk keys exactly `cls_loss` and `reg_loss`;
- all nine cells and one common population required before interpretation.

## Arms

`A0` Dense220; `A1` Fixed64; `A2` stateless Random64; `A3` residual-PL64;
`A4` context8+residual56-PL; `A5` context8+ROI56-PL; `A6` matched Hybrid-ST
8/28/28; `A7` sequential Hybrid-PL 8/28/28; `A8` A7 with temporal geometry
trajectory shift127.

## Mechanical gate

Before performance, one no-metric A7 P0 must prove conditional PL likelihood,
private RNG, both branch gradients, strict detector-risk binding, exact roles,
one-heavy-forward, full production horizon, finite AMP update, and support-only
representation. A separate two-rank NCCL KAT must prove default FP32 DDP
reduction/update. Failure writes a terminal P0 receipt and does not create any
performance cell.

## Execution contract

Accuracy/route telemetry and cost timing use separate complete replays. All
nine performance leaves are held until capacity/storage/input/P0 receipts are
sealed, then released together. An after-any finalizer validates stage and
artifact hashes. Missing or invalid cells force
`INCOMPLETE_NO_PERFORMANCE_INFERENCE` with empty contrasts.

## Admission and interpretation

Only a complete single-seed screen can emit descriptive contrasts. Confirmatory
admission requires A7 to exceed A4/A5/A1/A2/A8 on the frozen high-IoU checks,
not lose mAP@0.7 to simple controls, not fall below A6, and have isolated
model+postprocess p50 below Dense. No multiple-comparison-adjusted, official
test, accuracy-preserving, efficiency, mechanism-proof, or paper claim follows.

If a future claim compares PL with ST, the confirmatory protocol must retain a
matched ST arm across seeds; otherwise estimator-superiority language is
deleted.

## Current state

Exact clean runtime `0f64218d8f404ef652934844dcd97a3f9607c580` passed
remote Linux/CUDA pycompile, required C3 tests (`20 passed`), complete GeoRoute
tests (`171 passed, 1 skipped`), and the real data/config binder (SHA-256
`202d8d75b024ae6f080caba461ba05c33edd0790b99d683c9751b4f449f2e78d`).
The earlier `a9e1f742` validation namespace is retained as failed provenance:
one full-suite contract test caught a removed inherited inner-step memory clause;
it produced no model or performance result and was replaced, not reused.

Run root:
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_hybrid_causal_pilot_v1_0f64218d_20260802_111116`.
No-performance P0 Jobs `1213665`, `1213666`, and `1213667` all completed `0:0`.
The sealed suite status is `PASS_MECHANICAL_ONLY`, internal SHA-256
`6e86e8fec966dc9440140e746d3436926ce764f825f7081ad387b24dce9c8e71`,
file SHA-256
`142d0e64f67ddc5c4c72ff41e6cf2d346f4688e101d0116b4402c2c4905d1762`.
It recorded exact `8/28/28` roles, distinct ROI/residual private RNG seeds,
`global_rng_consumed=false`, finite nonzero branch-gradient L2 values
`6.960655212402344/6.96014928817749`, and a passing world-two FP32-DDP KAT.

Pilot deployment receipt SHA-256 is
`6dc3abebdb662393dc6faa8eac4bdf052622c0ea041a2c7eec8f232365c2b3f9`.
Jobs `1213694--1213702` bind A0--A8 respectively; after-any Job `1213703`
finalizes only after every cell is terminal. The study is running. Submission,
finite early detector losses, or a negative signed score-function term are not
performance evidence; checkpoint, prediction, common-population, metric, cost,
and final contrast validation remain pending.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
