---
type: experiment
node_id: exp:georoute-hybrid-causal-pilot-v1
title: "GeoRoute Hybrid causal exploratory pilot v1"
stage: implemented
status: p0_pending_no_runtime_result
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

Structured routing, wrapper, telemetry, contract, P0, stage runner, deployers,
all-terminal finalizers, and focused tests are implemented in the working tree.
Pure contract tests pass locally. Windows Torch cannot load `c10.dll`; clean
remote Linux/CUDA tests and P0 are pending. No job, checkpoint, prediction,
metric, or runtime result exists for this study.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
