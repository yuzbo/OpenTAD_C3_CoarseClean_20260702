---
type: experiment
node_id: exp:georoute-real-data-amp-stability-v2
title: "GeoRoute official-semantics real-data AMP stability v2"
idea: idea:geo-route-adatad
stage: designed
status: approved_design_pending_implementation
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

## Paper boundary

This gate can only authorize freezing an official-comparable experiment. Its
losses, scales, skips and gradients are numerical provenance and never paper
performance evidence.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.

