---
type: experiment
node_id: exp:spatial-zoom-s1-infrastructure
title: "Spatial Zoom S1 infrastructure verification"
stage: tested
outcome: replacement_protocol_implemented_remote_gate_pending
tags: ["offline-tad", "spatial-zoom", "infrastructure", "falsification-gate"]
added: 2026-07-13
updated: 2026-07-15
---

# Spatial Zoom S1 Infrastructure Verification

## Purpose

S1 asks whether spatial resolution contains enough localization headroom to
justify a later sparse spatial zoom/crop method. It keeps the full dense
temporal axis and the official-derived AdaTAD detector path. S1 itself contains
no ROI, scout, teacher, policy, fusion, temporal selection, or new detector.

## Implemented

- Matched dense160/dense224/dense256 configs and a resolved-config drift audit.
- Frozen fit/gate/sealed-test manifest, three seeds, immutable run namespace,
  gate-only checkpoint selection, and one study-level sealed-test marker.
- Exact-2x deterministic 384-to-768 temporal interpolation equivalent to
  PyTorch linear `align_corners=False` without CUDA atomic backward.
- Strict deterministic train/test/profile entrypoints and a real CUDA
  full-model forward, AMP loss, backward, finite-gradient precheck.
- AMP same-batch replay with state restoration and equal successful-update
  exposure, checkpoint sidecars, raw gate evidence, and hash-bound selection.
- Paired Bayesian video-cluster bootstrap with fixed class support,
  hierarchical seed pooling, official evaluator parity, simultaneous max-T,
  short-action and boundary diagnostics.
- Trained-checkpoint-only warm serial full-stack latency, memory, and gross GPU
  energy profiling, with an explicit claim boundary.
- Atomic immutable evidence publication and recoverable test-open certificate.

## Invalidated Prior Matrix

The `35204f5` matrix (`1164291`, `1164307-1164314`) emitted
`upsample_linear1d_backward_out_cuda` nondeterminism warnings in every cell and
was cancelled before completion. It left 222 checkpoints and 222 sidecars with
no Traceback/OOM/non-finite marker. These files are diagnostic only. They may
not be resumed, selected, tested, profiled, or reported as formal S1 evidence.

## Current Verification

- Syntax checks: passed.
- Combined S1/train-iteration tests: `40 passed, 1 skipped` locally. The
  skipped Torch parity check is mandatory on the Linux CUDA precheck node.
- Required C3 regression: `20 passed`.
- Config matrix: PASS; only spatial resolution differs; detector feature
  contract remains `[B,384,768]`.
- Current protocol fingerprint:
  `3dc356baec2d69b8f13fc2096f0df00b5e9e387935bb80bd2a73d3a25037eb0c`.
- Replacement exact-commit CUDA gate, pilot, 3x3 training, sealed test, cost
  profile, final analysis, and GO/KILL: not yet completed.

## Decision Boundary

`tested` here describes local infrastructure only. The route is not
`empirically_supported` or `paper_ready`. S1 KILL permanently blocks S2. S1 GO
only authorizes an oracle ROI/crop sufficiency experiment; it does not prove a
learned zoom method.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
