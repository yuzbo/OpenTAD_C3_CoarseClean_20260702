---
type: experiment
node_id: exp:spatial-zoom-s1-infrastructure
title: "Spatial Zoom S1 infrastructure verification"
stage: experiment_running
outcome: strict_cuda_gate_passed_formal_3x3_training_queued
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
- Combined S1/train-iteration tests: `43 passed, 1 skipped` locally. The
  skipped Torch parity check is mandatory on the Linux CUDA precheck node.
- Required C3 regression: `20 passed`.
- Config matrix: PASS; only spatial resolution differs; detector feature
  contract remains `[B,384,768]`.
- Current protocol fingerprint:
  `3dc356baec2d69b8f13fc2096f0df00b5e9e387935bb80bd2a73d3a25037eb0c`.
- Exact snapshot `opentad_spatial_zoom_s1_64e71dd_20260715_ghfast` passed all
  `41` Linux tests and reached a real pretrained full-model AMP backward in
  Slurm Job `1165648`. It then failed closed because the two trainable
  `backbone.model.backbone.fc_norm.{weight,bias}` tensors had no gradient.
- Source audit confirms these are classification-pretraining mean-pooling norm
  parameters. Every S1 config sets `return_feat_map=True`, so the official-derived
  VideoMAE path returns a dense feature map before `fc_norm`; backbone adapters,
  projection, and detection head did receive gradients. This is a gate-contract
  issue, not detector failure and not performance evidence.
- The replacement contract uses an exact two-name expected-unused allowlist,
  separately counts gradient-required tensors, and fails if either allowlisted
  parameter is unexpectedly used or any additional trainable parameter is
  disconnected. Local S1/train-iteration tests are now `43 passed, 1 skipped`.
- Fix commit `47842427eb373fb1f440b1661971a6a231a95f67` closes the exact
  expected-unused contract and component/global count conservation. Independent
  `gpt-5.6-sol/max` review returned `PASS_FOR_REMOTE_GATE`; its count-closure P2
  was fixed and re-reviewed as `P2_CLOSED`.
- Exact snapshot `opentad_spatial_zoom_s1_4784242_20260715_ghfast` and Slurm
  Job `1165667` passed the full CUDA gate. For every resolution, 339 tensors are
  trainable, the exact two `fc_norm` tensors are audited unused, all remaining
  337 gradients are finite, and backbone/projection/rpn_head have nonzero grads.
- Fresh suite root is
  `spatial_zoom_s1_4784242_20260715_2245`; canonical experiment namespace is
  `695803b687bf52197847e8b7fbf3d802c968d13070c660138f524ed31548f3a7`.
- Formal 3x3 jobs are `1165669-1165677` for dense160/224/256 and seeds
  3407/3408/3409. They are queued/running under normal one-GPU Slurm allocation
  without a physical-GPU override. Sealed test, cost profile, final analysis,
  and GO/KILL remain incomplete.

## Decision Boundary

`experiment_running` describes the formal 3x3 matrix, not a positive result. The route is not
`empirically_supported` or `paper_ready`. S1 KILL permanently blocks S2. S1 GO
only authorizes an oracle ROI/crop sufficiency experiment; it does not prove a
learned zoom method.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
