---
type: experiment
node_id: exp:spatial-zoom-s1-infrastructure
title: "Spatial Zoom S1 infrastructure verification"
stage: tested
outcome: formal_matrix_cancelled_before_completion_pending_protocol_fix
tags: ["offline-tad", "spatial-zoom", "infrastructure", "falsification-gate"]
added: 2026-07-13
---

# Spatial Zoom S1 Infrastructure Verification

## Purpose

Implement the authorized S1 falsification infrastructure without implementing
ROI, scout, teacher, policy, fusion, a new detector, or S2.

## Implemented

- Matched dense160/dense224/dense256 AdaTAD configs.
- Resolved-config drift validator and fault-injection tests.
- Deterministic fit/gate/sealed-test manifest, seeds, hashes, duration bins, and
  OpenTAD-compatible block lists.
- Static, real-clip, and full-768-window precheck modes.
- Full precheck certificate binding pretrained-weight, interpolation, shape,
  CUDA-memory, commit, and self-hash evidence; static/clip cannot unlock train.
- Manifest/precheck/commit-bound formal training configs and fail-closed
  training/test entrypoints.
- Immutable checkpoint metadata/sidecars, complete gate evidence, full
  eligible-epoch checkpoint selection, 3x3 test-open certificate, and an
  exclusive study-level sealed-test marker shared across commits, prechecks,
  and experiment namespaces.
- Frozen 3x3 profile order with hardware/software identity preflight before
  each sealed-test open.
- Trained-checkpoint-only full-stack latency, memory, and GPU-energy profiler,
  including decode, H2D, final cross-window NMS, canonical complete-test
  samples, and one canonical output path.
- Gate-only checkpoint selector with raw-prediction recomputation and an
  immutable selection proof; run descriptors cannot accept a hand-entered
  epoch.
- Immutable run descriptor and raw-prediction evidence binder.
- Full AP recomputation with official-evaluator parity, GT-conditioned short
  action bins, paired video-cluster and training-seed hierarchical bootstrap,
  one-sided max-T correction, boundary error, and cost-aware resolution freeze.

## Verification

- Current deployment commit:
  `35204f58fd3e91d7cf8f5888928a41e9bf6c2e72` on
  `codex/spatial-zoom-s1-formal-20260715`.
- Current focused regression: `36 passed` for S1/train-engine and `20 passed`
  for the required C3 checks; compilation and both Slurm launchers' shell
  syntax passed.
- Config protocol fingerprint:
  `343df802fddc5658b97fda4a917c8e8576a0a2801f7ddd77a58a3a462feec2c0`.
- Static runtime grids: 10x10, 14x14, and 16x16; expected detector feature
  contract remains `[B,384,768]`.
- Local real clip: blocked by Windows `c10.dll` initialization.
- Formal CUDA `--mode full` Job `1164289` completed with `PASS`; Slurm chose
  physical GPU 4 while the process used logical `cuda:0`. The precheck file
  SHA-256 is
  `b82d1de687e2c35b59b009eeec352d08b18184e848da451d1fe59447557d1ff5`.
- The canonical experiment namespace is
  `a5253eefbf9b066cfa6bda955c120b76ee8343a8d75517c6df5b784c3dcac2b8`.
- Independent `gpt-5.6-sol`/`max` review iterated through every P0/P1/P2
  blocker. The final review passed state-exact AMP replay and kernel-assigned
  c10d rendezvous ports, then separately passed bound-config immutability for
  checkpoint and gate evidence. This is code-readiness evidence only.

## 2026-07-15 Formal Deployment

- Commit `911448a` removed S1's fixed physical-GPU-1 rule and renamed the
  formal launchers to `*_slurm.sh`. Formal jobs request one generic Slurm GPU
  and consume only the allocation's logical `cuda:0`.
- The first `911448a` matrix (Jobs `1164198-1164206`) is invalid deployment
  evidence: eight jobs stopped on their first AMP-skipped update and concurrent
  jobs could collide on port 29400.
- Commit `7d1e9cc` added same-batch AMP retry and a first rendezvous repair, but
  Max review found that retry did not restore forward-mutated buffers such as
  `loss_normalizer`. Pilot `1164255` was cancelled before meaningful training.
- Commit `9298c0e` restored RNG plus every model buffer while preserving
  GradScaler backoff, and uses `c10d` with `127.0.0.1:0` for atomic port
  allocation. Pilot Job `1164261` completed two epochs but failed at the first
  checkpoint because optimizer/scheduler had legally mutated the bound runtime
  config before its identity was revalidated. Jobs `1164267-1164274` were
  cancelled before the same deterministic failure. This suite is invalid.
- Commit `35204f5` passes deep copies of optimizer, scheduler, inference, and
  post-processing configs to all runtime-mutating paths, leaving the bound
  config immutable for checkpoint and gate evidence.
- Current pilot Job `1164291` completed two epochs, replayed two AMP skips, and
  produced a validated epoch-1 checkpoint plus sidecar with 160 successful
  updates, 162 optimizer attempts, and commit-bound hashes.
- Current formal training matrix: `1164291` (160/3407), `1164307` (160/3408),
  `1164308` (160/3409), `1164309` (224/3407), `1164310` (224/3408),
  `1164311` (224/3409), `1164312` (256/3407), `1164313` (256/3408), and
  `1164314` (256/3409). All nine entered `RUNNING`; no mAP exists yet.

## Evidence Boundary

This verifies infrastructure and starts the preregistered S1 experiment. There
is still no S1 mAP, trained cost, GO/KILL verdict, or empirical support. S2
stays locked until the complete 3x3 evidence and cost-aware decision exist.

## 2026-07-15 post-deployment Pro audit

At 20:10 +0800, all nine Jobs `1164291/1164307-1164314` remained RUNNING.
There was no fatal, OOM, non-finite collapse, or AMP replay exhaustion; 31
same-batch retries had recovered and 213 checkpoints existed.

However, every cell emitted
`upsample_linear1d_backward_out_cuda` nondeterminism warnings under
`warn_only=True` (221 occurrences in total). The current matrix therefore
cannot satisfy the preregistered strict deterministic-formal claim. It may
finish as diagnostic/candidate evidence, but the sealed test remains closed
and no S1 GO/KILL may be issued. A deterministic interpolation equivalent,
strict-mode gate, full AMP-replay precheck, and an honestly specified bootstrap
must be frozen before a replacement 3x3 run.

## 2026-07-15 Task Correction And Preserved Breakpoint

The active task is Spatial Zoom only: preserve the dense temporal axis and
study spatial resolution, ROI sufficiency, and later learned zoom/crop
allocation before the official AdaTAD detector. DUCA is a separate historical
route and must not be implemented, repaired, or mixed into this experiment.

The nine warning-bearing S1 jobs were cancelled by the agent after incorrectly
combining the DUCA and Spatial Zoom audit queues. This was an execution error,
not a scientific GO/KILL decision. Logs, 222 checkpoints, and 222 metadata
sidecars remain preserved. Their stderr files contain 226 deterministic-kernel
warnings in total at cancellation, with no Traceback, OOM, or non-finite marker.
The last observed train epochs were 56/55/55 for dense160, 47/47/47 for
dense224, and 44/43/43 for dense256. No sealed test, final mAP, cost table, or
S1 verdict exists. These partial artifacts are diagnostic only and cannot be
resumed into formal evidence after the deterministic-protocol repair.

The clean continuation worktree is
`E:/DeskTop/TAD/OpenTAD_SpatialZoom_S1_AuditFix_20260715` on
`codex/spatial-zoom-s1-audit-fix-20260715` at base commit `35204f5`. The next
authorized work is to fix deterministic interpolation, bootstrap validity,
transactional evidence writes, and precheck coverage, then run a fresh exact-
commit CUDA gate and replacement S1 matrix. S2 oracle ROI/crop remains locked
until S1 produces a valid GO result.

## Connections

Maintained only in `research-wiki/graph/edges.jsonl`.
