---
type: experiment
node_id: exp:spatial-zoom-s1-infrastructure
title: "Spatial Zoom S1 infrastructure verification"
stage: experiment_running
outcome: selector_policy_bug_fix_pending_replacement_gate
tags: ["offline-tad", "spatial-zoom", "infrastructure", "falsification-gate"]
added: 2026-07-13
updated: 2026-07-16
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
- Early epoch-0 logs contain finite losses and isolated AMP-overflow attempts.
  These attempts restore RNG/model-buffer state and replay the same batch; they
  do not advance scheduler/EMA. Exhausting eight retries, losing successful
  update parity, or a non-finite raw loss remains fail-closed.
- Status at `2026-07-15T23:03+08:00`: all nine jobs remain `RUNNING`.
  Dense160 seeds have started epoch 2; dense224/256 seeds have started epoch 1.
  Latest logged losses are `1.0760-1.1135`, down from the epoch-0 range of
  approximately `1.43-1.50`. Each run has 2-3 recovered AMP retry attempts;
  there is no Traceback, OOM, non-finite loss, deterministic warning, or FAIL.
  Validation intentionally remains unopened until epoch 40.
- Automation `spatial-zoom-s1-experiment-monitor` is active every two hours.
  It is an idempotent state machine: training health monitoring, selection
  validation, one global sealed-test opening, official test and trained-model
  cost profiling, frozen GO/KILL diagnosis, GitHub evidence update, and only
  then a verified Pro-tier post-result audit. It must stop after the final Pro
  report so test jobs, evidence publication, and reviewer calls cannot repeat.
- Jobs `1165669-1165677` all failed during atomic checkpoint publication when
  the shared `/data` filesystem reached 100% usage. Every root cause is
  `PytorchStreamWriter file write failed`; losses remained finite and there is
  no performance conclusion. The no-resume contract invalidates this matrix.
  The replacement implementation persists only the ten frozen gate-eligible
  checkpoints per cell, removes partial temporary files on every write failure,
  and requires 96 GiB free at launch. A new commit, full CUDA gate, canonical
  namespace, and epoch-0 3x3 rerun are required before S1 can continue.
- The 222 forbidden checkpoint weights from invalid commit `35204f58` were
  removed after their sidecar identities were aggregated and preserved. The
  purge receipt SHA is
  `59f1d9d3499eb3cd105478672805f9a19c15a73c8a747b1249bc7c2372ad9ecf`;
  138,485,542,803 bytes were released without deleting sidecars or diagnostics.
- The storage-failed `47842427` matrix was independently hash-validated before
  removing 151 unusable weights (94,195,092,514 bytes) and nine temporary files
  (134,217,728 bytes). All 151 sidecars, configs, logs, and diagnostics remain.
  The purge receipt file SHA is
  `b5237253eaa8d196957da47d5ebd2c07ae6537596b6e53e1e4348286c88d58d9`;
  free capacity increased to 217 GiB. These artifacts cannot be resumed or used
  for model selection, test opening, profiling, or performance claims.
- Storage-safe commit `0421a8d9f6982a6d4ec1fb590cd108581fa2bb83`
  passed focused local checks (`66 passed, 1 skipped`) and exact snapshot
  `opentad_spatial_zoom_s1_0421a8d_20260716_ghfast` is clean. CUDA gate Job
  `1165774` completed 0:0 with `47` Linux tests, deterministic full-model AMP
  backward, 337 finite required gradients at every resolution, and the exact
  two audited-unused `fc_norm` tensors.
- Fresh suite `spatial_zoom_s1_0421a8d_20260716_0324` binds precheck internal
  SHA `3d30ea5489b2ac7f07785dff94ed057ac420aebdd8762ab6df6c76a2ffb003ea`
  to canonical namespace
  `bf71376e2d57946a3f898d25b7dcc88cfc002549a9ed78656293f1a95316a8f7`.
  Jobs `1165775-1165783` are the only formal replacement cells and all start at
  epoch 0. Initial logs show only recoverable first AMP retries; no performance
  evidence exists before gate-only selection and the single sealed test open.
- Status at `2026-07-16T04:57+08:00`: all nine replacement jobs remain
  `RUNNING`. Dense160 has reached epochs 14-15, dense224 epoch 12, and dense256
  epoch 11. Latest finite logged losses span `0.5830-0.7153`. Per-cell AMP
  retry counts are 2-4 and the maximum same-batch retry is `2/8`; every affected
  cell subsequently advanced. There is no Traceback, OOM, non-finite raw loss,
  determinism warning, exhausted retry, parity failure, checkpoint write error,
  or abnormal exit. Checkpoint/sidecar/evidence/selection counts remain zero as
  required before epoch 41, and `/data` retains 217 GiB free.
- Status at `2026-07-16T06:58+08:00`: all cells remain `RUNNING`; dense160 has
  reached epoch 37, dense224 epoch 31, and dense256 epochs 27-28. Latest finite
  losses span `0.4248-0.5028`. Retry counts and maximum retry depth are unchanged
  at 2-4 and `2/8`, respectively. No hard-failure signature is present. The
  pre-gate checkpoint/sidecar/evidence/selection counts remain zero, as frozen
  by protocol, and free storage remains 217 GiB.
- Status at `2026-07-16T09:05+08:00`: all nine jobs remain `RUNNING`; dense160
  has reached epochs 54-55, dense224 epoch 47, and dense256 epoch 43. Latest
  finite losses span `0.3745-0.4506`. Every cell has 3 recovered AMP attempts
  except dense256/seed3409 with 4; maximum same-batch retry remains `2/8`.
  Checkpoint and sidecar counts are both 37, all checkpoint epochs are members
  of the frozen set `41,43,...,59`, no temporary checkpoint exists, and no
  selection file has been created. Thirty-four gate-evidence records are fully
  published; the three-count lag is explained by currently running evaluations
  immediately after atomic checkpoint publication. Completed 40-video gate
  evaluations currently span `14.85-15.15` Avg-mAP. These are gate-only
  checkpoint-selection diagnostics, not sealed-test results or GO/KILL evidence.
  Sidecars bind commit `0421a8d9`, satisfy successful-update/optimizer-attempt
  parity, and retain `official_test_opened=false`. No Traceback, OOM, non-finite
  raw loss, determinism warning, exhausted retry, parity failure,
  `PytorchStreamWriter`, FAIL, or abnormal Slurm state is present. Free storage
  remains approximately 195 GiB.
- Status at `2026-07-16T11:00+08:00`: dense160 Jobs `1165775-1165777` and
  dense224 Jobs `1165778-1165780` reached epoch 59 with all ten checkpoints,
  sidecars, gate predictions, and 4,800 successful updates, but exited `1:0`
  during post-training checkpoint selection. The common error is
  `prediction contains a non-finite or invalid segment`; no training loss,
  checkpoint, evidence, or Slurm GPU failure occurred. Dense256 Jobs
  `1165781-1165783` remain running and must be allowed to terminate normally.
- Raw inspection found no NaN, Inf, or reversed interval. The rejected rows are
  finite zero-length `[-0.0, 0.0]` proposals created by the official-derived
  clipping path: 75-319 rows among 80,000 predictions per completed gate file.
  OpenTAD's official evaluator retains them as zero-IoU false positives, while
  the S1 `DetectionCorpus` incorrectly required `end > start`. Deleting these
  rows would inflate AP and is forbidden. An in-memory policy probe that allows
  equality while retaining every prediction passed exact per-class official
  evaluator parity. For dense160/seed3407 epoch 59 it recomputed gate-only
  Avg-mAP `64.7391`, mAP@0.6 `58.0680`, and mAP@0.7 `46.1726`.
- Fix commit `cbc63d07` changes only the S1 analysis validation to allow
  `end == start`, still rejects reversed/non-finite predictions, and adds an
  official-parity regression (`41 passed, 1 skipped`; C3 regression `20
  passed`). The approximately 15% metrics printed during training used the
  complete development-subset GT against the 40-video gate predictions and
  are monitoring artifacts, not selection metrics. Under the frozen contract,
  failed terminal Job states do not authorize selection, sealed test, profile,
  GO/KILL, or reuse as the formal matrix; a fresh commit-bound CUDA gate and
  epoch-0 namespace are required.

## Decision Boundary

`experiment_running` describes the formal 3x3 matrix, not a positive result. The route is not
`empirically_supported` or `paper_ready`. S1 KILL permanently blocks S2. S1 GO
only authorizes an oracle ROI/crop sufficiency experiment; it does not prove a
learned zoom method.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
