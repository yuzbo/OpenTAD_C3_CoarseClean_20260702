---
type: experiment
node_id: exp:spatial-zoom-s1-infrastructure
title: "Spatial Zoom S1 infrastructure verification"
stage: experiment_running
outcome: sidecar_recovery_tested_local_long_gate_pending
tags: ["offline-tad", "spatial-zoom", "infrastructure", "falsification-gate"]
added: 2026-07-13
updated: 2026-07-17
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
- Formal replacement commit `18139b930bef6ee234f6220a6adc898eb9c23c0c`
  includes the evaluator-policy fix and its research-memory audit. Exact
  snapshot `opentad_spatial_zoom_s1_18139b9_20260716_ghfast` passed full CUDA
  gate Job `1166358` (`COMPLETED 0:0`) with precheck internal SHA
  `4275cadaf28cc78d548fe220dcfc3496cd3150b668074c560da791958e0838f1`.
  The new canonical namespace is
  `d95a36db4bc70aa2ac9d15e5fb5be82174a8a3488c5150c71d2ad4c10c7234a7`.
- Fresh epoch-0 Jobs `1166361-1166369` cover dense160/224/256 and seeds
  3407/3408/3409. They use normal one-GPU Slurm allocation, no resume, and no
  physical-GPU override. An `afterany:1165781:1165782:1165783` scheduling
  dependency prevents overlap with the final three old diagnostic cells; it
  does not alter the training protocol. No selection, sealed test, profile, or
  S1 GO/KILL is currently authorized.
- Status at `2026-07-16T13:02+08:00`: old diagnostic Jobs
  `1165781-1165783` also terminated `FAILED 1:0`, so the entire `0421a8d9`
  matrix remains protocol-invalid. The dependency released all nine formal
  Jobs `1166361-1166369`, which are now `RUNNING`: dense160 has entered epoch
  11, dense224 epoch 9, and dense256 epochs 7-8. Latest finite losses span
  `0.6482-0.7552`; isolated same-batch AMP retries are at most `2/8` and all
  affected cells continued. There is no Traceback, OOM, non-finite raw loss,
  determinism warning, retry exhaustion, parity failure, checkpoint write
  error, or FAIL. Checkpoint/sidecar/evidence/selection/tmp counts are all zero
  as required before epoch 41, and `/data` retains about 164 GiB free.
- Status at `2026-07-16T15:08+08:00`: all nine formal Jobs remain `RUNNING`.
  Dense160 has entered epoch 34, dense224 epoch 28, and dense256 epochs 24-25.
  Latest finite losses span `0.4371-0.5452`; per-cell AMP retry totals are 2-4
  and maximum same-batch depth remains `2/8`, followed by continued progress.
  The hard-failure scan is empty. Checkpoint, sidecar, evidence, selection, and
  temporary-file counts remain zero before epoch 41. `jobs.tsv` contains the
  expected header plus nine cells, the deployment-summary SHA remains
  `32f693fb391e2fa9777d6263e683210cac28a58ba30688009b60525e398529b0`,
  and `/data` retains about 164 GiB free.
- Status at `2026-07-16T17:09+08:00`: all nine formal Jobs remain `RUNNING`.
  Dense160 has entered epoch 52, dense224 epoch 45, and dense256 epoch 41.
  Latest finite losses span `0.3440-0.4326`; per-cell AMP retry totals remain
  2-4 with maximum same-batch depth `2/8`, and every affected cell continued.
  The hard-failure scan remains empty. Dense160 has six allowed checkpoints,
  metadata files, gate-evidence records, and prediction files per seed for
  epochs 41-51. Dense224 has reached epochs 41-45, with a transient checkpoint
  versus evidence count difference while epoch-45 evaluation is running;
  dense256 has only just entered epoch 41. No selection file exists. The
  approximately 15% values printed by the training monitor use the known
  full-development-GT/40-video-prediction mismatch and are not gate scores;
  only the post-training selector may compute the frozen-corpus gate score.
  `jobs.tsv` and the frozen deployment-summary SHA remain unchanged, and the
  shared filesystem reports 25% aggregate use with no new write error.
- Status at `2026-07-16T19:10+08:00`: dense160 Jobs `1166361-1166363` are
  `COMPLETED 0:0`. Each seed has exactly ten allowed checkpoints, metadata
  sidecars, gate-evidence records, and prediction files; each selector passed
  and selected epoch 59 after 4,800 successful updates. Selected gate Avg-mAP
  values are `64.739055`, `64.842109`, and `63.078053` for seeds
  3407/3408/3409. Their mAP@0.6/0.7 pairs are
  `58.068004/46.172643`, `58.669883/45.247624`, and
  `56.636446/46.260109`. All three selection records retain
  `official_test_read=false` and `paper_claim_allowed=false`. Dense224 remains
  `RUNNING` at epochs 58-59 with nine complete candidate artifact sets per
  seed; dense256 remains `RUNNING` at epochs 52-53 with six complete sets per
  seed, plus one newly written checkpoint awaiting its matching evaluation in
  seed 3408. Retry totals remain 3-4 with maximum same-batch depth `2/8`; the
  hard-failure scan and temporary-file count remain zero. No sealed test or
  profile is authorized until all nine cells complete and validate.
- Status at `2026-07-16T19:45+08:00`: dense224 Jobs `1166364-1166366` also
  completed `0:0`, each with exactly ten checkpoint/metadata/evidence/
  prediction sets and a valid selector record. Seeds 3407/3408/3409 selected
  epochs 57/47/49 with gate Avg-mAP `65.695322/63.205058/63.783346`; their
  mAP@0.6/0.7 pairs are `59.705184/48.593286`,
  `56.315001/45.085447`, and `56.947290/46.056919`. All six completed
  dense160/dense224 selections retain `official_test_read=false` and
  `paper_claim_allowed=false`. Dense256 remains `RUNNING` at epochs 56-57
  with eight complete candidate sets per seed. The hard-failure and temporary
  file scans remain empty, the deployment-summary SHA remains frozen, and no
  sealed-test certificate or profile result exists.
- Status at `2026-07-16T21:28+08:00`: all nine formal Jobs
  `1166361-1166369` completed `0:0`. Every cell has exactly ten frozen
  checkpoint/sidecar/gate-evidence/prediction sets, a valid gate-only
  `checkpoint_selection.json`, closed successful-update exposure, and no hard
  failure or temporary checkpoint. The complete selected gate results are:

  | resolution | seed | epoch | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 160 | 3407 | 59 | 64.739055 | 79.497449 | 73.172632 | 66.784546 | 58.068004 | 46.172643 |
  | 160 | 3408 | 59 | 64.842109 | 79.585664 | 73.841392 | 66.865981 | 58.669883 | 45.247624 |
  | 160 | 3409 | 59 | 63.078053 | 77.731175 | 70.467999 | 64.294536 | 56.636446 | 46.260109 |
  | 224 | 3407 | 57 | 65.695322 | 80.917640 | 72.782082 | 66.478416 | 59.705184 | 48.593286 |
  | 224 | 3408 | 47 | 63.205058 | 78.116034 | 72.844057 | 63.664751 | 56.315001 | 45.085447 |
  | 224 | 3409 | 49 | 63.783346 | 78.745612 | 72.042296 | 65.124614 | 56.947290 | 46.056919 |
  | 256 | 3407 | 51 | 65.184669 | 80.253653 | 74.011853 | 67.580054 | 59.267475 | 44.810311 |
  | 256 | 3408 | 57 | 63.316455 | 78.964777 | 71.383421 | 63.814347 | 56.706985 | 45.712747 |
  | 256 | 3409 | 51 | 64.255928 | 78.063761 | 72.910590 | 67.195584 | 57.436628 | 45.673079 |

  These are development gate scores used only for checkpoint selection. They
  are not official sealed-test scores and do not authorize S1 GO/KILL.
- After all nine selections validated, exactly one global sealed-test opening
  was published. The certificate file SHA-256 is
  `a6d1bf973e3b55c20e30c8e99521a0317219e579df90c5bf61564d3f436a3c57`,
  its internal certificate SHA-256 is
  `8627866a3dfed48a7ddab8df9cb6276d5710e4530c7d8089f929470b0f42f040`,
  and the global marker file SHA-256 is
  `9cf603afa1f2794e2f3b84958eb6a23e9acffaf73d4b89767e8caafbae9bb646`.
  A long checkpoint-hash operation outlived two client calls and briefly left
  three builder processes; the two duplicates were terminated before they
  could publish, and the oldest process alone produced the one canonical
  certificate/marker pair. No official test evidence had been read at that
  point.
- Initial post-processing Job `1167230` failed closed in nine seconds before
  any test read or profile marker. Root-cause Job `1167232` showed that the
  Slurm cgroup exposes the allocated GPU to `nvidia-smi` as local index `0`,
  while the profiler passed the host physical ID from `SLURM_JOB_GPUS`.
  Read-only adapter diagnostic Job `1167238` passed both the frozen eight-field
  identity query and 20 ms power sampling. Full first-cell preflight Job
  `1167239` then passed against the real manifest, certificate, checkpoint, and
  clean `18139b9` snapshot, without creating a test artifact or profile marker.
  The adapter SHA-256 is
  `2693cac2aaa7572045f9c69321e57944f3a09d8e5bb68227724cb83f2047888e`;
  it only maps the Slurm physical ID to the cgroup-local selector and does not
  override `CUDA_VISIBLE_DEVICES`, select a physical GPU, or modify model code.
- Remediation Job `1167257` is queued under one Slurm allocation and will run
  all nine official tests and trained-checkpoint profiles serially in the
  frozen order. This construction fixes node, physical GPU, software, and
  profile order across the 3x3 cost matrix. Its immutable submission receipt is
  `post_test_profile_resubmission_r1.json`. Official test/profile results,
  final statistics, Pro review, and the unique S1 GO/KILL decision remain
  pending; `paper_claim_allowed` remains false.
- Status at `2026-07-16T22:55+08:00`: remediation Job `1167257` is `RUNNING`
  on one Slurm-assigned RTX 4090 allocation. Frozen-order cell zero,
  dense256/seed3408 at selected epoch 57, completed its official 211-video
  sealed test with raw Avg-mAP `67.09` and mAP@0.3/0.4/0.5/0.6/0.7
  `82.14/77.76/70.36/59.53/45.67`. Its canonical test-evidence file SHA-256
  is `10c0182d6fae42f37dec108988f22fbfd732725fc270426121ff2608837261e9`;
  prediction SHA-256 is
  `d4b6df44b0be9c9f735ef233dd39a9b28ad487ebfd6d530383285d9de7269194`.
  The evidence has `official_test_read=true` and
  `paper_claim_allowed=false`. The cell's trained-checkpoint profile is still
  running: the matrix currently has one test-evidence file and zero profile
  markers, summaries, or run descriptors. No hard-failure signature is
  present. This single-cell raw result must not be used to choose a resolution
  or issue S1 GO/KILL before the frozen 3x3 test/profile matrix and paired
  analysis close.
- Status at `2026-07-17T01:12+08:00`: Job `1167257` ended `FAILED 1:0` after
  `01:55:03`, inside the first cell's cost-summary validation. No second
  official test started. Dense256/seed3408 retains its valid sealed-test
  evidence and immutable profile-attempt marker, but no profile summary,
  samples, power trace, or run descriptor was published. The exact error is
  `formal S1 profile window identities must be unique`. A read-only replay of
  the inherited test-window construction found 792 loader exposures and 791
  physical `(video,start)` identities. The only duplicate is
  `video_test_0001431:7680`, emitted twice with the same `[1920,2688)` snippet
  interval because `snippet_num=2688` is exactly divisible by the 384-point
  window stride and the official tail-window branch repeats the preceding
  full window. The official test path consumes this duplicate; removing it
  would change the evaluated workload. Therefore the profiler's physical-
  identity uniqueness assertion is the defect. This is not a training,
  prediction, mAP, OOM, non-finite, or storage failure. The failed marker must
  not be deleted and the Job must not be silently rerun. An audited recovery
  must distinguish deterministic loader exposure identity from physical
  window identity, preserve all 792 exposures, and record the failed attempt
  before any new profile campaign is authorized. No cost claim, resolution
  selection, S1 GO/KILL, S2, or Pro review is authorized.
- Recovery implementation status at `2026-07-17T03:40+08:00`: the profiler
  now gives every dataloader exposure a unique ordinal-bound ID while retaining
  the repeated physical `(video,start)` ID separately. Formal validation keeps
  all 792 exposures, requires the measured 791-physical-window topology and
  exact duplicate `video_test_0001431:7680`, and records both manifest hashes,
  duplicate count, and maximum multiplicity. The old v4 marker remains
  immutable. A new self-hashed recovery certificate binds Job `1167257`, its
  failure log and marker hashes, training commit `18139b9`, a clean
  post-processing-only repair commit, and an independent campaign namespace.
  Its Git-diff allowlist rejects any `opentad/`, config, model, data, or
  checkpoint change. Official tests continue to execute from the clean
  `18139b9` snapshot; profile/descriptor/analysis execute from the certified
  repair snapshot. Existing dense256/seed3408 test evidence is validated and
  reused, never reopened; the remaining tests run once in frozen profile order
  under one Slurm allocation. Local focused verification is `44 passed, 1
  skipped`, including a formal 200-exposure duplicate-physical-window test.
  This is `tested` recovery infrastructure, not a recovered cost matrix: no
  recovery certificate, CUDA gate, replacement post Job, GO/KILL, S2, or Pro
  review has yet been issued.
- Provenance-recovery status at `2026-07-17T04:05+08:00`: clean remote replay of
  the first recovery implementation (`20b84d2`) correctly failed before
  certificate creation because the generic bound-config validator rebuilt the
  historical `18139b9` training binding and full precheck against the repair
  checkout. No test/profile artifact or Job was created. Commit `341cf97`
  separates these identities without weakening the training entrypoint:
  normal binding still uses the current repository, while cross-snapshot
  validation derives the repository root from the recorded audited config,
  requires its exact Git HEAD and clean status, validates the complete
  160/224/256 config matrix there, and rebuilds the original precheck there.
  Tests explicitly accept an exact clean historical repository and reject a
  dirty repository, wrong commit, wrong source path, and precheck commit drift.
  Local S1+C3 verification is `66 passed, 1 skipped`. This remains tested
  recovery infrastructure only; a final clean repair snapshot, immutable
  recovery certificate, Slurm preflight and replacement post Job are pending.
- Recovery-gate status at `2026-07-17T04:42+08:00`: final repair snapshot
  `abd1effac5376a7cf13d1111193f22992ec66d25` passed remote focused tests
  (`67 passed`) and issued campaign `bb56f9d0283b12c0`, recovery certificate
  SHA `1a0bc133d5006f31409ce9ea86a8ee70cc1e275ceef205d21aa7c9cb3334004f`.
  Gate Job `1167497` failed in one second before Python/CUDA because its batch
  script enabled `set -u` before sourcing `/etc/profile`; that script and log
  are preserved. Gate Job `1167500` then ran for `18:55`, read about 249.5 GB,
  and successfully traversed the nine historical bindings/checkpoint hashes.
  Its real first-cell preflight failed with `FileNotFoundError` for
  `data/thumos-14/annotations/thumos_14_anno.json`: the clean repair clone does
  not contain the ignored data mount owned by the clean training snapshot.
  It did not open a test or profile and published only a failed temporary JSON.
  Adding a hidden symlink would violate clean-checkout provenance, so the
  launcher now runs repair Python from the historical training working
  directory with explicit repair-root `PYTHONPATH`. A new
  `SPATIAL_ZOOM_S1_PREFLIGHT_ONLY=1` path exits after validating existing test
  evidence and topology, before any new test/profile. Local S1+C3 regression is
  `66 passed, 1 skipped`. Because this changes the certified launcher, the
  `bb56...` campaign is retained as failed infrastructure and cannot authorize
  the matrix; a new commit, certificate and GPU gate are required.
- Recovery-gate status at `2026-07-17T04:50+08:00`: commit `2d988b2` passed the
  full local S1+C3 regression (`66 passed, 1 skipped`) and remote focused suite
  (`67 passed`). Recovery campaign `10105b8b590cd7fc` issued certificate SHA
  `0f02a64b3150e97a8a172de75af756b677922085ae21ca8b4f48a2e654b7bdf0`.
  Its no-open Gate Job `1167504` failed `127:0` in one second, before any Python
  preflight, CUDA work, test read, or profile marker, because the formal test
  launcher used `python` to parse the certificate before sourcing the OpenTAD
  environment. The campaign retains the Slurm logs and self-hashed submission
  receipt (`80c230fe...`). The launcher now activates CUDA/Miniforge/OpenTAD
  before its first Python call, with a static ordering regression test. Because
  this changes a certificate-bound launcher, campaign `10105...` cannot
  authorize the matrix; another clean commit, recovery certificate, and GPU
  gate are required. S1 remains `experiment_running`, with no cost matrix,
  GO/KILL, S2, or Pro review.
- Recovery-gate status at `2026-07-17T05:03+08:00`: clean snapshot `04111ad`
  passed the remote focused suite (`67 passed`) and produced campaign
  `e647d6feff89cfd7`, certificate SHA
  `b76fa4afb9917452928612a9eeba38daa7152212eaf5efc63c9cd0a53fb766fc`.
  Gate `1167507` activated the OpenTAD environment correctly but failed `2:0`
  after three seconds, still before the Python preflight or any sealed-test or
  profile access. Direct `sbatch` copies its script to `/var/spool/slurmd`, so
  `BASH_SOURCE` incorrectly resolved the profile code root to the spool
  directory. This would also affect the recovery-matrix launcher. Rather than
  hide it in an unbound wrapper, both formal launchers now require an explicit
  `SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT` and validate that checkout against the
  certificate's `profile_code_commit` and clean Git state. Campaign `e647...`
  remains failed infrastructure and cannot authorize a matrix. A new clean
  commit, certificate, and no-open GPU gate are required; S1 stays
  `experiment_running` with no GO/KILL, S2, or Pro review.
- Recovery-matrix status at `2026-07-17T05:32+08:00`: commit
  `04f8c28c85f333ea9b992c1e5bc4fade06f2fe06` passed the remote focused suite
  (`67 passed`) and issued campaign `bc9bacf31bae3749`, certificate SHA
  `77caf621f2c453fc90a627189727dde590a586134d1279be2f95b8b836e7d093`.
  Its explicit-root no-open Gate `1167512` completed `0:0` in `17:09` on
  Slurm-assigned `g0024`. Raw gate output reports 211 videos, 792 loader
  exposures, 791 physical windows, exact duplicate topology, matching
  hardware/software fingerprints, and validated reuse of dense256/seed3408's
  existing test evidence; it published no profile or descriptor. Gate receipt
  SHA is `820a9721...`. Exactly one serial same-allocation recovery matrix,
  Job `1167516`, was then submitted in the frozen nine-cell order with no
  physical GPU override; matrix receipt SHA is `273bb526...`. It is currently
  `RUNNING` on `g0024`. This authorizes only completion of the sealed test and
  trained-checkpoint profile matrix. S1 remains `experiment_running`; cost
  analysis, paired statistics, GO/KILL, S2, and Pro review remain blocked.
- Power-evidence failure status at `2026-07-17T07:39+08:00`: recovery matrix
  Job `1167516` ended `FAILED 1:0` after `01:17:22` on `g0024`, during the
  first frozen cell dense256/seed3408. The full 792-exposure model profile ran,
  but `build_profile_summary` failed with `formal S1 power trace is too sparse
  for auditable energy integration`. The immutable v5 attempt marker remains;
  no profile summary, latency samples, raw power trace, or descriptor was
  published, and no later cell started. Existing dense256/seed3408 sealed-test
  evidence remains valid and was reused rather than reopened. The failure-log
  SHAs are `5d777a11...` (stdout) and `22661dd3...` (stderr); the
  `nvidia-smi` audit SHA is `41079027...`; the marker file SHA is
  `eeac13f9...` with internal marker SHA `cf74a5f2...`. Static inspection found
  that the formal validator permits at most a 100 ms gap while the implementation
  assumes `nvidia-smi --loop-ms=20` yields timestampable 20 ms pipe records;
  because raw samples were intentionally not committed after validation failed,
  the old attempt cannot distinguish one buffered record from gaps over 100 ms.
  This is a cost-evidence infrastructure failure, not model/mAP evidence.
- A test-blind power-cadence diagnostic is now implemented locally. It uses one
  Slurm-local `cuda:0` allocation, reads no annotation/test/checkpoint, applies
  the same ten-second CUDA load to the original persistent `nvidia-smi` sampler
  and an in-process native-NVML candidate, and atomically records every observed
  timestamp plus min/median/p95/max gaps. It keeps the frozen 20 ms target and
  100 ms audit limit unchanged. Local S1 focused verification is `49 passed, 1
  skipped`; remote GPU cadence evidence is still pending. No formal profile
  backend switch or replacement matrix is authorized until that diagnostic
  proves a backend satisfies the unchanged threshold.
- Power-cadence diagnostic Job `1167536` completed `0:0` in 30 seconds on a
  Slurm-assigned `g0059` GPU, using clean commit `7e75b43`. Under matched
  ten-second CUDA loads, inherited `nvidia-smi-persistent-loop-ms` produced 767
  records with median/P95 gaps `20.212/20.331` ms but burst delivery with
  minimum `0.002` ms and maximum `678.458` ms, so it failed the unchanged
  100 ms audit. Native `nvml-persistent-poll-v1` produced 511 records with
  median/P95/max gaps `20.000/20.018/57.709` ms and passed. The diagnostic
  reads no test data, has `paper_claim_allowed=false`, and its file/internal
  SHAs are `14c12730.../596568ed...`; submission receipt SHA is `e32675c3...`.
  This proves the old failure was pipe-arrival buffering rather than an
  infeasible formal cadence. The formal profiler now resolves the native NVML
  device by the frozen Slurm-allocated GPU UUID, rather than conflating the
  `cuda:0` logical index with a node-physical NVML index, while retaining 20 ms
  sampling and the 100 ms gap limit. A versioned chained certificate must bind
  the original identity failure,
  failed campaign `bc9...`, Job `1167516` marker/log, and this diagnostic before
  any new no-open gate or profile matrix is authorized. The combined local
  S1+C3 regression for this implementation is `71 passed, 1 skipped`; the skip
  remains the Linux/CUDA-only interpolation check. Remote exact-snapshot tests,
  real chained-certificate validation, and a no-open GPU gate are still
  required.
- Formal NVML commit `2f8eb06f98ce35b61ce78b2b0cffa3eeb27a1b22`
  passed `72` exact remote Linux tests in clean snapshot
  `opentad_spatial_zoom_s1_2f8eb06_20260717_nvml`. Real chained campaign
  `02f8e8bf7c2d6d25` recursively validated the original v4 failure, parent
  campaign `bc9...`, v5 power failure, Job `1167536` diagnostic and current
  restricted Git diff. Its certificate internal/file SHAs are
  `e70cccc3.../74ba2f55...`. No-open Gate `1167537` completed `0:0` in `20:57`
  on `g0059`, with zero-byte stderr and no profile publication. Crucially,
  Slurm exposed logical `cuda:0` while assigning physical GPU 1; UUID-resolved
  NVML still passed with 510 samples and median/P95/max gaps
  `20.000/20.025/57.848` ms, while the inherited pipe again failed at
  `674.014` ms. Gate diagnostic internal/file SHAs are
  `037992ca.../e271056e...`; gate receipt internal/file SHAs are
  `a20341be.../8f6f54c8...`. Exactly one serial frozen-order matrix, Job
  `1167538`, was submitted with receipt internal/file SHAs
  `a20768d5.../bacf8b0f...`; it is pending by Slurm priority and has not opened
  a new cell. S1 remains `experiment_running`; no cost matrix, paired result,
  GO/KILL, S2 or Pro review exists yet.
- Full-profile NVML failure at `2026-07-17T10:03+08:00`: the only authorized
  serial matrix Job `1167538` ended `FAILED 1:0` after `01:12:01` on `g0059`,
  in the first frozen dense256/seed3408 cell. The official test evidence was
  reused rather than reopened. The detector/profile completed the measured
  path and collected `107147` native-NVML samples, but the formal validator
  observed one `2413.519286` ms gap against the unchanged `100` ms limit.
  Fail-closed publication worked: there is one started marker and zero
  completed markers, profile summaries, latency traces, raw power traces, or
  descriptors; none of the later eight cells started. Slurm reported
  `63687456K` MaxRSS under a `62200M` allocation. The matrix stdout/stderr,
  nvidia-smi audit, and started-marker file SHAs are respectively
  `f82d2a9d...`, `bce50032...`, `300c4a3b...`, and `c9692531...`; the internal
  marker SHA is `1851fe1d...`.
- This failure falsifies the sufficiency of the short cadence Gate, not the S1
  model or its existing mAP. The formal native-NVML sampler is still a Python
  `threading.Thread` in the same process that accumulates detector outputs and
  executes official finalization. The prior ten-second synthetic Gate
  established correct UUID resolution and typical cadence, but could not
  certify GIL/CPU scheduling under the full high-RSS inference/NMS path. The
  most likely code-level failure mode is long-tail sampler starvation inside
  the profiled process; the evidence does not justify blaming NVML itself or
  weakening the 100 ms contract. No replacement matrix is authorized until a
  separate minimal sampler process, atomic preservation of failed raw cadence,
  and a representative long-duration no-open stress Gate are implemented,
  tested, committed, and audited. Pro, S2, and GO/KILL remain blocked.
- Out-of-process recovery implementation status: the current branch replaces
  the formal in-process sampler with a minimal UUID-bound native-NVML process
  pinned to a fifth Slurm CPU while the detector is pinned to four CPUs. It
  records `time.monotonic_ns` samples in node-local scratch, then atomically
  seals the raw trace and self-hashed attempt report; launcher salvage also
  seals a distinct parent-failure record if the detector fails after the
  sidecar attempt was already finalized. Recursive v3 recovery binds Job
  `1167538`, the prior v2 certificate, the restricted repair diff, the 20 ms
  target/100 ms limit, exact 4+1 CPU topology, and the representative Gate.
  The Gate reuses dense256/seed3408's existing official-test evidence, executes
  all 792 loader exposures plus official finalization, and is forbidden from
  publishing a profile, samples, power table, prediction, or descriptor.
  Formal cells recursively validate the Gate and bind attempt/Gate hashes into
  profile, descriptor, and analyzer evidence. An independent max-level
  implementation audit returned `HOLD` with no P0 and five P1 findings:
  partial report/trace recovery, Gate runtime binding, report-trace closure in
  descriptor/analyzer, concurrent matrix submission, and absence of a true
  child-process lifecycle test. The branch now addresses all five: a shared
  pair validator is used end to end; Gate UUID plus matrix hardware/software
  class semantics are explicit; trace-only/report-only salvage is immutable;
  the matrix has a persistent atomic lock and self-hashed receipts; and a Linux
  subprocess lifecycle test is included. Local S1 verification is
  `61 passed, 4 skipped`; three skips are Linux-only real child
  lifecycle/failure cases and must execute remotely. The required AGENTS C3
  regression is `20 passed`. This is `tested_local`, not a remote Gate result.
  No replacement matrix, Pro review, GO/KILL, S2, or learned crop is authorized
  yet.
- Remote process-test closure and resource-scope status: clean commit
  `35c7c5f6fdf2a85c7ecbadc2249f83476b7cdc3e` passed the exact combined
  Linux S1+C3 suite with `85 passed`. No formal Gate was submitted from that
  commit. A first formal request for one GPU, five CPUs, and 96 GB was rejected
  by the N16R4 site rule that caps memory per requested GPU at 55 GB. Read-only
  resource diagnostics then established a compliant two-level scope:
  Jobs `1168504` and `1168506` showed that an outer two-GPU/eight-CPU job
  receives `124400M`, while an exact inner step receives one GPU, five CPUs,
  and `96000M`; Jobs `1168509-1168510` located the tightest finite cgroup v2
  limit at exactly 96,000 MiB even though the leaf reports `max` and
  `SLURM_MEM_PER_NODE` is absent. Job `1168508` failed only because its
  resource-only diagnostic invoked an unavailable `python` command and
  produced no research evidence.
- The current implementation now prefers `SLURM_STEP_GPUS`, verifies one
  Slurm-visible GPU and the tightest finite cgroup/Slurm memory limit, and
  re-executes Gate, cell, or matrix launchers into one exact
  `--gpus=1 --cpus-per-task=5 --mem=96000M` step when the outer reservation
  contains two GPUs. It never overwrites `CUDA_VISIBLE_DEVICES`. All nine
  matrix cells remain inside one such step. The idle outer GPU is explicitly
  classified as N16R4 scheduling overhead, not model compute. Local S1
  verification for this uncommitted resource repair is `62 passed, 4 skipped`;
  a clean commit, exact remote tests, a new v3 certificate, and the full
  no-open Gate remain mandatory.

## Decision Boundary

`experiment_running` describes the formal 3x3 matrix, not a positive result. The route is not
`empirically_supported` or `paper_ready`. S1 KILL permanently blocks S2. S1 GO
only authorizes an oracle ROI/crop sufficiency experiment; it does not prove a
learned zoom method.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
