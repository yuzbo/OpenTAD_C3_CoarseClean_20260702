# Spatial Zoom S1 Infrastructure Contract

## Scope

S1 is a falsification gate for offline TAD spatial-resolution headroom. It is
not a Zoom model and contains no ROI, scout, teacher, routing policy, temporal
selector, or detector replacement.

The matched matrix is `Dense-160`, `Dense-224`, and `Dense-256`. After config
inheritance is resolved, only spatial resize/crop parameters and `work_dir` may
differ. The temporal window, VideoMAE-S backbone, AdaTAD adapters, ActionFormer
projection/head, optimizer, update schedule, evaluator, NMS, and checkpoint
rule remain identical.

## Infrastructure

- Configs: `configs/adatad/thumos/s1_dense*_videomae_s_768x1_adapter.py`
- Config audit: `tools/bata/validate_spatial_zoom_s1.py`
- Split/seed bundle: `tools/bata/build_spatial_zoom_s1_manifest.py`
- Shape gate: `tools/bata/run_spatial_zoom_s1_precheck.py`
- Bound-config builder: `tools/bata/build_spatial_zoom_s1_training_config.py`
- Canonical experiment resolver: `tools/bata/resolve_spatial_zoom_s1_experiment.py`
- Training evidence: `tools/bata/spatial_zoom_s1_training.py` and
  `tools/bata/spatial_zoom_s1_evidence.py`
- Cost profiler: `tools/bata/profile_spatial_zoom_s1.py`
- Pre-test profile-order gate:
  `tools/bata/preflight_spatial_zoom_s1_profile.py`
- Checkpoint selector: `tools/bata/select_spatial_zoom_s1_checkpoint.py`
- Test-open certificate: `tools/bata/build_spatial_zoom_s1_test_open_certificate.py`
- Evidence binder: `tools/bata/build_spatial_zoom_s1_run_descriptor.py`
- Matrix lifecycle evidence: `tools/bata/spatial_zoom_s1_matrix.py`
- Result gate: `tools/bata/analyze_spatial_zoom_s1_results.py`
- Slurm launchers: `scripts/run_spatial_zoom_s1_precheck_slurm.sh`,
  `scripts/run_spatial_zoom_s1_train_slurm.sh`, and
  `scripts/run_spatial_zoom_s1_test_profile_slurm.sh`
- Tests: `tests/test_spatial_zoom_s1_infrastructure.py` and
  `tests/test_spatial_zoom_s1_matrix.py`

## Required Order

1. Freeze the manifest before observing candidate results.
2. Run the strict config validator.
3. Run `--mode full` in a normal one-GPU Slurm allocation for all resolutions.
   Preserve Slurm's `CUDA_VISIBLE_DEVICES` mapping and use logical `cuda:0`.
   The self-hashed certificate
   must match a preregistered pretrained-checkpoint SHA, prove exact loading of
   every non-adapter VideoMAE core parameter, prove the exact interpolation
   call sequence, detector shape, CUDA memory, clean Git commit, and exact
   3-config matrix. It must also execute one real AMP training loss and strict
   deterministic backward for every resolution. Static/clip mode does not
   unlock training.
4. Materialize each training config from the manifest and full-precheck
   certificate. Formal training requires a clean checkout, a one-GPU Slurm
   allocation, one process, deterministic execution, a fresh canonical workdir, no
   resume, and no CLI config override.
5. Train all resolutions with seeds `3407/3408/3409`, frozen fit/gate splits,
   and equal successful optimizer updates. An AMP-overflow attempt restores the
   same batch RNG/model-buffer state and retries without advancing the
   scheduler or EMA; failure to obtain one successful update within eight
   retries fails the run.
6. Persist checkpoints only at the frozen gate-eligible epochs. Pre-gate
   periodic checkpoints cannot participate in selection and must not consume
   formal-matrix storage. The launcher requires at least 96 GiB free before a
   cell starts, and a failed atomic checkpoint transaction must remove temporary
   or partially published artifacts. Save raw gate predictions for every eligible checkpoint, then use the
   checkpoint selector to recompute `(mAP@0.6 + mAP@0.7) / 2` from those
   predictions. The maximum wins and exact ties use the earliest epoch.
7. Validate all nine selections before issuing one test-open certificate. All
   commits, prechecks, and experiment namespaces share one preregistered
   `sealed_study_v1` marker, so an equivalent rerun cannot open test again.
8. Before each test, fail closed on the frozen 3x3 profile order and require
   the same hardware/software fingerprint as completed cells. Then save raw
   test predictions and profile the same selected checkpoint. Profiling reuses
   the official single-rank
   DDP result aggregation/NMS path, the complete test loader, batch size 1,
   zero workers, 50 warmups, and a UUID-bound native-NVML sidecar process at
   the frozen 20 ms interval. The Slurm allocation exposes exactly five CPUs:
   four remain bound to the detector and one is reserved for the sidecar.
   On N16R4, a one-GPU batch request cannot receive the required memory
   headroom. The site-specific launcher may therefore reserve two GPUs and
   eight CPUs at the outer job level without an explicit memory request, but
   it must immediately enter one `srun --exact` step with exactly one GPU,
   five CPUs, and 96,000 MiB. The model, test, profiler, and power sidecar run
   only inside that step. They may not access the idle reserved GPU, override
   `CUDA_VISIBLE_DEVICES`, or describe the run as two-GPU computation. The
   process must verify a finite cgroup v2 step limit of at least 90,000 MiB
   before opening evidence.
   If the frozen training commit predates this two-level Slurm scope, a formal
   test may run from a newer recovery-certificate-bound runtime commit only
   when the certificate proves that all changes are confined to audited S1
   test/profile/evidence infrastructure and that `opentad/`, resolved configs,
   model code, checkpoint semantics, and evaluator code are unchanged. The
   test-open marker, test evidence, matrix binding, profile, and descriptor
   must bind both training and runtime commits plus the recovery certificate.
   Rewriting Slurm identity variables or modifying the historical checkout in
   place is forbidden. Formal recovery launchers must disable Python user-site
   packages so package resolution stays inside the frozen Conda environment;
   ambient login-node packages cannot participate in evidence recomputation.
   Sampling uses `time.monotonic_ns`, preserves a sequence-numbered raw trace
   and a self-hashed attempt report even when profile validation fails, and
   uses one canonical output prefix. Cost claims are limited to same-node,
   same-GPU, warm serial per-window latency and gross GPU energy; they do not
   represent cold-start, whole-video latency, incremental energy, or CPU and
   storage energy.
9. Before consuming the single-use matrix namespace, complete all artifact,
   test-evidence, source-checkout, representative-cell, Slurm-step, cgroup,
   logical-CUDA UUID, software, and Gate-class checks without a canonical
   write. Only then acquire the persistent matrix lock and atomically publish
   one start receipt. The receipt binds one numeric Slurm job/inner-step pair,
   a step GPU that belongs to the outer job allocation, logical `cuda:0` to
   its NVML UUID, the exact 4+1 CPU partition, effective step memory, software,
   Gate, recovery certificate, and frozen order. A failed lock is never
   removed or reused.
10. Bind checkpoint, prediction, marker, certificate, manifest, config,
    profile, Git commit, precheck, matrix-start receipt, Slurm job/step, GPU
    UUID, and internal/file hashes into one descriptor per resolution and
    seed. After all nine cells complete, atomically publish one completion
    receipt that revalidates the exact frozen-order descriptor set. The final
    analyzer must reject a matrix without this completion receipt.
11. Recompute full class AP under a paired Bayesian video-cluster bootstrap
    with fixed class support and hierarchical training-seed resampling. Require
    parity with the official THUMOS evaluator, apply a one-sided simultaneous
    max-T lower bound across 224 and 256, report boundary error, and use
    measured full-stack cost in resolution freezing.

## Current Verification

- The previous `35204f5` matrix is protocol-invalid diagnostic evidence. All
  nine cells emitted CUDA `upsample_linear1d_backward_out_cuda`
  nondeterminism warnings under warning-only enforcement and were cancelled
  before completion. They cannot be resumed into the formal table.
- The replacement configs use an analytical exact-2x temporal interpolation
  that is forward/backward equivalent to `linear, align_corners=False` while
  avoiding the nondeterministic CUDA backward kernel. Formal train, test,
  profile, and full precheck entrypoints request strict deterministic
  algorithms.
- The result analysis uses 10,000 positive paired Bayesian video-cluster
  weights without class-support rejection. Test-open recovery and immutable
  evidence files use atomic publication; the global marker embeds the exact
  recoverable certificate.
- Local syntax checks pass. The combined focused suite reports
  `66 passed, 1 skipped`, including the required C3 regression.
  The skipped
  interpolation parity test requires the Linux Torch runtime and is mandatory
  in the Slurm precheck.
- The resolved matrix validator passes for 160/224/256 with protocol
  fingerprint
  `3dc356baec2d69b8f13fc2096f0df00b5e9e387935bb80bd2a73d3a25037eb0c`.
- The `64e71dd` exact snapshot passed `41` Linux tests. Slurm Job `1165648`
  reached a real pretrained full-model AMP backward and failed only because
  VideoMAE's classification-pooling `fc_norm.{weight,bias}` remain trainable but
  are bypassed by the configured `return_feat_map=True` TAD feature path.
- Precheck v6 therefore permits exactly those two fully qualified parameter
  names to be absent from the backward graph, reports trainable and
  gradient-required counts separately, closes all component counts against the
  global totals, and rejects any missing or additional disconnected parameter.
  Replacement commit `47842427eb373fb1f440b1661971a6a231a95f67` passed
  CUDA gate Job `1165667`: all three resolutions have 339 trainable tensors,
  the exact two audited-unused tensors, 337 finite gradients, and nonzero
  backbone/projection/head gradients. Formal jobs `1165669-1165677` were later
  invalidated by shared-storage exhaustion and cannot be resumed or reported.
- Storage-safe commit `0421a8d9f6982a6d4ec1fb590cd108581fa2bb83`
  persists only gate-eligible checkpoints, rolls back failed checkpoint
  transactions, and enforces the 96 GiB preflight. CUDA gate Job `1165774`
  passed with `47` Linux tests and the same full-model gradient contract. Fresh
  jobs `1165775-1165783` run the epoch-0 3x3 replacement in canonical namespace
  `bf71376e2d57946a3f898d25b7dcc88cfc002549a9ed78656293f1a95316a8f7`.
  S1 is only `experiment_running`; no S1 GO, crop-model success, empirical
  support, or paper claim is allowed.
- Power diagnostic Job `1167536` used no test data and compared the inherited
  persistent `nvidia-smi` pipe against native NVML under matched CUDA load.
  The pipe's maximum arrival gap was `678.458` ms and failed the frozen 100 ms
  limit; NVML's maximum was `57.709` ms and passed. Formal profiles therefore
  obtain the NVML handle by the frozen Slurm-allocated GPU UUID, retain the
  20 ms target and 100 ms limit, and require a versioned chained recovery
  certificate preserving both failed attempts plus the diagnostic before
  another matrix can start.
- Serial profile Job `1167538` falsified the sufficiency of the short
  in-process NVML Gate: the first full 792-exposure path reached
  `2413.519` ms maximum sampling gap at about 60.7 GiB RSS. The replacement
  recovery is therefore v3 and binds Job `1167538`, rejects the old
  `nvml-persistent-poll-v1` backend, and requires
  `nvml-sidecar-process-v1`. Before any replacement matrix, a representative
  dense256/seed3408 no-new-test-open Gate must execute all 792 exposures and
  official finalization with at least 90,000 MiB allocated memory. It may
  publish only its sidecar trace/report and a self-hashed Gate record, never a
  formal profile, prediction, latency table, or descriptor. Every formal cell
  recursively validates that Gate record. The Gate attempt is bound to the
  actual GPU UUID assigned to its own Slurm job. A later matrix allocation need
  not receive that same physical UUID, but its stable GPU/CPU/resource class
  and software fingerprint must match the Gate. All nine matrix cells must
  still run serially in one allocation on one physical GPU, and each cell's
  sidecar is bound to that allocation's actual UUID.
- Attempt report and raw trace are validated as one indivisible pair by the
  profiler, Gate, descriptor builder, and final analyzer: self-hash, exact
  trace hash/path, recomputed cadence, monotonic clock, UUID, affinity, and
  child records must all agree. A launcher-level salvage path seals node-local
  samples as an immutable FAIL attempt if the detector worker exits before
  normal finalization; if only one artifact exists it may complete only the
  missing hash-matching artifact, never overwrite the survivor. If the attempt
  was already sealed, salvage publishes a separate immutable parent-failure
  record instead of rewriting the attempt.
- The serial matrix acquires an atomic, persistent campaign lock before any
  cell starts and publishes self-hashed start/completion receipts. A failed
  matrix never releases that lock; retry requires a new certificate-bound
  campaign. The current local S1 suite reports `61 passed, 4 skipped`; three
  Linux real-subprocess lifecycle/failure cases must execute remotely, while
  the CUDA-only parity case remains mandatory in the formal GPU precheck.
- Resource-only Slurm diagnostics `1168504`, `1168506`, `1168509`, and
  `1168510` established the N16R4 execution scope without reading model, data,
  checkpoint, annotation, or sealed-test evidence. An outer two-GPU/eight-CPU
  allocation receives `124400M`; an inner exact step receives one GPU, five
  CPUs, and a finite cgroup limit of exactly `96000` MiB. Inside that step,
  `CUDA_VISIBLE_DEVICES` exposes one logical device and `SLURM_STEP_GPUS`
  identifies one physical GPU. This authorizes only the audited launcher
  construction; it is not model, accuracy, or cost evidence. The second outer
  GPU remains an idle site-policy reservation and must be disclosed separately
  from the measured single-GPU profile.
- Resource-step commit `84a7144` was independently audited as `HOLD` with no
  P0 and four P1 gaps: preflight failures could enter salvage, the matrix lock
  preceded complete no-write validation, GPU identity did not prove
  job/step/logical-CUDA closure, and the nine descriptors were not sealed to
  one matrix step. It is diagnostic-only and cannot issue a formal recovery
  campaign.
- The current local repair moves Gate resource checks before evidence access,
  guards salvage on actual child evidence, requires cgroup v2, proves step-GPU
  membership and logical-CUDA/NVML UUID equality, performs full matrix-start
  construction before the canonical lock. All nine frozen cells must pass a
  no-write dry-run before that lock is acquired. New official-test evidence is
  cryptographically bound to the canonical matrix-start receipt before
  profiling; the sole pre-existing dense256/seed3408 evidence is allowed only
  through the v3 certificate's exact path, file hash, internal hash, and cell
  identity. Every profile, marker, canonical descriptor, completion receipt,
  and final report is bound to the same matrix job/step/GPU. The exact local
  S1, matrix, train-engine, and required C3 regression reports
  `99 passed, 5 skipped`. A second independent max-level review found no P0 or
  P1 and returned `DEPLOY`, conditional on a clean commit, exact remote replay,
  and a newly issued certificate. The skips remain platform/CUDA cases that
  all passed in clean remote snapshot `5bfdc36` (`104 passed`). Recovery
  campaign `e3fccb9b12a5d24d` then passed no-open Gate Job `1168608`: all
  `792` loader exposures and `791` physical windows completed, the existing
  test evidence hash stayed unchanged, and the out-of-process sidecar recorded
  `110699` samples with median/P95/max gaps
  `20.000/20.022/63.098` ms. This Gate authorized the sole frozen-order matrix
  Job `1168823`; that campaign is now the immutable failure described below.
- Matrix Job `1168823` later failed closed in its first formal
  dense256/seed3408 profile. Its out-of-process sampler exited normally and
  preserved `112107` raw samples, but three observed intervals exceeded the
  frozen `100` ms limit; median/P95/max were
  `20.000005/20.023177/146.048168` ms. No profile summary, samples, paper power
  trace, descriptor, completion receipt, or later cell was published. The
  v3 campaign remains immutable and cannot be resumed. The current v4 repair
  keeps the `20/100` ms thresholds and 4+1 CPU partition unchanged, removes
  filesystem I/O from the sampling loop, buffers sequence-numbered JSONL in
  sidecar memory, and atomically publishes it only after sampling stops. The
  v4 certificate recursively binds the failed marker, raw attempt pair,
  parent-failure evidence, matrix-start receipt, Job/step/GPU identity, and
  the old v3 certificate. It also proves
  `0 < start <= trace_first == ready_first <= trace_last <= finish`; formal
  profiles and new Gates accept only this v4 trace mode, while v3 remains
  readable only to validate immutable parent evidence. Local
  S1/matrix/train-engine/C3 verification is `102 passed, 5 skipped`. A
  three-pass independent review returned HOLD for three initial P1s, HOLD for
  one remaining trace-lifecycle P1, and finally DEPLOY with no P0/P1. State
  remains `tested_local`: a clean commit, exact remote Linux/CUDA replay, a new
  immutable v4 campaign, and another full no-open Gate are mandatory before
  one replacement matrix.
- Commit `bc9350e628c7ec3f0abafa8814429eb0d8476c4a` passed `107` exact
  remote Linux tests and issued v4 campaign `6021eaba62337726`. Its no-open
  Gate Job `1170341` failed `6:0` after six seconds, after the 96000M resource
  preflight and existing-test-evidence hash but before any profile/test run,
  sidecar marker, or new evidence. Resource-only Job `1170342` proved the
  mismatch: `SLURM_STEP_GPUS=1` is the node physical slot, while the cgroup
  exposes only `CUDA_VISIBLE_DEVICES=0`; `nvidia-smi -i 1` returned 6 and
  `nvidia-smi -i 0` returned the allocated UUID. Campaign `6021...` is
  immutable failed infrastructure and cannot be retried. The local repair
  retains physical slot `1` as Slurm identity, uses the sole cgroup-visible
  selector `0` only for NVML queries, and still requires the NVML UUID to equal
  logical `cuda:0`'s runtime UUID. Verification is `102 passed, 5 skipped`.
  Independent review returned `DEPLOY` with no P0/P1 and made no file changes;
  a clean commit, remote replay, a new v4 campaign, and a fresh no-open Gate
  are still required.
- Environment-pinned runtime commit
  `3d01d3b7fc956ae17568ac3c8c04f9d6f36c42c5` and its v5 campaign
  `3180634880aa8de0` closed the step-scoped test runtime and Python environment
  provenance. Its sole no-open Gate Job `1170765` nevertheless failed closed
  before sidecar startup because the formal profiler required the literal v4
  recovery reason instead of validating the buffered-sidecar capability that
  v5 inherits. The Job published only its immutable submission receipt and
  stdout/stderr; it opened no new test, changed no existing test evidence,
  emitted no sidecar Gate/profile/descriptor, and authorized no matrix.
- Recovery descendants must be accepted by capability, not by schema name.
  A formal buffered-sidecar recovery must preserve the exact backend,
  post-sampling atomic publication mode, no sampling-loop I/O, 20/100 ms
  cadence, 4+1 CPU split, long no-open Gate requirement, and Gate path. The v6
  schema-compatibility certificate recursively binds the v5 certificate,
  Job `1170765` submission receipt and exact failure logs, the exact four-file
  parent campaign inventory, absence of sidecar evidence, and the unchanged
  official-test evidence hash. The parent certificate and both logs must use
  their canonical campaign paths; stdout/stderr may not share a path or inode,
  and their frozen file hashes plus structured failure payloads must match.
  Legacy, matrix, Gate, and power-diagnostic evidence modes are mutually
  exclusive for each recovery transition. It permits changes only to the
  three affected runtime consumers plus their contract/spec/test records;
  model, config, checkpoint, evaluator, test protocol, resources, frozen
  order, and statistics remain unchanged.

This remains infrastructure evidence, not an S1 result. A clean commit, exact
remote replay, independently reviewed v6 certificate, and one successful
no-new-test-open full-path Gate are still mandatory before any replacement
matrix.
