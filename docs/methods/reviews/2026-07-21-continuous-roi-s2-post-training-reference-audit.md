# Continuous-RoI S2 Post-Training Reference Audit

Date: 2026-07-21

## Scope

This audit covers only the development-only Continuous-RoI S2 v2.1 route.
It does not open official test, implement a learned ROI policy, or authorize a
paper claim. The training runtime remains commit
`9a61da27e65c2227c8d2a0c547d8f3cb44966738`.

## Exact-Nine Training Evidence

Jobs `1177668-1177676` all completed `0:0`. Every cell was reloaded from its
live bound config, final checkpoint, metadata sidecar and completion receipt.
All cells contain 60 epochs, 80 successful updates per epoch, exactly 4,800
successful updates and final EMA only. The bound deployment/completion
evidence prohibits official-test use, and no official-test Job, result, or
evidence artifact exists. Historical training did not instrument syscall-level
file access, so no runtime zero-open claim is made.

| Family | Seed | Final loss | Attempts | AMP skips | Max retry |
|---|---:|---:|---:|---:|---:|
| D160 | 3407 | 0.2190 | 4804 | 4 | 2 |
| D160 | 3408 | 0.2172 | 4804 | 4 | 1 |
| D160 | 3409 | 0.2115 | 4803 | 3 | 1 |
| G96 | 3407 | 0.2259 | 4803 | 3 | 2 |
| G96 | 3408 | 0.2184 | 4804 | 4 | 2 |
| G96 | 3409 | 0.2219 | 4803 | 3 | 1 |
| U128 | 3407 | 0.2517 | 4803 | 3 | 1 |
| U128 | 3408 | 0.2483 | 4803 | 3 | 1 |
| U128 | 3409 | 0.2404 | 4803 | 3 | 1 |

This evidence establishes training integrity only. It does not establish
development detection quality, reference adequacy, crop sufficiency or cost.

The saved scheduler state closes at successful update `4800`, while the
inherited cosine horizon is `8000` updates (`100` epochs) and warmup is `400`
updates. This is matched across all arms and is not a receipt-integrity
failure, but the final checkpoint is a registered 60-epoch truncation rather
than the endpoint of the inherited cosine schedule. Any later interpretation
of convergence must keep that distinction explicit.

## Evidence-Finalizer Review

An initial independent read-only review identified four P1 evidence risks:

1. Slurm completion was not bound strongly enough to the submitted job.
2. Validator provenance could be supplied or differ from tracked Git bytes.
3. Checkpoint validation did not inspect optimizer, scheduler and raw/EMA
   tensor state deeply enough.
4. Config loading could execute Python before trust validation, while the
   aggregate no-test wording exceeded the available runtime instrumentation.

The repaired finalizer now binds `JobIDRaw`, `JobName` and the frozen Slurm
comment token, hashes stdout/stderr, requires a clean tracked validator commit,
rejects non-data config syntax before MMEngine loading, compares the live
semantic config hash with the submitted cell intent, reloads every optimizer,
scheduler, raw and EMA state, and checks all optimizer steps at `4800`.
The aggregate receipt explicitly sets
`official_test_runtime_access_audited=false` and `official_test_opened=null`.
It therefore certifies only development-input binding and the absence of an
official-test Job/result/evidence artifact, not syscall-level zero access.

A second independent review held this implementation on two additional P1
risks: PyTorch could preserve optimizer state not referenced by any serialized
parameter group, and checkpoint loading still used unrestricted pickle. The
final repair requires unique serialized parameter IDs, rejects orphan state
before and after loading, checks real runtime parameter-group sizes, and loads
checkpoint data with `weights_only=true`. It also binds deployment-intent and
job-receipt status plus base/campaign namespaces. A third read-only review
returned `NO_P0_P1` after these changes.

Residual P2 risk remains explicit. The local negative tests use a small Torch
model and do not replace strict loading of the three real D160/G96/U128
detectors, optimizers, historical receipts and checkpoints. That integration
check is the purpose of the one formal Linux training-matrix finalizer run. AMP
retry evidence closes the aggregate attempt/update ledger, but historical logs
do not independently reconstruct every same-batch replay.

The first formal Linux replay, Slurm Job `1178693`, failed closed before
publishing a matrix receipt. Its validator compared the historical Gate source
hashes with the newer finalizer checkout instead of the Gate-bound training
checkout. The correction passes the already-validated clean training repository
root into Gate source verification; it does not weaken the commit, source-hash
or self-hash requirements. Job `1178693` is immutable diagnostic evidence and
cannot be reused as a successful receipt.

An independent read-only review of this repair returned `NO_P0_P1`. It noted a
P2 API-hardening opportunity: direct callers of the low-level Gate validator
could supply a non-Git source directory with matching bytes. The formal path is
not exposed to that ambiguity because it first requires the derived historical
repository root to be clean and exactly at the Gate-bound commit.

Replacement finalizer Job `1178735` then failed closed on one legitimate
raw/EMA buffer dtype difference. Diagnostic Job `1178737` isolated the D160
seed-3407 checkpoint, and all-nine diagnostic Job `1178739` confirmed the same
single key in every D160/G96/U128 checkpoint:
`module.rpn_head.loss_normalizer`. Training reassigns the registered integer
buffer to a floating tensor in the raw model, while the EMA copy keeps the
constructor's integer dtype. The official EMA consumer therefore remains
dtype-exact; this is not parameter corruption. Neither diagnostic published a
matrix receipt or accessed a reference or official-test phase.

The validator now requires exact keys and shapes, exact EMA-to-runtime dtype,
and exact raw parameter dtype, including every shared-parameter alias. It
allows only the frozen `module.rpn_head.loss_normalizer` raw-buffer cast,
requires the generic raw/EMA audit and real-model classification to identify
the same key set, and records that set in every cell. This is a one-key model
contract, not permission for arbitrary registered-buffer coercion. The
validation model also reproduces the training DDP `module.` state prefix
instead of loading into an unwrapped detector.

The repaired code was committed as `4543205`; a clean Linux snapshot at the
exact commit passed `83` focused tests with NumPy `1.23.5`. Finalizer Job
`1178742` did not enter Python: Slurm's generated `/bin/sh` wrapper rejected
the Bash-only `set -o pipefail` and exited `2:0` in zero seconds. It produced no
matrix receipt and is retained as immutable submission-infrastructure evidence.
The next submission must invoke an explicit Bash process; it may not change
the runtime commit, checkpoints, deployment hash or scientific protocol.

## Reference-Phase Code Audit

The repository contains reusable continuous geometry, runtime sampling,
external `[B,48,4]` boxes, official proposal/NMS behavior and training receipt
validation. It does not contain an end-to-end S2 reference implementation:
there is no frozen paired candidate generator/seal, annotation-free raw
129-window inference path, typed raw shards, privileged CPU join, S2 analyzer,
or S2 ABBA profiler.

Implementing these pieces directly from v2.1 would be scientifically unsafe:

1. `docs/methods/continuous_roi_s2_v2_1_contract.md:93` says FS and VS share
   physical center trajectories and differ only in size/aspect. Lines 110-111
   define physical center as a function of width/height, and lines 251-253
   share only `sx,sy` while VS changes `sa,sr`. The implemented decoder has the
   same dependence at
   `opentad/models/backbones/continuous_roi_geometry.py:173`. Therefore the
   decisive contrast currently changes both physical centers and sizes.
2. Lines 244-254 freeze a Sobol seed and draw shape, but not the exact engine,
   scrambling implementation, dtype, byte serialization, transform order,
   stable-hash encoding or known-answer population hash. The required exact
   generator identity at line 310 is not reproducible from the contract alone.
3. Lines 269-273 forbid a reference ID in the raw object graph but require
   candidate IDs in the sealed raw output. The contract must distinguish an
   enumerated result-blind candidate ID from a preferred ID chosen after GT.
4. Lines 275-301 do not fully freeze tie ordering, independent 0.7/0.5
   matching state, false-positive matching state and frame interval details.
   The D0 box set/order, Short-Q1 cutoff, Recall@100 matching, boundary error,
   bootstrap/max-T edge cases and search-adequacy tube-IoU pooling are also not
   fully machine-defined.
5. The current validation data path retains annotation-bearing metadata. A
   dedicated annotation-free raw manifest and entrypoint are required before
   the GPU sweep can truthfully satisfy the no-GT object-graph contract.

## Verdict

`TRAINING_MATRIX_COMPLETE / REFERENCE_PROTOCOL_HOLD`.

The only authorized implementation now is a training-only exact-nine receipt
that binds Slurm accounting and revalidates all live artifacts. It must state
`reference_sweep_completed=false`, `crop_sufficiency_established=false`,
`official_test_open_allowed=false`,
`official_test_runtime_access_audited=false`, `official_test_opened=null`, and
`paper_claim_allowed=false`.

The next scientific action is a minimal v2.2 corrigendum that resolves the
five blockers without looking at S2 development predictions. Only after a new
static audit and development Gate may the reference pipeline be implemented
and queued. Official test and S3 remain sealed.
