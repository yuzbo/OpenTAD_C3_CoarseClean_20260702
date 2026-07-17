# Spatial Zoom S1 Power Sidecar Design

Date: 2026-07-17

## Decision

Replace the formal in-process NVML polling thread with a minimal, UUID-bound
NVML sidecar process. Preserve the frozen 20 ms target interval and 100 ms
maximum-gap limit. Do not interpolate, discard, or excuse missing samples.

This is an infrastructure repair only. It does not change the trained model,
checkpoint selection, sealed-test predictions, official evaluator, detector
path, spatial-resolution matrix, or S1 GO/KILL thresholds.

## Rejected Alternatives

1. Keep the Python thread and increase the gap threshold.
   Rejected because Job 1167538 demonstrated a 2413.519 ms long-tail stall
   under the full high-RSS detector path. Relaxing the threshold would move the
   protocol after seeing a failure.
2. Return to `nvidia-smi --loop-ms`.
   Rejected because Jobs 1167516 and 1167536 demonstrated burst-buffered pipe
   delivery with gaps above 600 ms.
3. Use a separate native-NVML process.
   Selected because it removes detector GIL coupling while retaining direct
   NVML sampling, UUID identity, raw monotonic timestamps, and strict cadence
   validation.

## Process Contract

The detector process launches one child Python interpreter using the same
certificate-bound source checkout. The child imports no Torch/OpenTAD model
code, resolves the allocated GPU by UUID, verifies the actual UUID, and polls
`nvmlDeviceGetPowerUsage` at 20 ms.

The sidecar writes sequence-numbered records containing `monotonic_ns` and
`power_w` to an exclusive node-local trace. It publishes a ready record before
measurement and a terminal record after graceful stop. Parent and child use
Linux monotonic time, which is system-wide and comparable across processes.

The launcher requires at least five Slurm CPUs. Four CPUs remain available to
the detector, matching the previous detector budget; one CPU is reserved for
the sidecar. The allocated, detector, and sidecar CPU sets are recorded and
hashed. If affinity expansion is unavailable, the run fails before sampling.

N16R4 limits a one-GPU outer request to 55 GB, below the observed full-path
requirement. The launcher therefore supports a site-specific two-level
allocation: an outer two-GPU/eight-CPU reservation obtains the site's default
124400M, then immediately re-executes in one exact Slurm step requesting one
GPU, five CPUs, and 96000M. All detector and sidecar work remains inside that
single-GPU step. The process records `SLURM_STEP_GPUS`, preserves Slurm's
`CUDA_VISIBLE_DEVICES`, and verifies the tightest finite cgroup/Slurm memory
limit before evidence access. The unused outer GPU is scheduling overhead,
not model compute, and must be disclosed as such.

## Evidence Contract

Every sampling attempt atomically publishes two immutable campaign artifacts:

- a raw sidecar trace copied from node-local storage;
- a self-hashed attempt report containing UUID, PID, CPU affinity, clock
  identity, sample count, cadence statistics, process exit status, and trace
  hash.

These artifacts are published before profile-summary validation. Therefore a
cadence failure still leaves auditable raw evidence. Formal profile
summary/sample/power/descriptor publication remains transactional and occurs
only after all validation passes.

If the detector worker exits before its Python `finally` block completes, the
Slurm launcher invokes a certificate-bound salvage entrypoint. It terminates
the sidecar if still alive, copies the node-local trace once, and publishes an
immutable self-hashed FAIL report. If the sidecar attempt was already sealed
successfully before a later detector/profile failure, salvage leaves that
attempt immutable and publishes a separate parent-failure record bound to both
attempt artifacts. Salvage is idempotent and never converts a failed parent
flow into a valid profile.

Successful formal summaries and descriptors bind the attempt report and raw
attempt trace hashes. Old in-process profiles are rejected by the new backend
and metadata schema.

## Gate Contract

A new recursive recovery certificate binds the immutable Job 1167538 marker,
logs, parent recovery certificate, and exact repair diff. It requires the
sidecar backend and a separate no-open sidecar Gate.

The Gate:

- reuses the already valid dense256/seed3408 sealed-test evidence;
- performs the complete 792-exposure model/profile/finalizer path;
- publishes no paper profile, latency table, new prediction, or descriptor;
- publishes only sidecar attempt evidence and a self-hashed Gate result;
- binds the Gate sidecar to the actual UUID assigned to the Gate job;
- requires the matrix to match the Gate's stable GPU/CPU/resource class,
  software fingerprint, recovery certificate, and full exposure topology.
  A separate matrix allocation may receive a different physical UUID, but
  every matrix attempt must bind to its own actual UUID.

Any sample gap above 100 ms, child crash, timeout, UUID mismatch, affinity
drift, missing raw trace, or hash mismatch fails closed. A matrix launcher must
validate the successful Gate result before its first cell.

## Buffered Trace Amendment

The first formal v3 matrix, Job `1168823`, falsified the assumption that CPU
isolation alone was sufficient. Its first cell preserved `112107` samples and
failed on three gaps above `100` ms (maximum `146.048` ms), while the child
process itself exited normally. No formal profile or descriptor was
published. The failed campaign and start receipt are immutable.

The v4 repair does not move the cadence threshold. It removes line-buffered
filesystem writes from the measured loop: the sidecar records each
sequence-numbered sample in an in-memory byte buffer with cyclic garbage
collection disabled, then atomically publishes the complete JSONL trace after
SIGTERM ends sampling. The ready and PID records remain available while the
child is live. A normal stop must publish trace and result before the parent
finalizes the attempt; a crash or timeout remains a FAIL and enters the
existing salvage path. The v4 recovery certificate binds the v3 parent,
Job `1168823`, its v7 marker, failure log, FAIL attempt report/raw trace,
parent-failure record, matrix-start receipt, Slurm step and GPU UUID. Formal
markers, Gate evidence, and profile metadata explicitly record
`post_sampling_atomic_jsonl_v1` and `trace_io_inside_sampling_loop=false`.
The validator binds the process lifecycle to the exact raw trace with
`0 < start <= trace_first == ready_first <= trace_last <= finish`; rehashed
records with a shifted ready timestamp or a finish before the final sample are
rejected. Formal profiles and new no-open Gates accept only v4. Historical v3
is read only for recursive validation of the failed parent campaign.

The Slurm physical slot and the cgroup-visible NVML selector are separate
identities. `SLURM_STEP_GPUS` remains the physical resource identity recorded
in receipts. A one-GPU step may renumber that device to
`CUDA_VISIBLE_DEVICES=0`; therefore `nvidia-smi` queries use that sole visible
selector, record it explicitly, and accept the result only when its UUID equals
the CUDA driver UUID for logical `cuda:0`. Neither field substitutes for the
other.

## Formal Matrix

After the Gate passes, exactly one Slurm allocation runs the frozen nine cells
serially on one node and physical GPU. All cells use the same five-CPU
allocation with four detector CPUs plus one sidecar CPU and sufficient memory
headroom. The existing dense256/seed3408 test evidence is reused, not reopened.
On N16R4 the matrix's outer reservation may contain a second idle GPU solely
to satisfy the site's memory policy, but the persistent matrix lock, Gate
identity, all nine cells, and all measured costs bind to the same one-GPU
96000M inner step.

Before the first cell, the matrix atomically creates a persistent campaign
lock and self-hashed start receipt binding the Slurm job, recovery certificate,
Gate, code commit, resources, and frozen order. Concurrent or repeated jobs
fail at lock acquisition. A successful matrix adds a self-hashed completion
receipt binding all nine descriptors. The lock is never removed after failure,
so the matrix cannot resume a failed campaign; a new audited recovery campaign
is required.

## Tests

Focused tests cover:

- subprocess start/ready/stop and no orphan process;
- UUID and CPU-affinity mismatch;
- child crash, timeout, malformed or non-monotonic records;
- sparse trace rejection without interpolation;
- raw failure-trace and attempt-report preservation;
- trace-only and report-only partial-state recovery without overwriting the
  surviving artifact;
- descriptor/analyzer rejection when an attempt report is paired with a
  different raw trace;
- immutable no-overwrite behavior and self-hashes;
- cross-process monotonic timestamp integration;
- a real Linux `Popen -> ready -> sample -> SIGTERM -> PASS` child lifecycle;
- Gate-only prohibition on formal profile/descriptor publication;
- matrix rejection without matching Gate hardware/software class evidence;
- atomic rejection of concurrent or repeated matrix launchers;
- recursive recovery binding of Job 1167538;
- recursive v4 binding of Job 1168823 and its failed matrix-start receipt;
- no trace-file publication during the sampling loop and atomic publication
  after normal stop;
- rejection of rehashed ready/trace-first and finish/trace-last mismatches;
- physical Slurm slot versus cgroup-visible NVML selector mapping, including a
  nonzero physical slot renumbered to logical selector zero;
- rejection of old in-process backend profiles;
- existing S1 and C3 regression contracts.

## Success Criterion

Implementation is deployable only after local focused tests, exact remote Linux
tests, and one representative full-path no-open Gate all pass. Only then may
the single replacement 3x3 matrix be submitted.
