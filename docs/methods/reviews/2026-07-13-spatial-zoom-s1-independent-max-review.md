# Spatial Zoom S1 Independent Max Review

## Review identity

- Date: 2026-07-13
- Reviewer agent: `019f5b3c-9b7f-73f2-8eea-00157a60a119`
- Model: `gpt-5.6-sol`
- Reasoning effort: `max`
- Review mode: independent, read-only, no file edits
- Scope: S1 infrastructure before any remote CUDA training

## Design boundary

The authorized object is an offline-TAD falsification gate. Dense-160,
Dense-224, and Dense-256 must retain the same 768-frame temporal grid,
VideoMAE-S/AdaTAD/ActionFormer detector, optimizer, evaluator, and checkpoint
rule. S1 must not contain an ROI cropper, scout, policy, teacher, oracle, new
detector, S2 implementation, or DART-Zoom implementation.

## First review

Initial verdict: `FAIL_BEFORE_REMOTE_TRAINING`.

The reviewer accepted the high-level S1 boundary but found that the first
implementation was not yet auditable:

1. Source configs could still be trained without a manifest-bound runtime
   split, so fit/gate separation was not enforced by the training entrypoint.
2. Checkpoint selection did not prove that every eligible epoch was present,
   and downstream files could trust hand-entered candidate metrics.
3. Test artifacts and run descriptors did not revalidate the complete hash
   chain, leaving room for test-result or artifact substitution.
4. Short-action AP filtered predictions by predicted duration and could
   inflate AP; class support, non-finite handling, and simultaneous inference
   were insufficiently strict.
5. The profiler omitted final cross-window NMS and an independent continuous
   total, used too few samples, and did not bind complete sample identities.
6. Full precheck could run a partial matrix or random initialization and did
   not prove positional interpolation for all three resolutions.
7. Manifest stratification and tests did not cover the corresponding failure
   modes.

## Remediation implemented

- Manifest-bound materialized configs are now required for formal training.
  The binding freezes fit/gate/test identities, seeds, work directory, Git
  commit, annotation/config hashes, and a full-CUDA precheck certificate.
- Formal entrypoints require a clean checkout, physical GPU1, Slurm,
  deterministic single-process execution, no resume, no config override, a
  fresh work directory, and exact runtime dataset identities.
- Every successful checkpoint has immutable metadata and a SHA sidecar. Any
  skipped AMP update fails the run. Gate evidence is written in-process and
  proves complete dataloader coverage.
- Selection requires the complete eligible-epoch set and recomputes every
  candidate metric from raw gate predictions. A complete 3x3 selection matrix
  is required before a test-open certificate can be issued.
- Sealed test opening has one canonical output path and an exclusive marker.
  The marker, prediction, checkpoint, certificate, profile, descriptor, and
  analyzer are linked by file hashes and internal self-hashes.
- Duration AP is GT-conditioned while every prediction remains a possible
  false positive. Invalid/non-finite rows fail closed. The local AP
  implementation is checked against the repository's official THUMOS
  evaluator, including score ties.
- The result gate uses paired video-cluster resampling, training-seed
  resampling, a one-sided simultaneous max-T lower bound, boundary-error
  diagnostics, and a cost-aware resolution-freeze rule.
- The full-stack profiler includes decode/preprocess, H2D, detector forward,
  per-window postprocess, cross-window final NMS, continuous wall time, peak
  memory, and GPU energy. It uses the complete sealed-test loader, canonical
  window identities, 50 warmups, and a fixed 20 ms power interval.
- Full precheck requires all three resolutions, the real pretrained VideoMAE
  checkpoint and hash, the correct positional-interpolation branches, the
  768-point detector grid, CUDA AMP memory evidence, a clean commit, and a
  self-hashed certificate.

## Local evidence before second review

- S1 tests: `13 passed`.
- Required C3 plus S1 focused regression: `33 passed`.
- Python compilation: passed.
- Shell syntax checks: passed.
- Config matrix validator: passed.
- Protocol fingerprint:
  `343df802fddc5658b97fda4a917c8e8576a0a2801f7ddd77a58a3a462feec2c0`.
- Static precheck v2: passed.
- Formal full-CUDA precheck: not run.
- S1 training/test/profile matrix: not run.

## Second review

Second-round verdict: `FAIL_BEFORE_REMOTE_TRAINING`.

No design-scope leakage or split leakage was found, and all first-round
blockers improved. Five P1 issues still affected the formal S1 decision:

1. The simultaneous lower max-T bound used the opposite bootstrap pivot.
2. A partial cost vector could veto an accuracy-valid candidate.
3. Full precheck asserted that a checkpoint was loaded without proving core
   key/value coverage, and interpolation evidence accepted extra/wrong calls.
4. Profiling stopped the continuous timer before result accumulation, did not
   reuse the official DDP gather/output path, and did not preserve an auditable
   power trace or actual sampling-gap bound.
5. Per-seed profile comparisons did not prove global comparability across all
   nine runs.

## Second-round remediation

- The max-T helper now calibrates the upper tail of
  `(theta_bootstrap - theta_observed) / SE`; a skewed correlated-candidate test
  distinguishes it from the reversed pivot.
- S1 GO is determined only by the six frozen accuracy conditions. Cost cannot
  veto GO and only freezes the lowest-p50 resolution among candidates that
  independently pass all accuracy conditions.
- Precheck v3 requires a preregistered checkpoint SHA and exact key, shape, and
  loaded-value equality for every non-adapter VideoMAE core parameter. The
  interpolation call sequence must be exactly `[[10,10]]`, `[]`, and
  `[[16,16]]` for 160/224/256.
- Profiling now runs under `torchrun` with single-process DDP, includes result
  accumulation in continuous wall time, and directly reuses
  `opentad.cores.test_engine.gather_ddp_results` for gather, NMS, and output
  reconstruction. A persistent `nvidia-smi --loop-ms=20` process records an
  embedded and standalone raw trace; a gap over 100 ms fails closed.
- The analyzer now requires one globally consistent hardware/software/sample/
  protocol/precheck/test-open identity across the complete 3x3 matrix.

## Local evidence before third review

- S1 tests: `18 passed`.
- Required C3 plus S1 focused regression: `38 passed`.
- Python compilation, shell syntax, config validator, and `git diff --check`:
  passed.
- Static precheck v3: passed.
- Formal full-CUDA precheck: not run.
- S1 training/test/profile matrix: not run.

## Third review

Third-round verdict: `FAIL_BEFORE_REMOTE_TRAINING`.

The reviewer accepted the second-round statistical, GO/KILL, real-pretrain,
DDP-finalizer, and cost-comparability fixes, but found five remaining P1
provenance failures:

1. A different fit/gate partition could be accepted after recomputing the
   manifest's self-hash and summaries.
2. The expected VideoMAE checkpoint SHA was supplied at runtime rather than
   frozen by repository code.
3. A crashed profiler could be rerun before formal outputs existed.
4. Hardware/software fingerprints were too weak to distinguish nodes, GPU
   UUIDs, drivers, CPU state, and the decode stack.
5. The test-open certificate did not require every one of the nine selections
   to share one precheck and pretrained-checkpoint identity.

## Third-round remediation

- Manifest v2 is rebuilt deterministically from the annotation and compared as
  a complete object; a rehashed alternative fit/gate partition now fails.
- The exact VideoMAE-S filename and SHA-256 are repository constants and are
  bound through manifest, full precheck, training config, checkpoint metadata,
  checkpoint selection, test opening, profiling, descriptor, and analysis.
- Each resolution/seed profile acquires a unique atomic `*.started.json`
  marker before dataset/model work. A crash leaves the marker in place, while
  the other eight matrix cells remain independent.
- Profile protocol v4 hashes full node, CPU, GPU UUID/PCI/driver/state, Python,
  PyTorch/CUDA/cuDNN/NCCL, OpenMMLab, Decord, NumPy, and FFmpeg identities. The
  complete 3x3 cost matrix must share these identities.
- Test-open certificate v3 requires the complete 3x3 selection matrix to share
  the same precheck file SHA, precheck internal SHA, and frozen pretrained SHA.

## Local evidence before final review

- S1 tests: `21 passed`.
- Required C3 plus S1 focused regression: `41 passed`.
- Python compilation, shell syntax, config validator, static precheck, and
  `git diff --check`: passed.
- Formal full-CUDA precheck: not run.
- S1 training/test/profile matrix: not run.

## Final review

The fourth blocking pass found three P1 issues and one P2 issue: checkpoint
metadata validation was incomplete, the final report was not deterministically
sealed from all nine descriptors, test opening was scoped only to a mutable run
root, and profile order was registered but not enforced. These were fixed with
a shared checkpoint validator, real writer regression, deterministic report
rebuild, canonical experiment namespaces, and a frozen profile schedule.

A later bypass test showed that equivalent precheck JSON could change its file
hash and create a second experiment namespace. The once-only marker was moved
to the preregistered `sealed_study_v1` root, independent of commit, precheck,
and experiment namespace. The launcher now validates profile order and the
same hardware/software fingerprint before opening each cell's test.

Final reviewer verdict: `PASS_BEFORE_REMOTE_TRAINING`.

- Reviewer: `019f5b3c-9b7f-73f2-8eea-00157a60a119`
- Model/effort: `gpt-5.6-sol` / `max`
- Findings: no P0, P1, or P2
- Independent reproduction: `26` S1 tests passed
- Combined focused regression: `46` tests passed

This verdict means the S1 infrastructure is ready for the formal remote gate.
It is not S1 GO, empirical support, or a paper-ready result. Full CUDA precheck,
the 3x3 training matrix, sealed test, and trained-checkpoint cost remain unrun.
