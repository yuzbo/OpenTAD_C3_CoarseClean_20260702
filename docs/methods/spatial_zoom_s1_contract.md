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
- Result gate: `tools/bata/analyze_spatial_zoom_s1_results.py`
- Slurm launchers: `scripts/run_spatial_zoom_s1_precheck_slurm.sh`,
  `scripts/run_spatial_zoom_s1_train_slurm.sh`, and
  `scripts/run_spatial_zoom_s1_test_profile_slurm.sh`
- Tests: `tests/test_spatial_zoom_s1_infrastructure.py`

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
   and equal successful optimizer updates. A skipped AMP update fails the run.
6. Save raw gate predictions for every eligible checkpoint, then use the
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
   zero workers, 50 warmups, a persistent 20 ms power sampler with raw trace,
   and one canonical output prefix. Cost claims are limited to same-node,
   same-GPU, warm serial per-window latency and gross GPU energy; they do not
   represent cold-start, whole-video latency, incremental energy, or CPU and
   storage energy.
9. Bind checkpoint, prediction, marker, certificate, manifest, config, profile,
   Git commit, precheck, and internal/file hashes into one
   descriptor per resolution and seed.
10. Recompute full class AP under a paired Bayesian video-cluster bootstrap
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
- Local syntax checks pass. The combined S1/train-iteration suite reports
  `43 passed, 1 skipped`, and the required C3 regression reports `20 passed`.
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
  backbone/projection/head gradients. Formal jobs `1165669-1165677` now run the
  fresh 3x3 matrix. S1 is only `experiment_running`; no S1 GO, crop-model
  success, empirical support, or paper claim is allowed.
