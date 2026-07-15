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
   3-config matrix. Static/clip mode does not unlock training.
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
   and one canonical output prefix.
9. Bind checkpoint, prediction, marker, certificate, manifest, config, profile,
   Git commit, precheck, and internal/file hashes into one
   descriptor per resolution and seed.
10. Recompute full class AP under paired video-cluster and training-seed
    bootstrap. Require parity with the official THUMOS evaluator, apply a
    one-sided simultaneous max-T lower bound across 224 and 256, report
    boundary error, and use measured full-stack cost in resolution freezing.

## Current Verification

- Resolved-config validator: passed locally.
- Required focused tests: `46 passed`; S1 tests: `26 passed`.
- Static geometry precheck: passed for 160/224/256.
- Local real clip: blocked by the known Windows `c10.dll` failure.
- Formal CUDA full-window precheck and S1 training: not run.

Repeated independent Max reviews first returned `FAIL_BEFORE_REMOTE_TRAINING`
and exposed protocol/provenance bypasses. After remediation, the same
`gpt-5.6-sol`/`max` reviewer returned `PASS_BEFORE_REMOTE_TRAINING` with no
P0/P1/P2 finding.
Thus the S1 infrastructure is locally `tested`; the route remains `designed`.
No S1 GO, empirical support, or paper claim is allowed.
