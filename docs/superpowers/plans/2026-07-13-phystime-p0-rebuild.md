# PhysTime P0 Rebuild Implementation Plan

> **Execution note:** follow this plan in order. G1b/G2 are deliberately out of
> scope until the G1a pilot is interpreted.

**Goal:** replace the invalid K=384-as-feature assumption with an auditable
K384 -> J192 -> Q native pipeline and deploy a matched selected-axis versus
physical-metric pilot.

**Architecture:** preserve raw-frame atomic provenance in the data pipeline,
align raw masks to native tubelet tokens after VideoMAE, then reuse the exact
same ActionFormer projection/head and seconds-domain post-processing in both
arms. One transform supplies either uniform rank-derived seconds or physical
native-token seconds through the same metadata key.

**Stack:** PyTorch, MMAction/OpenTAD registries, pytest, Slurm on N16R4.

## Task 1: Native Tubelet Geometry Contract

**Files:**
- Modify: `opentad/datasets/transforms/phystime_raw.py`
- Modify: `opentad/datasets/transforms/formatting.py`
- Test: `tests/test_phystime_native_tubelet_geometry.py`

1. Write failing tests for K/J grouping, prefix padding, odd valid counts,
   disconnected patch atoms, padding-repeat compute slots, canonical-seconds
   GT preservation, and fail-closed provenance.
2. Add `BuildPhysTimeNativeTubeletGeometry` with explicit tubelet/chunk
   contracts and no sampling decisions.
3. Expose only auditable metadata through `Collect`.

## Task 2: Post-Backbone Native Mask Alignment

**Files:**
- Add: `opentad/models/utils/native_temporal_geometry.py`
- Modify: `opentad/models/detectors/actionformer.py`
- Test: `tests/test_phystime_native_actionformer_contract.py`

1. Write failing tests proving that [B, C, 192] features reject a 384 mask
   without the explicit native geometry hook.
2. Implement strict raw-mask-to-tubelet-mask reduction and metadata count
   validation behind an opt-in config.
3. Keep all existing ActionFormer configs behavior-identical by default.

## Task 3: Matched G1a Configurations

**Files:**
- Add: `configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py`
- Add: `configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py`
- Modify: `opentad/models/dense_heads/anchor_free_head.py`
- Test: `tests/test_phystime_g1a_configs.py`

1. Remove post-backbone interpolation and set `max_seq_len=192` in both arms.
2. Extend physical-grid metadata keys narrowly so both arms consume the same
   seconds-axis key while differing only in that tensor's values.
3. Assert model type, parameter schema, backbone, projection, loss, K/J/Q,
   optimizer, and sampling parity; allow only coordinate-mode differences.

## Task 4: G0 Auditor and Real Gate

**Files:**
- Add: `tools/bata/audit_phystime_g0_native_geometry.py`
- Add: `tools/bata/run_phystime_g1a_real_gate.py`
- Test: `tests/test_phystime_g0_audit.py`
- Test: `tests/test_phystime_g1a_real_gate_contract.py`

1. Emit a versioned G0 JSON with K/J/Q, checksums, atom gaps, envelope
   inflation, parameter parity, and forbidden-metadata checks.
2. Build each official config on a real THUMOS sample, run finite
   forward/backward, and verify expected gradients and candidate counts.
3. Fail closed on missing data, checkpoint, CUDA, or parity evidence.

## Task 5: Slurm Pilot Deployment

**Files:**
- Add: `scripts/run_phystime_g1a_gate_slurm.sh`
- Add: `scripts/run_phystime_g1a_pilot_slurm.sh`
- Add: `scripts/submit_phystime_g1a_pilot.sh`
- Test: `tests/test_phystime_g1a_deployment.py`

1. Require a non-empty scheduler-provided CUDA mask, clean snapshot manifest,
   fixed commit, real data roots, and pretrained checkpoint. Use logical
   `cuda:0` inside the allocation and never overwrite Slurm's device mask.
2. Submit one gate followed by two `afterok` matched pilot jobs.
3. Record job IDs, config hashes, run root, and artifact paths.

## Task 6: Verification and Memory Update

1. Run `py_compile` and focused tests locally where possible.
2. Run the complete focused suite in the remote OpenTAD environment.
3. Commit and push a clean branch, create a clean remote snapshot, rerun the
   real gate from that snapshot, then submit the pilot only after gate success.
4. Update `research-wiki/experiments/phystime-performance-drop-diagnosis.md`,
   `research-wiki/query_pack.md`, `research-wiki/anti_repetition.md`, and
   `research-wiki/log.md` with exact status levels and job IDs.
