# PhysTime-TAD 2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a registered, trainable, and auditable offline TAD detector that maps irregular feature observations with explicit support intervals directly to detections in physical seconds.

**Architecture:** A strict geometry layer converts explicit observation supports into non-overlapping ownership intervals and globally aligned physical query cells. Every pyramid level uses support-overlap mass attention to project the original irregular observations onto a regular seconds-based grid, followed by a physical-time anchor-free head with integrated endpoint likelihood. A dedicated detector owns seconds-based GT validation, optional paired-view consistency, optimizer groups, and seconds-preserving post-processing.

**Tech Stack:** Python 3, PyTorch, MMEngine registry/config, OpenTAD losses/NMS/datasets, pytest, Slurm CUDA smoke.

---

### Task 1: Freeze the 2.0 contract

**Files:**
- Create: `docs/superpowers/specs/2026-07-10-phystime-tad-2-design.md`
- Create: `docs/superpowers/plans/2026-07-10-phystime-tad-2.md`

- [ ] Verify that the specification names seconds as canonical, requires explicit support intervals, forbids support expansion over gaps, and separates feature-token and raw-video evidence tracks.
- [ ] Verify that the plan contains no unfinished placeholder markers.
- [ ] Read both documents once from top to bottom and confirm every mandatory gate maps to an implementation task.

### Task 2: Implement strict physical-time geometry with TDD

**Files:**
- Create: `tests/test_phystime_geometry.py`
- Create: `opentad/models/utils/phystime_geometry.py`
- Modify: `opentad/models/utils/__init__.py`

- [ ] Write failing tests for shape validation, strict timestamp ordering, support containment, no expansion during ownership clipping, padding invariance, globally aligned query cells, `K`-independent query count, and zero mass across true gaps.
- [ ] Run `python -m pytest tests/test_phystime_geometry.py -q` and confirm import failure for `phystime_geometry`.
- [ ] Implement `validate_physical_observations`, `clip_to_ownership_intervals`, `build_physical_query_pyramid`, `support_overlap_mass`, and `geometry_from_metas` using tensor operations only.
- [ ] Run `python -m pytest tests/test_phystime_geometry.py -q` and expect all tests to pass on a Torch-capable host.

### Task 3: Implement support-integrated measure attention with TDD

**Files:**
- Create: `tests/test_phystime_measure_attention.py`
- Create: `opentad/models/projections/phystime_projection.py`
- Modify: `opentad/models/projections/__init__.py`

- [ ] Write failing tests for exact constant-kernel support-split invariance, unchanged output under padding, finite zero output for uncovered cells, finite backward gradients, and direct per-level projection from the original observations.
- [ ] Run `python -m pytest tests/test_phystime_measure_attention.py -q` and confirm the registered class is missing.
- [ ] Implement `SupportIntegratedMeasureAttention` with overlap mass outside the exponential and stable normalization over covered observations.
- [ ] Implement `PhysTimeMeasureProjection.forward(x, masks, metas)` returning `(feat_list, mask_list, geometry_list)` with each level independently projected from `x`.
- [ ] Run the focused tests and expect all tests to pass.

### Task 4: Implement the physical-time detection head with TDD

**Files:**
- Create: `tests/test_phystime_head.py`
- Create: `opentad/models/dense_heads/phystime_head.py`
- Modify: `opentad/models/dense_heads/__init__.py`

- [ ] Write failing tests for physical point construction, seconds-based target assignment, positive ordered segment decoding, integrated endpoint probability scaling, empty-GT stability, and non-zero gradients for class/regression/endpoint branches.
- [ ] Run `python -m pytest tests/test_phystime_head.py -q` and confirm `PhysTimeHead` is missing.
- [ ] Implement masked classification/regression towers, physical regression ranges, DIOU regression, focal classification, integrated endpoint BCE, and pre-NMS raw output return.
- [ ] Run the focused tests and expect all tests to pass.

### Task 5: Implement the detector and seconds post-processing with TDD

**Files:**
- Create: `tests/test_phystime_detector.py`
- Create: `opentad/models/detectors/phystime_tad.py`
- Modify: `opentad/models/detectors/__init__.py`
- Modify: `opentad/models/utils/post_processing/utils.py`

- [ ] Write failing tests that build `PhysTimeTAD` through the registry, reject non-seconds GT and leakage metadata, preserve absolute seconds in post-processing, compute one `cost` term without double counting, cover every trainable optimizer parameter, and backpropagate through projection and every head branch.
- [ ] Write a paired-view test that checks consistency only on the common valid physical query mask and gives finite gradients.
- [ ] Run `python -m pytest tests/test_phystime_detector.py -q` and confirm expected missing behavior.
- [ ] Implement shared `_forward_features`, train/test paths, paired-view consistency, strict inference audit, and generic AdamW decay/no-decay optimizer grouping.
- [ ] Add `prediction_time_unit="seconds"` handling to `convert_to_seconds` with clamp-only semantics.
- [ ] Run the focused tests and expect all tests to pass.

### Task 6: Implement auditable feature-token data transforms with TDD

**Files:**
- Create: `tests/test_phystime_data_pipeline.py`
- Create: `opentad/datasets/transforms/phystime.py`
- Modify: `opentad/datasets/transforms/__init__.py`
- Modify: `opentad/datasets/transforms/formatting.py`
- Modify: `opentad/datasets/transforms/loading.py`

- [ ] Write failing tests that preserve crop origin through `RandomTrunc`, convert local THUMOS GT to absolute seconds, retain original support cells after token dropping, expose real gaps, and reject raw sparse-frame token groups without contiguous support provenance.
- [ ] Run `python -m pytest tests/test_phystime_data_pipeline.py -q` and confirm missing transforms/metadata.
- [ ] Make `RandomTrunc` record `phystime_window_start_feature_idx` without changing ordinary outputs.
- [ ] Implement `SampleIrregularFeatureObservations` with GT-independent uniform/random/bursty/contiguous-gap policies and original-index provenance.
- [ ] Implement `BuildPhysTimeFeatureGeometry` using original token ownership cells of width `snippet_stride/fps`, absolute seconds GT conversion, and explicit unit/provenance flags.
- [ ] Add PhysTime keys to `Collect` and run focused tests.

### Task 7: Add a real configuration and deployment gates

**Files:**
- Create: `configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py`
- Create: `tools/bata/run_phystime_tad_precheck.py`
- Create: `scripts/run_phystime_tad_gate0b_gpu1.sh`
- Create: `tests/test_phystime_config_precheck.py`
- Create: `docs/methods/phystime_tad_contract.md`

- [ ] Write a failing config test that loads the MMEngine config, builds the registered model, and asserts there is no selector, ledger, actionness, budget, or selected-axis remap.
- [ ] Add a THUMOS I3D feature config using `SampleIrregularFeatureObservations`, `BuildPhysTimeFeatureGeometry`, `PhysTimeMeasureProjection`, and `PhysTimeHead`.
- [ ] Add a CPU/CUDA precheck that constructs synthetic irregular observations, runs train backward and inference, verifies optimizer coverage and seconds output, and emits a JSON audit record.
- [ ] Add a GPU1-only Slurm launcher that loads the repository environment and runs the precheck without starting a full training job.
- [ ] Run local static checks: `python -m py_compile` for every new Python file and `python -m pytest tests/test_phystime_config_precheck.py -q` where Torch is available.

### Task 8: Run gates and freeze the first deployable commit

**Files:**
- Modify only files listed in Tasks 2-7 if a gate exposes a defect.

- [ ] Run all PhysTime tests on the remote OpenTAD environment and record the exact pass count.
- [ ] Submit the GPU1 precheck, wait for completion, and inspect stdout/stderr for `Traceback`, OOM, non-finite values, `FAIL`, and seconds-contract violations.
- [ ] Run the repository's existing focused physical-grid tests to check regressions.
- [ ] Inspect `git diff --check`, `git status --short`, and ensure unrelated untracked DUCA review/profile files are not staged.
- [ ] Commit only the PhysTime-TAD 2.0 implementation after all mandatory gates pass.
