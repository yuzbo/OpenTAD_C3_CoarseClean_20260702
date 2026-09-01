# ChronoTransport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the reviewed offline 48-clip × layer-group ChronoTransport runtime, detector-level paired replay, transport/risk training integration, validation, and GPU1 deployment gates.

**Architecture:** Keep patch embedding, AdaTAD temporal adapters, detector grid, head, and postprocessing dense. Dynamically gather only RECOMPUTE clip rows through VideoMAE heavy attention/MLP; synthesize other rows from latest-cache TRANSPORT or bitwise HOLD, then run the original dense adapter innovation. Train risk from same-batch, same-RNG dense/counterfactual detector regret and fail closed without measured schedule cost, calibration, or a dedicated checkpoint.

**Tech Stack:** Python, PyTorch, OpenTAD, MMEngine/MMAction2, pytest, Bash, Slurm, THUMOS14.

---

### Task 1: Record and reconcile the reviewed contract

**Files:**
- Modify: docs/superpowers/specs/2026-07-10-chronotransport-design.md
- Create: docs/methods/2026-07-10-chronotransport-implementation-plan.md
- Modify: AGENTS.md
- Modify: RTK.md

- [ ] Verify the corrected spec distinguishes 768 detector points from 384 internal tubelets.
- [ ] Verify v1 routes [B,48,G], keeps AdaTAD adapters dense, and uses latest-cache transport.
- [ ] Append the parallel-route repository contract without deleting C3/DUCA rules.
- [ ] Run: git diff --check.
- [ ] Commit only the contract and plan files.

### Task 2: RED — install focused tests before production code

**Files:**
- Create: tests/test_chronotransport_core.py
- Create: tests/test_chronotransport_repository_contract.py
- Create: tests/test_chronotransport_vit_adapter_integration.py

- [ ] Add the reviewed delivery tests unchanged after manifest verification.
- [ ] Run: python -m pytest tests/test_chronotransport_core.py tests/test_chronotransport_repository_contract.py tests/test_chronotransport_vit_adapter_integration.py -q.
- [ ] Confirm failures are caused by the absent chronotransport package/config/runtime, not syntax or import corruption.

### Task 3: GREEN — core actions, cache, transport, risk, and scheduler

**Files:**
- Create: opentad/models/chronotransport/actions.py
- Create: opentad/models/chronotransport/cache.py
- Create: opentad/models/chronotransport/transport.py
- Create: opentad/models/chronotransport/risk.py
- Create: opentad/models/chronotransport/scheduler.py
- Create: opentad/models/chronotransport/losses.py
- Create: opentad/models/chronotransport/profiler.py
- Create: opentad/models/chronotransport/__init__.py

- [ ] Implement strict [B,48,G] schedule and contiguous full-depth layer-group validation.
- [ ] Implement anchor/latest/recompute_age/source_time cache with HOLD bitwise invariance.
- [ ] Implement latest-conditioned zero-initialized low-rank transport.
- [ ] Implement quantile risk, split-conformal offset, one-sided regret, and pinball loss.
- [ ] Implement finite-library measured-cost selection and dense fail-closed guards.
- [ ] Run the focused core tests until green.

### Task 4: GREEN — runtime and VideoMAE integration

**Files:**
- Create: opentad/models/chronotransport/runtime.py
- Modify: opentad/models/backbones/vit_adapter.py
- Modify: opentad/models/__init__.py

- [ ] Implement forced-dense as the untouched original block loop.
- [ ] Implement mixed routing by gathering RECOMPUTE rows for heavy attention/MLP.
- [ ] Execute the original AdaTAD adapter densely at each block using the real h,w geometry.
- [ ] Scatter mixed rows back to the dense clip batch and update latest cache.
- [ ] Add packed-route mutual exclusion and legacy-checkpoint/dedicated-checkpoint guards.
- [ ] Run core and real adapter integration tests.

### Task 5: Stage-A config, validator, and protected launcher

**Files:**
- Create: configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py
- Create: tools/bata/validate_chronotransport_adatad.py
- Create: scripts/run_chronotransport_adatad_gpu1.sh

- [ ] Default to dense diagnostic precheck with every scientific/deployment claim false.
- [ ] Validate geometry, forbidden inference inputs, checkpoint readiness, measured-cost readiness, and output boundary.
- [ ] Fail closed unless CUDA_VISIBLE_DEVICES is physical GPU1 and formal runs have Slurm/protected allocation.
- [ ] Run validator and PRECHECK_ONLY launcher locally where dependencies permit.

### Task 6: RED/GREEN — detector-level paired counterfactual replay

**Files:**
- Create: tests/test_chronotransport_paired_replay.py
- Create: tools/bata/run_chronotransport_paired_replay.py
- Create: opentad/models/chronotransport/replay.py

- [ ] Write failing tests for identical dense regret≈0, RNG restoration, one-sided regret, compact ledger schema, deterministic hash, and rejection of raw predictions/full-token state.
- [ ] Run the new tests and confirm expected failures.
- [ ] Implement RNGSnapshot covering Python, NumPy, torch CPU, and torch CUDA state.
- [ ] Implement paired dense no-grad and counterfactual detector callbacks over the same batch.
- [ ] Persist only sample id, split, schedule, compact signals, pooled targets, costs, and regret labels.
- [ ] Run replay tests green.

### Task 7: RED/GREEN — Stage-B transport and calibrated-risk training

**Files:**
- Create: tests/test_chronotransport_stage_b.py
- Create: tools/bata/train_chronotransport_stage_b.py
- Create: opentad/models/chronotransport/training.py
- Create: configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py

- [ ] Write failing tests for frozen detector parameters with input gradients, split isolation, one-sided targets, no argmin backpropagation, and loss composition.
- [ ] Implement parameter-freeze helpers and explicit fit/calibration/evaluation split guards.
- [ ] Implement task, transport consistency, and pinball losses without duplicate detector loss.
- [ ] Implement checkpoint metadata for calibration and measured-cost readiness.
- [ ] Run Stage-B focused tests green.

### Task 8: RED/GREEN — Stage-C AdaTAD adapter joint training

**Files:**
- Create: tests/test_chronotransport_stage_c.py
- Create: configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_c.py
- Modify: opentad/models/chronotransport/training.py

- [ ] Write failing tests that only official AdaTAD adapters, transport, and risk parameters are trainable.
- [ ] Keep dense reference no-grad and frozen VideoMAE heavy parameters.
- [ ] Add three deterministic seed variants and prevent claim flags from unlocking.
- [ ] Run Stage-C focused tests green.

### Task 9: Nonlinear measured-cost lookup and profiler

**Files:**
- Create: tests/test_chronotransport_cost_lookup.py
- Create: tools/bata/profile_chronotransport_schedules.py
- Modify: opentad/models/chronotransport/profiler.py
- Modify: opentad/models/chronotransport/scheduler.py

- [ ] Write failing tests for hardware/precision/batch/schedule/selected-row/statistic keys and missing-key dense fallback.
- [ ] Implement JSON cost-table loading with schema/version/hardware guards.
- [ ] Record p50/p95 per schedule and all mandatory pipeline stages.
- [ ] Reject additive costs in learned/deployment mode.
- [ ] Run cost focused tests green.

### Task 10: Full local verification

**Files:**
- Verify all files above.

- [ ] Run py_compile for train/test and all new tools.
- [ ] Run all ChronoTransport focused tests.
- [ ] Run repository-required C3 focused tests to prove no regression.
- [ ] Run bash -n on the launcher.
- [ ] Run git diff --check and inspect all touched files.

### Task 11: N16R4 precheck and GPU1 deployment

**Files:**
- Deploy only changed source/config/test/launcher files under /data/run01/sczc063/yuzibo.

- [ ] Check GPU1 availability before assignment.
- [ ] Sync only permitted small source files; do not copy datasets, checkpoints, logs, archives, or unrelated dirty files.
- [ ] Activate the configured OpenTAD environment.
- [ ] Run real ViT adapter tests and PRECHECK_ONLY=1 launcher.
- [ ] In a legal Slurm/protected allocation, launch Stage-A smoke on physical GPU1 and verify process/GPU/log visibility.
- [ ] Launch paired replay only after Stage-A gate passes.
- [ ] Do not launch Stage B/C or claim success until their preceding gates pass.
