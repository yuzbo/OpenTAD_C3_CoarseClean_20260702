# ChronoTransport CT-P3R-3S-r2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the independently approved CT-P3R-3S-r2 protocol, verify every behavioral contract on the remote server, freeze immutable registration commit `R`, and deploy the first physical-GPU1 experiment only as far as the hard gate chain permits.

**Architecture:** Keep ChronoTransport isolated under its existing namespace. Add small pure protocol/statistics modules around the existing cache, scheduler, runtime, replay, and training code; preserve AdaTAD's dense embedding, block ordering, full-row temporal adapter, detector grid, head, and NMS. All decisions are offline window-level decisions. Registration and gate outputs are machine-readable, hash-bound artifacts; requested and executed actions/costs remain separate.

**Tech Stack:** Python 3, PyTorch, NumPy/SciPy, OpenTAD/AdaTAD, pytest, shell launchers, Slurm, Git/GitHub.

---

## Global execution contract

- [ ] Work only on `codex/chronotransport-r2-implementation` in the isolated worktree.
- [ ] Treat the approved r2 spec at commit `e4422f5` and SHA-256 `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8` as normative.
- [ ] Update `research-wiki/` and `research-wiki/log.md` in the same turn as every design, implementation, test, experiment, or claim-state transition.
- [ ] Run local static, hash, and text checks only. Run every Python behavior test, integration test, CUDA test, precheck, and experiment remotely.
- [ ] Use the declared OpenTAD environment under `/data/run01/sczc063/yuzibo`; formal jobs must use physical GPU1 through a protected allocation and must never train on the login node.
- [ ] Obey Gate 1 → Gate 2 → Gate 3 → Stage C → Gate 4 stop-chain. A failed gate forbids downstream deployment.

## Task 1: Canonical r2 protocol, manifests, and exposure schedules

**Files:**

- Create: `opentad/models/chronotransport/protocol.py`
- Modify: `opentad/models/chronotransport/__init__.py`
- Create: `tests/test_chronotransport_r2_protocol.py`
- Create: `tools/bata/build_chronotransport_r2_manifest.py`

- [ ] Write remote failing tests for NFC UTF-8 canonical JSON hashing, split digests, one-window-per-video selection, exact tail rules, and immutable manifest fields.
- [ ] Run the focused test remotely and capture the expected import/assertion failure.
- [ ] Implement canonical serialization and digest functions with explicit byte contracts.
- [ ] Implement the r2 split/window manifest and validation without GT-dependent window selection.
- [ ] Implement Stage B's 140-success candidate exposure and Stage C's 4,200-success three-seed exposure.
- [ ] Verify per-seed candidate counts, position-mod-4 balance, exact tail sequences, and aggregate 525/candidate Stage C exposure remotely.
- [ ] Commit the smallest green protocol slice.

## Task 2: Frozen candidate library, controls, and dual-age contracts

**Files:**

- Modify: `opentad/models/chronotransport/actions.py`
- Modify: `opentad/models/chronotransport/scheduler.py`
- Modify: `opentad/models/chronotransport/cache.py`
- Create: `opentad/models/chronotransport/controls.py`
- Create: `tests/test_chronotransport_r2_actions_cache.py`

- [ ] Write remote failing tests for the exact 16-candidate order, action tensor shape/domain, motion-top-k exact count, seeded random exact count, effective feature-age, and executed-action-age.
- [ ] Run red remotely.
- [ ] Freeze the r2 library and reject reordered, duplicated, or malformed candidates.
- [ ] Implement deterministic motion-top-k and random exact-count controls with defined tie-breaking and seeds.
- [ ] Separate feature staleness from executed action age in the cache API.
- [ ] Run focused tests remotely and commit.

## Task 3: Runtime semantics and dense parity

**Files:**

- Modify: `opentad/models/chronotransport/runtime.py`
- Modify: `opentad/models/chronotransport/transport.py`
- Modify: `opentad/models/chronotransport/cost_lookup.py`
- Create: `tests/test_chronotransport_r2_runtime.py`

- [ ] Write remote failing tests for current-row live tensors, detached historical cache, all-row adapter writeback, requested/executed separation, repair/fallback ledgers, and forced-dense parity.
- [ ] Run red remotely.
- [ ] Change block execution so heavy-path actions do not suppress full-row AdaTAD adapter output.
- [ ] Enforce current-row live-tensor and historical-cache detach boundaries.
- [ ] Record requested action/cost separately from executed action/cost after repair or fallback.
- [ ] Make unsafe cache, missing calibration, invalid cost, and unsupported execution deterministically fall back to dense.
- [ ] Verify tensor parity and backward connectivity remotely; commit.

## Task 4: Window-level quantile risk head

**Files:**

- Modify: `opentad/models/chronotransport/risk.py`
- Create: `tests/test_chronotransport_r2_risk.py`

- [ ] Write remote failing tests for the D=23 per-cell input, 8-D action embedding, 8-D layer-group embedding, mean/max window pooling, monotone quantiles, and 16-candidate batched output.
- [ ] Run red remotely.
- [ ] Implement the single schedule-conditioned window-level quantile head exactly as specified.
- [ ] Remove incompatible cellwise-sum risk aggregation from the r2 path while retaining legacy import compatibility where needed.
- [ ] Verify gradients and output invariants remotely; commit.

## Task 5: Paired replay and formal Stage B training

**Files:**

- Modify: `opentad/models/chronotransport/replay.py`
- Modify: `opentad/models/chronotransport/formal_stage_b.py`
- Modify: `opentad/models/chronotransport/losses.py`
- Create: `tools/bata/train_chronotransport_r2_stage_b.py`
- Create: `tests/test_chronotransport_r2_stage_b.py`

- [ ] Write remote failing tests for identical RNG restoration, dense/candidate paired windows, exactly 140 successful optimizer updates, exposure indexing by successful update, retry non-advancement, and checkpoint provenance.
- [ ] Run red remotely.
- [ ] Implement paired replay snapshots for Python, NumPy, CPU CUDA RNG, and data-transform state used by the loader.
- [ ] Implement Stage B loss composition and successful-update accounting.
- [ ] Emit row-level train ledger with video/window/candidate/seed/checkpoint/digest provenance.
- [ ] Run a remote CPU behavioral test and a one-update CUDA smoke test; commit.

## Task 6: Gate 1 oracle-headroom adjudication

**Files:**

- Create: `opentad/models/chronotransport/adjudication.py`
- Create: `tools/bata/run_chronotransport_r2_gate1.py`
- Create: `tests/test_chronotransport_r2_gate1.py`

- [ ] Write remote failing synthetic tests for equal-cost oracle selection, cost-bin matching, hierarchical video bootstrap, all hard conditions, and fail-closed missing evidence.
- [ ] Run red remotely.
- [ ] Implement Gate 1 statistics and a schema-validated JSON decision artifact.
- [ ] Ensure no learned predictor, GT, teacher, or raw-prediction cache participates in inference decisions.
- [ ] Verify remotely and commit.

## Task 7: Gate 2/3 calibration, baselines, ranking, and scheduler

**Files:**

- Modify: `opentad/models/chronotransport/formal_stage_b.py`
- Modify: `opentad/models/chronotransport/scheduler.py`
- Modify: `opentad/models/chronotransport/adjudication.py`
- Create: `tools/bata/run_chronotransport_r2_gates23.py`
- Create: `tests/test_chronotransport_r2_gates23.py`

- [ ] Write remote failing tests for conformal ranks 28/30 and 127/140, matched TRANSPORT/HOLD, fit-only schedule-conditioned constant baseline, window-level Spearman/bootstrap, simultaneous marginal coverage, support margins, and fail-closed scheduler selection.
- [ ] Run red remotely.
- [ ] Implement calibration using only the frozen calibration split.
- [ ] Implement matched Gate 2 and the exact Gate 3 ranking/coverage/support contracts.
- [ ] Implement the fixed-budget risk-constrained scheduler with dense fallback.
- [ ] Run synthetic and fixture-backed tests remotely; commit.

## Task 8: Stage C parameter ownership and AMP retry

**Files:**

- Modify: `opentad/models/chronotransport/training.py`
- Create: `opentad/models/chronotransport/stage_c.py`
- Create: `tests/test_chronotransport_r2_stage_c.py`

- [ ] Write remote failing tests for object-identity ownership, disjoint parameter sets, loss-specific scaled autograd, scaler overflow detection, exact retry rollback, RNG restoration, and successful-update counters.
- [ ] Run red remotely.
- [ ] Implement object-identity parameter partitioning and reject missing/duplicate ownership.
- [ ] Implement the specified AMP gradient algorithm without cross-loss gradient leakage.
- [ ] Snapshot and restore model, optimizer, scaler, RNG, exposure cursor, and ledger cursor on overflow; cap retries and fail closed.
- [ ] Run remote CPU simulations and CUDA overflow tests; commit.

## Task 9: Stage C and matched-dense runners

**Files:**

- Create: `tools/bata/train_chronotransport_r2_stage_c.py`
- Create: `tools/bata/train_chronotransport_r2_matched_dense.py`
- Create: `configs/adatad/thumos/e2e_actionformer_videomae_s_768x1_160_adapter_chronotransport_r2_stage_b.py`
- Create: `configs/adatad/thumos/e2e_actionformer_videomae_s_768x1_160_adapter_chronotransport_r2_stage_c.py`
- Create: `configs/adatad/thumos/e2e_actionformer_videomae_s_768x1_160_adapter_chronotransport_r2_matched_dense.py`
- Create: `tests/test_chronotransport_r2_runners.py`

- [ ] Write remote failing config/runner tests for frozen Stage B checkpoint consumption, 4,200 successful updates, fixed candidate exposures, matched dense controls, EMA/checkpoint identity, and resume determinism.
- [ ] Run red remotely.
- [ ] Implement Stage C and matched-dense runners without dynamic action policy training.
- [ ] Emit immutable per-seed manifests and row-level ledgers.
- [ ] Run remote precheck and one-successful-update CUDA smoke for each runner; commit.

## Task 10: Full-stack cost profiling and Gate 4

**Files:**

- Modify: `opentad/models/chronotransport/profiler.py`
- Create: `opentad/models/chronotransport/full_stack_profiler.py`
- Create: `tools/bata/profile_chronotransport_r2_full_stack.py`
- Create: `tools/bata/run_chronotransport_r2_gate4.py`
- Create: `tests/test_chronotransport_r2_gate4.py`

- [ ] Write remote failing tests for six-order crossover profiling, warmup/sample accounting, total invocation cost, official-video population, matched timing, seed-level mAP bootstrap, and the approved detector-regret hierarchical bootstrap statistic.
- [ ] Run red remotely.
- [ ] Implement full-stack invocation profiling and immutable cost lookup provenance.
- [ ] Implement Gate 4 hard-condition adjudication with all confidence intervals and fail-closed missing timing/calibration/cache evidence.
- [ ] Run synthetic tests remotely and a small GPU1 profiling precheck; commit.

## Task 11: Immutable registration, claim unlocks, and repository contract

**Files:**

- Create: `opentad/models/chronotransport/registration.py`
- Create: `tools/bata/register_chronotransport_r2.py`
- Create: `tools/bata/validate_chronotransport_r2.py`
- Create: `tests/test_chronotransport_r2_registration.py`
- Modify: `tests/test_chronotransport_repository_contract.py`

- [ ] Write remote failing tests for pre-Gate1 registration completeness, commit/hash binding, immutable fields, implementation commit `I`, registration commit `R`, artifact schemas, and gate-based claim unlocks.
- [ ] Run red remotely.
- [ ] Implement registration creation/validation and refuse dirty/unpushed/mismatched inputs.
- [ ] Implement claim-state transitions without promoting beyond actual evidence.
- [ ] Run remotely and commit implementation commit `I`.
- [ ] Create the immutable registration artifact in a separate registration commit `R`, then push both.

## Task 12: GPU1 launchers and static safety guards

**Files:**

- Create: `scripts/run_chronotransport_r2_precheck_gpu1.sh`
- Create: `scripts/run_chronotransport_r2_gate1_gpu1.sh`
- Create: `scripts/run_chronotransport_r2_stage_b_gpu1.sh`
- Create: `scripts/run_chronotransport_r2_gates23_gpu1.sh`
- Create: `scripts/run_chronotransport_r2_stage_c_gpu1.sh`
- Create: `scripts/run_chronotransport_r2_gate4_gpu1.sh`
- Create: `tests/test_chronotransport_r2_launchers.py`

- [ ] Write remote failing text/behavior tests for `CUDA_VISIBLE_DEVICES=1`, protected-allocation checks, environment activation, clean commit pinning, upstream gate artifacts, and no login-node training.
- [ ] Run red remotely.
- [ ] Implement guarded launchers and precheck-only mode.
- [ ] Run local text/hash checks, then remote launcher behavior tests and `PRECHECK_ONLY=1`; commit.

## Task 13: Remote verification matrix and documentation closure

**Files:**

- Modify: `research-wiki/experiments/chronotransport.md`
- Modify: `research-wiki/ideas/chronotransport.md`
- Modify: `research-wiki/query_pack.md`
- Modify: `research-wiki/anti_repetition.md`
- Modify: `research-wiki/log.md`
- Modify: `research-wiki/graph/edges.jsonl`

- [ ] Sync the pinned clean commit to the remote writable root.
- [ ] Run remote `py_compile`, all focused r2 pytest files, existing ChronoTransport regression tests, and the required project focused checks.
- [ ] Run remote CUDA parity, backward, AMP-overflow, resume, and one-update smoke tests.
- [ ] Save only schema-approved evidence under the remote run directory; do not add logs/checkpoints/data to Git.
- [ ] Update the wiki with exact commit, commands, pass/fail evidence, and honest status (`implemented`/`tested` only where demonstrated).
- [ ] Push the verified implementation and documentation commits.

## Task 14: Formal deployment under the stop-chain

**Files:**

- Remote artifacts only; wiki updates in the files listed in Task 13.

- [ ] Submit the physical-GPU1 Gate 1 job from a protected allocation using registration commit `R`.
- [ ] Monitor to terminal state and validate every Gate 1 artifact/hash/schema.
- [ ] If Gate 1 fails, stop and record the negative result; do not deploy Stage B or later gates.
- [ ] If Gate 1 passes, submit Stage B, then Gate 2/3; validate and stop on any failed hard condition.
- [ ] Only after Gate 3 passes, submit Stage C plus matched dense, then Gate 4.
- [ ] Record raw results, adjudication, uncertainty, cost, and unlocked claims in the wiki without calling the method paper-ready unless Gate 4 and evidence-integrity checks pass.

## Final verification checklist

- [ ] Git worktree is clean and all intended commits are pushed.
- [ ] Approved spec hash, implementation commit `I`, registration commit `R`, checkpoint hashes, split/window manifests, seeds, cost lookup, and gate artifacts form a complete provenance chain.
- [ ] Existing C3/DUCA paths remain unchanged except shared compatibility fixes backed by regression tests.
- [ ] No validation/test GT, teacher outputs, raw-prediction cache, or counterfactual ledger affects inference-time scheduling.
- [ ] The final report states exactly which stage is implemented, remotely tested, running, passed, failed, or blocked.
