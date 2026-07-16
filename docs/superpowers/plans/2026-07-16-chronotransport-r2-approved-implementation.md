---
type: implementation_plan
protocol: CT-P3R-3S-r2
authority_commit: 537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37
authority_sha256: E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6
spec_review: APPROVE_SPEC_FOR_PLAN
implementation_review: REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
status: active
date: 2026-07-16
---

# ChronoTransport r2 approved implementation and experiment plan

## Scope and invariant

This plan implements the exact A1--A4-amended specification at commit `537f692`. It does not change
seeds, splits, candidate library, thresholds, update counts, bootstrap units, official population or
the Gate 1 → 2 → 3 → Stage C/matched → post-Stage-C Gate 3 → Gate 4 stop-chain.

The current branch is not implementation `I`; registration is `NOT_READY`. No PRECHECK or formal GPU
job is allowed until work packages W0--W7 pass exact-byte verification and an independent implementation
review approves creation of I/R.

## Claim map

| Claim | Minimum evidence | Blocking stage |
|---|---|---|
| H1: the frozen library has same-cost oracle headroom over fixed periodic schedules | registered 23-schedule full-stack profile plus Gate 1 PASS | Gate 1 |
| H2/H3: TRANSPORT/HOLD risk is learnable, calibrated and safe without inference leakage | registered Stage B plus Gates 2/3 PASS on frozen windows | Gates 2/3 |
| H4: learned CT improves real end-to-end speed while protecting high-IoU localization | Stage C/matched completion and official-population Gate 4 PASS | Gate 4 |

Anti-claims to rule out are proxy cost, requested/executed mismatch, GT/teacher/replay leakage, static
periodic gains mislabeled as learned scheduling, hidden transaction overhead and repeated timing samples
misused as metric samples.

## Work packages

### W0 — Plan, source inventory and frozen authority

- Mark the 2026-07-12 GPU1 plan superseded.
- Freeze an explicit classification for all tracked `tests/test_chronotransport*.py` files. Every file is
  exactly one of `formal_r2_source`, `legacy`, or `nonformal`; unclassified additions fail closed.
- Include the classification artifact itself, all formal producers, validators, launchers and hardening
  tests in the registration exact-source vector.
- RED: omit a tracked test, add an unclassified test, or leave a formal entrypoint outside the vector.

Acceptance: current 21-test inventory is exhaustively classified and registration rejects any mismatch.

### W1 — A1 registration authority and random controls

- Move registration authority to commit `537f692` and its exact spec hash.
- Bind unsuffixed `random_p2/p4/p8` to integer seed 3407 in candidate factory config, generated action
  bytes/hash, replay identity and recomputation validation.
- Remove the historical unconditional random lock only after exact-seed validation is reachable from every
  formal entrypoint.
- Migrate all flat/legacy aliases to the canonical nested registration fields.

Acceptance: positive seed-3407 generation/revalidation passes; missing, string, alternate or caller-only
seed evidence fails before profiling.

### W2 — A2 Slurm single-device and observed environment

- Formal launchers never set, append or replace `CUDA_VISIBLE_DEVICES`.
- Each producer asserts one torch-visible device and uses only logical `cuda:0`.
- A shared producer-owned observer records raw scheduler variables, Job/Step identity, torch device count,
  GPU model/UUID, driver, CUDA, PyTorch, cuDNN and precision; caller JSON/environment aliases are not
  accepted as evidence.
- Registration freezes required model/software policy. Each formal artifact binds the live allocation,
  step and UUID observed under its writer lock.
- Remove physical-GPU1 semantics from Gate 1 shell/backend, full-stack CLI, Stage-B CLI, Gates-2/3 claims
  and every current formal launcher. Legacy launchers remain unreachable and classified.

Acceptance: synthetic no-Slurm/multi-device/mismatch tests fail; an allocated one-device precheck records
the unchanged scheduler visibility and a stable observed identity.

### W3 — Lexical filesystem, source and execution identity

- Use one shared component-wise lexical path validator before any resolution.
- Open mandatory inputs with no-follow semantics; hash and deserialize the same opened bytes; bind device,
  inode, regular-file mode and byte length.
- Validate again inside exclusive writer locks; publish terminal artifacts with atomic no-clobber and
  inode-safe cleanup/recovery.
- Bind clean detached worktree, Git blob, imported module origin/bytes and formal entrypoint to the same
  registered source vector. A source hash for bytes Python did not import is invalid.

Acceptance: parent/leaf symlink, hard-link replacement, TOCTOU, import shadowing, pathname reload and
partial-publication tests fail closed.

### W4 — Real ActionFormer per-window task loss

- Extend the real configured ActionFormer/AnchorFreeHead path to expose an exact batch-length task-loss
  vector from the same logits, targets, valid mask, shared normalizer and dynamic loss weight as the
  unchanged aggregate objective.
- Preserve the legal registered loss namespace, including configured auxiliary losses; do not freeze the
  reviewer's example dataclass or invent a detector feature boundary.
- Assert the vector reduction reproduces the existing detector task loss exactly for batch size two.

Acceptance: real batch-two integration proves aggregate identity, differentiability and one head pass;
permutation, empty-positive and dynamic-weight cases are covered.

### W5 — A3/A4 Stage-C paired transaction

- Snapshot RNG and all mutable state.
- Run exactly one no-grad forced-dense model reference; retain detached per-window losses/features; restore
  every dense-only state/RNG change, including `loss_normalizer`.
- Run exactly one differentiable counterfactual model forward and one risk-predictor forward using its
  audited runtime signals/actions.
- Define each target as `max(L_cf_window - L_dense_window, 0)` and preserve the official aggregate LD.
- On success, only A/T/R parameter transitions and exactly one normalizer progression persist; CT and
  matched-dense traces remain bitwise aligned. On overflow, everything except scaler backoff is restored.
- Audit and separately measure transaction overhead.

Acceptance: real ActionFormer CPU integration, protected CUDA FP32/AMP, retry/overflow/resume and forged
loss/action/signal/buffer tests pass.

### W6 — Formal Stage-C, matched-dense and Gate-4 workflows

- Implement repository-owned Stage-C and matched-dense runners with 4,200 successful updates per seed,
  exact exposure ledger, common batch/augmentation/LR/EMA/normalizer trace, atomic checkpoints and resume.
- Matched dense trains A only, forces dense, preserves the shadow candidate ledger and shares the registered
  starting checkpoint.
- Implement post-Stage-C Gate-3 rerun/unlock.
- Implement a repository-owned Gate-4 producer over the official full-video/sliding-window population,
  official OpenTAD evaluation, raw predictions/GT identity, matched D/C/S timing blocks and live profiler.
- Keep the existing caller-mapping Gate-4 adjudicator test-only.

Acceptance: formal builders cannot accept caller-owned rows/evidence; one-update CUDA smokes and interrupted
resume tests pass before any full run.

### W7 — Verification, implementation I and registration R

- Run local syntax/static checks and all focused ChronoTransport tests.
- In an isolated remote checkout, run the complete ChronoTransport suite, required C3 compatibility tests,
  protected CUDA parity/AMP/overflow/one-update smokes and launcher behavior checks.
- Record exact source hashes and remote commands/results in the research wiki; production/test byte changes
  invalidate earlier GREEN evidence.
- Obtain an independent exact-byte implementation review. Only an approval permits a clean pushed
  implementation commit I.
- Create R as exactly one single-parent successor of I whose only diff is one canonical registration
  artifact; validate tree/blob/source/import/environment/checkpoint/manifest identities.

Acceptance: clean detached `HEAD=R`, `parent(R)=I`, exact one-file I..R, registration `READY`, no unresolved
review blocker.

## Formal experiment execution

### E0 — PRECHECK

Run under a Slurm-allocated single GPU only after R. PRECHECK validates all identities and uses synthetic or
registered non-official probes only; it cannot create reusable Gate evidence.

### E1 — Gate 1

Profile all 23 schedules with 50 warmups and at least 200 complete windows per schedule using real
full-stack `total_ms`. Adjudicate oracle headroom and the 20% dense-saving requirement. FAIL permanently
freezes the route and stops E2--E5.

### E2 — Stage B

For seeds 3407/3408/3409, train the unique risk predictor on the 140 fit windows under registered exposure,
paired replay and 140 successful-update completion contracts. Calibration/evaluation remain untouched until
the complete Stage-B phase artifact exists.

### E3 — Gates 2/3

Use the 30 calibration and 30 evaluation windows only. Gate 2 tests matched TRANSPORT vs HOLD. Gate 3 tests
window-level ranking, simultaneous marginal coverage/support and learned scheduler superiority without
GT/teacher/replay access. Either FAIL freezes the route and stops E4--E5.

### E4 — Stage C and matched dense

For every seed, run CT and matched dense from the same registered checkpoint with identical ordered
materialized batches, augmentation, LR, successful updates, EMA and normalizer trace. Complete 4,200
successful updates, then rerun the required post-Stage-C Gate 3. Failure stops Gate 4.

### E5 — Gate 4

Evaluate the official full-video population. Hard conditions are the frozen specification conditions,
including latency-saving 95% lower bound ≥15%, mAP@0.7 loss 95% upper bound ≤1.5, short-action loss ≤1.5,
actual heavy-compute reduction, learned CT no slower than the best fixed static strategy and improved
detector regret. Any FAIL permanently freezes the route.

## Run-order tracker policy

Only one stage may be `RUNNING`. Downstream entries remain `LOCKED`, not `TODO`, until their upstream
terminal artifact validates. Each status transition records commit/R, Slurm Job/Step, run root, artifact
hash and terminal marker in `research-wiki/experiments/chronotransport-r2-execution-tracker.md`.

## Verification commands

At minimum, before I/R:

```bash
python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py
python -m pytest tests/test_chronotransport*.py -q
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q
```

Formal CUDA and training commands are emitted only by R-bound launchers inside Slurm; no login-node CUDA or
training command is permitted.
