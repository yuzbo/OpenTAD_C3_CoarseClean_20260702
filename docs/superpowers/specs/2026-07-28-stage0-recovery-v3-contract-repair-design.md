# Stage-0 Recovery v3 Contract Repair

**Date**: 2026-07-28

**Status**: `user_approved / implementation_authorized`

**Scope**: engineering recovery only; no model, dataset, budget, evaluator, NMS,
or paper-claim protocol change.

## 1. Problem statement

Immutable recovery transaction
`duca_rime_recovery_0ab242f3_20260728_201613` exposed two independent
fail-closed defects after its code gate passed:

1. `run_duca_rime_phase1_uniform_eval.sh` did not require, hash-check, apply, or
   receipt-bind the absolute VideoMAE initialization used by every other
   Phase-1 evaluator. Both exact-uniform budgets therefore fell back to the
   repository-relative base-config path during actual inference.
2. Dense ActionFormer/TriDet salvage ran checkpoint compaction and detector
   evaluation, but `tools/test.py` treated their registered `training`
   development subset as a generic `validation` evaluation. Structured evidence
   finalization failed before checkpoint evidence and recovery receipts existed.

Both failures are deployment/evidence-contract failures. They provide no model
performance evidence.

## 2. Considered repairs

### A. Add dense protocols to the ordinary RIME formal protocol set

Rejected. It would route dense configs into code that requires
`duca_rime_variant` and `duca_rime_contract`, neither of which exists for dense
references. It fixes one branch condition by creating a later false contract.

### B. Add an explicit dense-reference evaluation role

Selected. The two dense formal protocols receive a separate predicate and a
dedicated validator. `tools/test.py` uses that role for exact-checkout gating,
terminal-EMA requirements, registered training-subset normalization, a
dense-reference evidence schema, and an explicit engineering-only payload. It
does not enter the trainable RIME identity branch.

### C. Create a second detector evaluation entrypoint

Rejected for v3. A separate evaluator would duplicate checkpoint loading,
official mAP execution, saved-prediction handling, and evidence serialization,
increasing semantic-drift risk without changing the required computation.

## 3. Uniform evaluator repair

`run_duca_rime_phase1_uniform_eval.sh` must:

1. require `DUCA_RIME_PRETRAIN_PATH` and `DUCA_RIME_PRETRAIN_SHA256`;
2. hash-check the initialization before precheck or inference;
3. resolve the same absolute path in `PRECHECK_ONLY=1`;
4. pass
   `model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}` in the actual
   `tools/test.py` command;
5. record the initialization SHA-256 in its terminal evaluation receipt.

The parent Phase-1 pipeline already requires and checks both variables. No
dataset, checkpoint, seed, budget, selector, or metric setting changes.

## 4. Dense-reference evaluation role

`tools.bata.duca_rime_training` owns a separate mapping from the two dense
protocols to their registered detector backends. Its validator must reject any
drift in:

- offline-TAD task and `dense_adatad_baseline` role;
- detector backend, selector absence, dense 768-window execution, and
  `with_cp=False`;
- `certification_development` evaluation role;
- `training` evaluator/dataset subset and identical non-empty block list;
- `official_final_subset_consumed=False`;
- saved predictions and engineering-only claim scope;
- absolute runtime VideoMAE path and its registered SHA-256.

`tools/test.py` must then:

- include this role in exact clean-commit and one-process official-evaluator
  gates;
- require epoch-59 EMA, saved predictions, and structured metrics;
- normalize against the registered `training` subset;
- emit `duca_rime_dense_reference_terminal_evaluation_v1`;
- record the validated dense contract, detector backend, pretrain identity,
  `uses_official_final=False`, and engineering-only claim scope;
- avoid all trainable-RIME variant/training-receipt logic.

## 5. Verification

Focused tests must prove:

1. uniform pretrain variables, SHA check, precheck resolution, actual override,
   and receipt binding all exist;
2. both dense protocols are recognized only by the dense-reference predicate;
3. both dense configs pass the dedicated validator with their exact backend,
   development block list, and absolute initialization;
4. protocol/backend, subset, block-list, official-final, selector, checkpointing,
   and pretrain drift fail closed;
5. `tools/test.py` contains the dedicated role, subset, schema, and payload
   routing without broadening ordinary `rime_formal`.

Local checks, authoritative remote Linux/Torch tests, every affected
`PRECHECK_ONLY=1` launcher, and one independent clean-commit deployment audit
must pass before submission.

## 6. Redeployment

The failed `0ab242f3` transaction and all partial artifacts remain immutable.
Deployment requires:

1. one new exact clean implementation commit;
2. a new clean remote checkout;
3. new commit-bound physical-protocol and dense-salvage manifests;
4. a fresh transaction root and submission manifest;
5. the same source raw checkpoints, with their existing failed job identities
   preserved;
6. Phase 4 disabled and official-final sealed.

The transaction becomes `experiment_running` only after `sbatch` returns the
new job IDs and the released submission receipt exists. It becomes successful
only when Phase-1, both dense recovery, Phase-2, and Phase-3 terminal receipts
all exist. Partial checkpoint, evaluation, or salvage sidecars never substitute
for those receipts.

