# BUILDER_FOCUSED_CORRECTION_DUCA_DENSITY_UOR-v001

```yaml
receipt_version: BUILDER_FOCUSED_CORRECTION_DUCA_DENSITY_UOR-v001
message_id: msg-20260814T004525Z-3e02701138a3
role: Builder
status: FOCUSED_IMPLEMENTATION_CORRECTION_COMPLETE
finding_classification: IMPLEMENTATION_CORRECTION
scientific_ambiguity: NONE
next_owner: Critic
execution_state: NOT_EXECUTED
scientific_evidence_status: BLOCKED_PRE_RESULT
decision: NOT_APPLIED
```

## Immutable correction binding

- Worktree: `C:/Users/skywalker/.codex/worktrees/8712/OpenTAD_C3_CoarseClean_20260702-builder-a6bdc084`
- Parent implementation: `7f07e4545fafda5ca9b86ead14a089b3515a06d0`
- Focused correction commit: `6576789468c1a7692d49b2ba94a638e01e7970f4`
- Commit subject: `Close DUCA reachability pre-run bindings`
- Post-commit porcelain count: `0`
- New files: `0`
- Scope deviation: `none`

## Corrected Critic findings

### DUCA-REACH-IMP-001 — loader firewall

- Detector-FIT and reader-FIT resolved configs bind every train/val/test loader to the exact FIT annotation, FIT video root, class map, and `training` subset.
- `tools/train.py` now constructs no val/test dataset or loader for the two frozen FIT phases.
- CAL val/test loaders explicitly bind the exact CAL annotation, CAL video root, class map, and `training` subset; CAL evaluator ground truth is bound to the CAL annotation.
- Focused resolved-config tests verify these bindings and the no-validation-loader entrypoint branch without opening any dataset path.

### DUCA-REACH-IMP-002 — launch, gate, and artifact binding

- The authored launcher uses supported single-process distributed invocations: `torchrun --standalone --nnodes=1 --nproc_per_node=1`.
- Detector and reader terminal checkpoints are bound to the actual OpenTAD layout `gpu1_id0/checkpoint/epoch_59.pth`; CAL predictions are bound to `gpu1_id0/result_detection.json` under each arm root.
- The loaded resolved config is canonically identified and compared to a mandatory nonempty gate identity.
- Before loader/checkpoint/output access, the gate compares exact argv, cwd, phase paths, annotations, data roots, class map, checkpoints, workdirs, artifacts, active arm, Slurm job ID, `CUDA_VISIBLE_DEVICES`, and single-process rank environment.
- After OpenTAD appends its workdir suffix, train/test entrypoints compare the actual derived workdir to the frozen declared workdir before creating output.

### DUCA-REACH-IMP-003 — sealed evaluator inputs and final publication

- Each CAL arm seals its complete root recursively and publishes a separate immutable completion receipt at the exact PRE_RUN-declared path.
- Every predecessor receipt must be a non-writable sealed file before the next phase can start.
- The evaluator accepts only the three exact declared immutable receipt paths, derives prediction artifacts from those receipts, and rejects writable roots/receipts, missing inputs, root mismatch, or artifact substitution before importing the evaluator.
- FIT/CAL manifest, CAL ground truth, arm roots/artifacts/receipts, and dedicated final-result root must match the PRE_RUN package.
- The final result is written inside a sealed temporary directory and atomically renamed to the absent dedicated final-result root; existing or unsealed final roots are rejected.

## Changed tracked paths

1. `configs/adatad/thumos/duca_density_cal_uor_n16r4.py`
2. `configs/adatad/thumos/duca_density_detector_fit_n16r4.py`
3. `configs/adatad/thumos/duca_density_reader_fit_n16r4.py`
4. `opentad/utils/training_guard.py`
5. `scripts/run_duca_density_reachability_n16r4.sh`
6. `tests/test_duca_density_reachability_contract.py`
7. `tools/bata/duca_density_reachability.py`
8. `tools/test.py`
9. `tools/train.py`

## Focused verification

```text
command: git diff --check
result: PASS

command: conda run -n open_mmlab python -m pytest tests/test_duca_density_reachability_contract.py -q
result: 21 passed, 1 warning in 2.62s
warning: existing environment NumPy API initialization warning; no contract test failed

post-commit git status --porcelain=v1
result: empty
```

Focused substitutions and seal checks cover:

- FIT/CAL resolved loader roots and annotations;
- FIT no-validation-loader construction;
- resolved-config, argv, cwd, environment, and actual-workdir mismatch rejection;
- mutable predecessor receipt rejection;
- writable arm-root and completion-receipt rejection;
- substituted prediction artifact rejection;
- unsealed/existing final-root rejection and sealed atomic publication.

## No-execution boundary

```yaml
data_or_dataset_root_accessed_or_listed: false
official_validation_accessed_or_listed: false
checkpoint_or_model_accessed: false
remote_invoked: false
gpu_cuda_slurm_invoked: false
training_run: false
inference_run: false
evaluator_or_bootstrap_run: false
metric_or_result_computed: false
browser_sources_pro_used: false
claim_or_scientific_route_changed: false
another_role_created_or_dispatched: false
```

The launcher, entrypoint gates, evaluator, and final publication code remain authored future surfaces only. This receipt grants no PRE_RUN admission or execution authority.

BUILDER_TERMINAL: `FOCUSED_CORRECTION_COMPLETE_BLOCKED_PRE_RESULT_NEXT_OWNER_CRITIC`
