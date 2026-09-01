---
doc_id: BUILDER_DUCA_P0_CORRECTED_REMOTE_CLEAN_BINDING
version: v001
stage: DRAFT_P0_IDENTITY_PRE_RUN
author_role: builder
parent_message_id: msg-20260812T204130Z-5087b7d515b0
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_implementation: BUILDER_DUCA_P0_PRE_STATIC_001_IMPLEMENTATION-v001
parent_critic_return: CRITIC_DUCA_P0_PRE_STATIC_001_FOCUSED_IMPLEMENTATION_RECHECK-v001
implementation_revision: 402e582c4092275877828fd36b5b97a8c1213011
direct_parent: df8228072b871adbd8dedb480e80f1f7daaca69e
base_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
status: NEEDS_ATTENTION
evidence_class: blocked_pre_result
---

# Builder corrected remote clean binding — NEEDS_ATTENTION

Consumed exactly durable queue `msg-20260812T204130Z-5087b7d515b0`.
Ordinary Git transport succeeded, but the queue's required base-diff cardinality
does not match the frozen Git graph. The remote checkout was therefore not
created.

## Git transport evidence

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Ref: `refs/heads/codex/duca-p0-identity-execution-v002`
- Ref resolved on registered N16R4 to:
  `402e582c4092275877828fd36b5b97a8c1213011`
- Intended fresh checkout path:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_402e582c_20260813`
- Target preflight: `TARGET_ABSENT`
- Checkout creation: `NOT_EXECUTED`
- Existing checkout
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_df822807_20260813`:
  `UNTOUCHED`

The published transport ref was not deleted or rewritten because the queue
forbids fallback or cleanup outside a newly created target.

## Exact static transcript and blocking mismatch

Requested direct-parent diff, which matches the accepted correction:

```text
$ git diff --name-status df8228072b871adbd8dedb480e80f1f7daaca69e..402e582c4092275877828fd36b5b97a8c1213011
D tests/test_duca_p0_projection_policy.py
A tools/bata/run_duca_p0_projection_production.py
```

Requested net base diff, which contains two paths rather than the queue-required
three:

```text
$ git diff --name-status 63a726a4aaf48ecbf6780bb196de43a890c6b4df..402e582c4092275877828fd36b5b97a8c1213011
M opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py
A tools/bata/run_duca_p0_projection_production.py
```

The deletion of `tests/test_duca_p0_projection_policy.py` is visible versus the
direct parent `df822807...`, but that test did not exist in base `63a726a...`.
It therefore cannot be a third path in the net `63a726a...402e582c` diff.

The exact queue condition requiring a three-path net base diff is unsatisfied.
No alternate diff interpretation, checkout creation, fallback transport,
scientific change, or cleanup was attempted.

## Zero-execution attestation

- Python/import/compile/tests/adapter/projector: `NOT_EXECUTED`
- P0/C1-C3/E1-E2: `NOT_EXECUTED`
- Fixtures/materialization/reference/comparison: `NOT_ACCESSED_OR_EXECUTED`
- Data/model/checkpoint/metric: `NOT_ACCESSED`
- GPU/CUDA/Slurm/browser/Pro/Sources: `NOT_USED`
- Remote activity: `SSH_STATIC_SHELL_AND_ORDINARY_GIT_TRANSPORT_ONLY`
- Scope deviation: `none`

`P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED: BLOCKED_BINDING_CONTRACT_MISMATCH`.

Coordinator attention is required to correct or explicitly replace the
three-path base-diff requirement before a fresh remote checkout can be created.

