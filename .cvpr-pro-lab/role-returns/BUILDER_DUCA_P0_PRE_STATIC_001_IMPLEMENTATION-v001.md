---
doc_id: BUILDER_DUCA_P0_PRE_STATIC_001_IMPLEMENTATION
version: v001
stage: DRAFT_P0_IDENTITY_PRE_RUN
author_role: builder
parent_message_id: msg-20260812T203153Z-34285d5fcbda
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_critic_return: CRITIC_DUCA_P0_IDENTITY_PRE_RUN_STATIC_DEPENDENCY_RECHECK-v001
parent_plan: BUILDER_DUCA_P0_P0_PRE_STATIC_001_MINIMAL_CHANGE_PLAN-v001
finding: P0-PRE-STATIC-001
parent_revision: df8228072b871adbd8dedb480e80f1f7daaca69e
implementation_revision: 402e582c4092275877828fd36b5b97a8c1213011
status: IMPLEMENTED_NOT_EXECUTED
evidence_class: infrastructure_evidence
---

# Builder DUCA P0 PRE-STATIC-001 implementation receipt

Consumed exactly durable queue `msg-20260812T203153Z-34285d5fcbda` and
implemented only the accepted two-path correction for deterministic finding
`P0-PRE-STATIC-001`.

## Frozen lineage

- Direct parent: `df8228072b871adbd8dedb480e80f1f7daaca69e`.
- Child implementation commit: `402e582c4092275877828fd36b5b97a8c1213011`.
- Commit subject: `fix: add DUCA P0 production JSONL adapter`.
- Worktree after commit: zero `git status --porcelain=v1` entries.
- Remote transport or checkout: `NOT_EXECUTED`.

## Exact repository delta

```text
D tests/test_duca_p0_projection_policy.py
A tools/bata/run_duca_p0_projection_production.py
```

Git static summary:

```text
2 files changed, 162 insertions(+), 482 deletions(-)
```

No production projector or other repository path was changed by this child
commit. The frozen integer projector remains in
`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py` exactly as
in the parent revision.

## Implemented formal interface

`tools/bata/run_duca_p0_projection_production.py` is the sole formal production
JSONL adapter. Static source inspection establishes that it:

1. accepts only required `--input-jsonl` and `--output-jsonl` payload options;
2. admits only LF-terminated JSON object rows with exact ordered fields
   `T,K,Q,u,a` and exact `Q=1048576`;
3. invokes `project_duca_fixed_targets_v001` exactly once per admitted row;
4. preserves every nonempty `DUCAProjectionError.code` as the negative output
   status and fails closed on an untyped or unexpected failure;
5. emits Builder-owned rows containing the source line number, exact input JSON
   text, typed status, `p`, `E2`, `E_infinity`, `E1`, `U1`, ascending candidate
   order, and `scope_deviation="none"`;
6. completes all in-memory row projection before atomically publishing the
   output and rejects an output path inside the sealed Evaluator package;
7. imports only the frozen production projector plus Python standard-library
   modules; it imports no pytest, Evaluator, reference, fixture, comparison,
   data, model/checkpoint, metric, GPU/CUDA, Slurm, or browser surface.

The removed pytest module is not the formal gate interface, and no Evaluator-
owned reference/source/support artifact was copied into the Builder snapshot.
The sealed Evaluator root, index, mutation definitions, normative reference,
fixture membership/order `18/9/6`, expected values, and comparison remain under
independent Evaluator ownership and were not read, modified, or executed during
implementation.

## Static checks and execution boundary

- `git diff --cached --check`: passed before commit.
- exact staged and committed name-status: the two paths above.
- exact direct parent: confirmed by `git rev-parse HEAD^`.
- post-commit porcelain count: zero.
- Python/import/compile/pytest/tests/adapter: `NOT_EXECUTED`.
- Production/reference/comparison/P0 gate: `NOT_EXECUTED`.
- Fixture materialization: `NOT_EXECUTED`.
- Remote/SSH/data/model/checkpoint/metric: `NOT_ACCESSED_OR_EXECUTED`.
- GPU/CUDA/Slurm/browser/Pro/Sources: `NOT_USED`.
- C1-C3/E1-E2: `NOT_EXECUTED`.
- Scope deviation: `none`.

Git hooks were bypassed for the commit so the explicit no-runtime restriction
could not be violated by repository hook execution.

## Critic handoff

`P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED: PENDING_INDEPENDENT_CRITIC_RECHECK`.

This receipt is implementation/static evidence only. It does not authorize or
establish projector identity, optimality, a metric, cost, efficacy, P1
admission, or a paper claim. A later Coordinator-authorized clean rebind and
separate durable execution queue remain required before any production,
reference, or comparison command.

