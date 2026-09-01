---
doc_id: BUILDER_DUCA_P0_REMOTE_CLEAN_BINDING
version: v001
stage: DRAFT
author_role: builder
source_message: msg-20260812T200722Z-ab8c37824043
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
frozen_revision: df8228072b871adbd8dedb480e80f1f7daaca69e
direct_parent: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
status: BOUND_CLEAN_NOT_EXECUTED
evidence_class: infrastructure_evidence
created_at: 2026-08-12T20:12:35Z
---

# Builder P0 remote clean binding receipt

## Binding

- Canonical project root: `E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702`
- Builder replacement clean local snapshot: `C:\Users\skywalker\.codex\worktrees\cc5e\OpenTAD_C3_CoarseClean_20260702`
- Git common directory: `E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.git`
- Local actual revision: `df8228072b871adbd8dedb480e80f1f7daaca69e`
- Local porcelain entries: `0`
- Access used: ordinary Git transport and remote clean-checkout creation only
- Formal clean requirement: `true`
- Git transport ref: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git refs/heads/codex/duca-p0-identity-execution-v001`
- Remote path: `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_df822807_20260813`
- Outcome: `PASS_REMOTE_CLEAN_BINDING`

The transport ref resolved on N16R4 to
`df8228072b871adbd8dedb480e80f1f7daaca69e`. A new clone was created at the
remote path above and checked out detached at that exact revision.

## Static verification transcript summary

```text
git rev-parse HEAD
df8228072b871adbd8dedb480e80f1f7daaca69e

git rev-parse HEAD^
63a726a4aaf48ecbf6780bb196de43a890c6b4df

git diff --name-status 63a726a4aaf48ecbf6780bb196de43a890c6b4df df8228072b871adbd8dedb480e80f1f7daaca69e
M\topentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py
A\ttests/test_duca_p0_projection_policy.py

git status --porcelain=v1
<no output; zero entries>
```

## Scope and handoff

No Python/import/test command, C1-C3/E1-E2 step, fixture materialization or
invocation, production/reference/comparison execution, data/model/checkpoint/
metric access, GPU/CUDA/Slurm/browser operation, or scientific-definition change
occurred. `labctl.py verify-binding` was not invoked because this bounded task
explicitly prohibited Python; the concise Git binding fields required for the
handoff are recorded above. The Coordinator remains the only role authorized to
ingest this return into canonical project state or register a subsequent formal
execution assignment. This receipt does not advance P0, PRE_RUN, or any claim.

