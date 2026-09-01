---
doc_id: EVALUATOR_DUCA_P0_E1_REPLACEMENT_READONLY_PRE_RUN_ADMISSION_FACT_RECHECK
version: v001
status: PRE_RUN_FACTS_PASS_PENDING_COORDINATOR_ADMISSION
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T221712Z-c7b2f9d3e5a4
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_fact_check: EVALUATOR_DUCA_P0_E1_PHASE_SPECIFIC_READONLY_PRE_RUN_ADMISSION_FACT_CHECK-v001
shared_parent_admission: DUCA_P0_E1_SHARED_PARENT_ADMISSION-v001
parent_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v005
parent_manifest_critic: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V005_FOCUSED_RECHECK-v001
fresh_reserved_execution_identity: msg-20260812T220919Z-12f7e8a4d9c3
fresh_identity_state: OPAQUE_UNREAD_UNOPENED_NOT_CONSUMED_NOT_DISPATCHED_NOT_EXECUTED
old_execution_identity: msg-20260812T214652Z-7b2c8a5e61d4
old_identity_state: FAIL_CLOSED_PERMANENTLY_INELIGIBLE
phase: E1_INDEPENDENT_REFERENCE_FREEZE
fixed_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: p0_e1_readonly_pre_run_fact_recheck
execution_state: NOT_EXECUTED
aggregate_fact_verdict: PASS
pre_run_ready: NO_PENDING_COORDINATOR_ADMISSION_RECORD
---

# E1 replacement read-only PRE_RUN admission fact recheck

This record consumes only durable recheck queue
`msg-20260812T221712Z-c7b2f9d3e5a4`. It rechecks only facts 1 and 3. Facts 2
and 4 are reused exactly as `VERIFIED_BY_V001`; their checks were not repeated.

The fresh identity `msg-20260812T220919Z-12f7e8a4d9c3` remained opaque, unread,
unopened, unconsumed, undispatched, and unexecuted. Its queue payload was not
opened or inspected. Its disposition is taken only from the cited v005 manifest
and Critic v005 PASS receipt. The old identity remains fail-closed and
permanently ineligible.

## Per-fact and aggregate verdict

| formal fact | verdict | evidence basis |
|---|---|---|
| 1. Phase-specific clean Evaluator binding | `VERIFIED` | The registered workspace is exact, HEAD is frozen revision `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, and porcelain output is empty. |
| 2. Exact sealed 18/9/6 Evaluator input/root state | `VERIFIED_REUSED_FROM_V001` | Reused without any repeated input, root, fixture, or provenance check. |
| 3. Shared-parent/output/duplicate/identity state | `VERIFIED` | The literal pre-admitted shared parent is a real mode-`700` directory resolving to itself; all E1/B1/E2 output, partial, comparison, and E1-receipt paths are absent; the process metadata query found zero active E1/B1/E2 matches; v005/Critic v005 preserve the old fail-closed identity and fresh opaque identity states. |
| 4. Registered N16R4 CPU transport/interpreter/cwd availability | `VERIFIED_REUSED_FROM_V001` | Reused without repeating transport, interpreter, or cwd availability checks. |

Aggregate literal fact verdict: `PASS`.

Formal `PRE_RUN_READY` is not issued by this Evaluator record. It remains
`NO_PENDING_COORDINATOR_ADMISSION_RECORD`. No queue or command is admitted until
a later versioned Coordinator admission record cites this PASS and changes the
formal admission state.

## Fact 1 — exact read-only command and result

Registered Evaluator workspace:

~~~text
C:\Users\skywalker\.codex\worktrees\30f3\OpenTAD_C3_CoarseClean_20260702
~~~

Exact command:

~~~sh
git rev-parse --show-toplevel
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
~~~

Exact result:

~~~text
C:/Users/skywalker/.codex/worktrees/30f3/OpenTAD_C3_CoarseClean_20260702
63a726a4aaf48ecbf6780bb196de43a890c6b4df

exit_status=0
~~~

The empty third-command output establishes zero porcelain entries. With exact
registered workspace and HEAD, there is no worktree/index foreign or Builder
mutation. Fact 1 is `VERIFIED`.

## Fact 3 — exact bounded remote metadata command and result

The existing registered `N16R4` SSH transport was used only to reach the remote
metadata surface required by fact 3. No transport/interpreter/cwd availability
fact was retested. No fixture/reference/projector/adapter content was read.

Exact remote command body:

~~~sh
phase_parent=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001
reference_leaf=evaluator-reference
partial_leaf=${reference_leaf}.partial
production_leaf=builder-production
comparison_leaf=evaluator
comparison_leaf=${comparison_leaf}-comparison
reference_root=${phase_parent}/${reference_leaf}
partial_root=${phase_parent}/${partial_leaf}
production_root=${phase_parent}/${production_leaf}
comparison_root=${phase_parent}/${comparison_leaf}
receipt_leaf=DUCA_P0_E1_REFERENCE_RECEIPT
receipt_leaf=${receipt_leaf}-v001.json
e1_receipt=${reference_root}/${receipt_leaf}
if [ -d "$phase_parent" ] && [ ! -L "$phase_parent" ]; then
  stat -c "SHARED_PARENT|%n|type=%F|mode=%a|perm=%A|uid=%u|gid=%g" "$phase_parent"
  printf "SHARED_PARENT_REAL=%s\n" "$(readlink -f "$phase_parent")"
else
  printf "SHARED_PARENT=MISSING_OR_NOT_LITERAL_DIRECTORY\n"
fi
for absent_path in "$reference_root" "$partial_root" "$production_root" "$comparison_root" "$e1_receipt"; do
  if [ -e "$absent_path" ] || [ -L "$absent_path" ]; then
    printf "REQUIRED_ABSENT=FAIL|%s\n" "$absent_path"
  else
    printf "REQUIRED_ABSENT=PASS|%s\n" "$absent_path"
  fi
done
reference_name=DUCA_P0_REFERENCE_PROJECTOR
reference_name=${reference_name}-v001.py
production_name=run_duca_p0_projection
production_name=${production_name}_production.py
comparison_name=evaluator
comparison_name=${comparison_name}-comparison
phase_pattern="${reference_name}|${production_name}|${comparison_name}"
if pgrep -af "$phase_pattern"; then
  printf "ACTIVE_E1_B1_E2_PROCESS_MATCHES=NONZERO\n"
else
  printf "ACTIVE_E1_B1_E2_PROCESS_MATCHES=0\n"
fi
~~~

Invocation envelope:

~~~text
ssh -o BatchMode=yes -o ConnectTimeout=15 N16R4 '<exact command body above>'
~~~

Exact result:

~~~text
SHARED_PARENT|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001|type=directory|mode=700|perm=drwx------|uid=1258|gid=1258
SHARED_PARENT_REAL=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference.partial
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference/DUCA_P0_E1_REFERENCE_RECEIPT-v001.json
ACTIVE_E1_B1_E2_PROCESS_MATCHES=0
exit_status=0
~~~

The remote filesystem/process metadata closes the physical parent, absence,
prior-receipt, and active-phase conditions. The cited v005 and Critic v005
artifacts close the identity-disposition condition without opening either
identity queue. Fact 3 is `VERIFIED`.

## Reused facts and non-execution boundary

- Fact 2: `VERIFIED_REUSED_FROM_V001`; no fixture, sealed-root, membership,
  order, or content check repeated.
- Fact 4: `VERIFIED_REUSED_FROM_V001`; no interpreter or working-directory
  check repeated.
- Fresh identity payload: `UNREAD_AND_UNOPENED`.
- Fresh identity consumption/dispatch/execution: `NONE`.
- Old identity: `FAIL_CLOSED_PERMANENTLY_INELIGIBLE`.
- E1/B1/E2/C1/P0: `NOT_EXECUTED`.
- Python/import/compile/test: `NOT_EXECUTED`.
- Fixture/reference/projector/adapter content: `NOT_ACCESSED`.
- Data/model/checkpoint/metric: `NOT_ACCESSED`.
- GPU/CUDA/Slurm/browser: `NOT_USED`.
- Output or execution receipt creation: `NONE`.
- Remote mutation: `NONE`.
- Subagents/probes/extra role processes: `NONE`.
- Scientific/protocol change: `NONE`.

## Durable disposition

- Fact 1: `VERIFIED`.
- Fact 2: `VERIFIED_REUSED_FROM_V001`.
- Fact 3: `VERIFIED`.
- Fact 4: `VERIFIED_REUSED_FROM_V001`.
- Aggregate fact verdict: `PASS`.
- Formal PRE_RUN_READY: `NO_PENDING_COORDINATOR_ADMISSION_RECORD`.
- Execution authority: `NOT_GRANTED`.

EVALUATOR_DECISION: P0_E1_REPLACEMENT_READONLY_FACTS_PASS_PENDING_COORDINATOR_ADMISSION.
