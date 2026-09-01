---
doc_id: EVALUATOR_DUCA_P0_E1_PHASE_SPECIFIC_READONLY_PRE_RUN_ADMISSION_FACT_CHECK
version: v001
status: PRE_RUN_BLOCKED
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T215130Z-7a4c8fd1e602
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v002
parent_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v004
parent_manifest_critic: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V004_FOCUSED_RECHECK-v001
sealed_materialization_receipt: EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001
coordinator_admission_candidate: DUCA_P0_E1_PRE_RUN_ADMISSION_CANDIDATE-v001
reserved_execution_queue_id: msg-20260812T214652Z-7b2c8a5e61d4
reserved_execution_queue_consumption_state: NOT_CONSUMED
reserved_execution_queue_dispatch_state: NOT_DISPATCHED
phase: E1_INDEPENDENT_REFERENCE_FREEZE
fixed_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: p0_e1_readonly_pre_run_fact_check
execution_state: NOT_EXECUTED
pre_run_state: BLOCKED
---

# E1 phase-specific read-only PRE_RUN admission fact check

This record consumes only durable fact-check queue
`msg-20260812T215130Z-7a4c8fd1e602`. It performs no E1/B1/E2/C1 execution and
grants no execution authority. The candidate argv was not issued. Python was
not invoked; no module was imported or compiled; no fixture content, data,
model, checkpoint, metric, GPU/CUDA/Slurm service, or browser was accessed.

## Aggregate result

`PRE_RUN_BLOCKED`.

| fact | status | decisive evidence |
|---|---|---|
| 1. Phase-specific clean Evaluator binding at frozen revision | `BLOCKED` | Registered workspace and HEAD are exact, but `git status --porcelain` contains two untracked control artifacts; the required clean binding is false. |
| 2. Exact sealed 18/9/6 Evaluator input/root state | `VERIFIED` | Literal Evaluator root is a non-writable directory; all four literal source/input paths are project-local non-writable regular files; the accepted sealed-materialization receipt supplies the exact frozen ordered `18/9/6` provenance. No fixture content was read. |
| 3. Shared-parent/output/duplicate-run/reserved-identity state | `BLOCKED` | All E1/B1/E2 output paths and the E1 receipt are absent, and the corrected process query found zero active E1/B1/E2 process matches; however, the required shared parent is absent. In addition, strict `UNOPENED` status for the reserved queue cannot be certified after two broad control-plane text searches may have scanned its JSON. |
| 4. Registered N16R4 CPU transport, interpreter, and cwd availability | `VERIFIED` | SSH transport connected as `sczc063` to `ln01`; the literal cwd exists as mode `550`; the literal interpreter symlink resolves to an executable regular file. The interpreter was not invoked. |

Facts 1 and 3 fail independently. No versioned Coordinator admission record may
change E1 to `PRE_RUN_READY` from this result.

## 1. Static authority and provenance evidence

- `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`: accepted, fixed revision
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`.
- `CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V004_FOCUSED_RECHECK-v001`:
  `PASS`, with all four admission facts still pending this read-only check.
- `.cvpr-pro-lab/state.json`: Evaluator process
  `019febf2-690b-7093-bcf5-8eb69636770e`, access `evaluation-only`, registered
  workspace
  `C:\Users\skywalker\.codex\worktrees\30f3\OpenTAD_C3_CoarseClean_20260702`.
- `EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001`: sealed
  remote root, exact positive order count `18`, negative order count `9`, and
  certificate-mutation order count `6`; its closeout recorded read-only
  permissions and no materialization rerun or repair.

## 2. Fact 1 command and result

Working directory:

~~~text
C:\Users\skywalker\.codex\worktrees\30f3\OpenTAD_C3_CoarseClean_20260702
~~~

Exact read-only command:

~~~sh
git rev-parse --show-toplevel
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
~~~

Exact result:

~~~text
C:/Users/skywalker/.codex/worktrees/30f3/OpenTAD_C3_CoarseClean_20260702
63a726a4aaf48ecbf6780bb196de43a890c6b4df
?? .cvpr-pro-lab/evaluator-returns/EVALUATOR_DUCA_DENSITY_P0P1-v001.md
?? .cvpr-pro-lab/messages/coordinator/msg-20260811T051246Z-c7c415c8b2f1.json
exit_status=0
~~~

Interpretation: the registered workspace and frozen revision match exactly, but
the porcelain count is `2`, not `0`. One entry is outside the current fact-check
artifact and one is Coordinator-owned. The required phase-specific clean
Evaluator binding, including no foreign mutation, is therefore `BLOCKED`. This
check did not alter or clean either entry.

## 3. Facts 2 and 3 remote path/metadata command and result

The existing registered SSH alias `N16R4` was used with `BatchMode=yes` and a
15-second connection timeout. The remote shell used only variable assignment,
`printf`, `hostname`, `id`, `stat`, `readlink`, shell existence/type tests, and a
process-metadata query. It performed no write and read no fixture content.

Authoritative path/metadata operations:

~~~sh
ssh -o BatchMode=yes -o ConnectTimeout=15 N16R4 '<read-only shell>'

# Inside the read-only shell, with the literal v004 paths bound to variables:
printf "TRANSPORT=CONNECTED\n"
printf "HOSTNAME="; hostname
printf "REMOTE_USER="; id -un
for checked_path in "$eval_root" "$reference_source" "$fixture_stream" "$fixture_index" "$mutation_defs"; do
  if [ -e "$checked_path" ] || [ -L "$checked_path" ]; then
    stat -c "PATH_META|%n|type=%F|mode=%a|perm=%A|uid=%u|gid=%g|size=%s" "$checked_path"
    printf "PATH_REAL|%s|%s\n" "$checked_path" "$(readlink -f "$checked_path")"
  else
    printf "PATH_MISSING|%s\n" "$checked_path"
  fi
done
if [ -d "$shared_parent" ] && [ ! -L "$shared_parent" ]; then
  stat -c "SHARED_PARENT|%n|type=%F|mode=%a|perm=%A|uid=%u|gid=%g" "$shared_parent"
  printf "SHARED_PARENT_REAL=%s\n" "$(readlink -f "$shared_parent")"
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
~~~

Exact authoritative result:

~~~text
TRANSPORT=CONNECTED
HOSTNAME=ln01
REMOTE_USER=sczc063
PATH_META|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001|type=directory|mode=550|perm=dr-xr-x---|uid=1258|gid=1258|size=4096
PATH_REAL|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001
PATH_META|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py|type=regular file|mode=444|perm=-r--r--r--|uid=1258|gid=1258|size=18256
PATH_REAL|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py
PATH_META|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl|type=regular file|mode=444|perm=-r--r--r--|uid=1258|gid=1258|size=66500
PATH_REAL|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl
PATH_META|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json|type=regular file|mode=444|perm=-r--r--r--|uid=1258|gid=1258|size=2840
PATH_REAL|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json
PATH_META|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json|type=regular file|mode=444|perm=-r--r--r--|uid=1258|gid=1258|size=1290
PATH_REAL|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json
SHARED_PARENT=MISSING_OR_NOT_LITERAL_DIRECTORY
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference.partial
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison
REQUIRED_ABSENT=PASS|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference/DUCA_P0_E1_REFERENCE_RECEIPT-v001.json
exit_status=0
~~~

Fact 2 interpretation: `VERIFIED`. The root resolves to itself and has no write
bits. Each exact v004 input resolves to itself, is a regular file, and has mode
`444`. The accepted sealed-materialization receipt—not a fresh fixture read—
supplies exact membership and order:

- positive `18`: `G16-U`, `G17-E2`, `G17-EINF`, `G17-E1`, `G17-U1`,
  `G17-PLEX`, `G31-U`, `G32-U`, `G383-U`, `G384-U`, `G385-X`, `G767-U`,
  `F768-U`, `F768-PERIODIC`, `F768-DISP16`, `F768-CONVEX`,
  `F768-CONCAVE`, `F768-ALT`;
- negative `9`: `N-T15`, `N-K`, `N-U-LEN`, `N-A-LEN`, `N-U-CANON`,
  `N-A-END`, `N-A-ORDER`, `N-INFEASIBLE`, `N-ARITH`;
- mutation `6`: `M-DUPLICATE`, `M-STRIDE5`, `M-DISP17`, `M-OBJECTIVE`,
  `M-SCALAR-TIE-LOSER`, `M-CANDIDATE-ORDER`.

Fact 3 interpretation: `BLOCKED`. Output and receipt absence passes, but the
literal shared parent does not exist as the required directory. That alone is a
hard block before any write or argv.

## 4. Fact 4 and active-process correction command/result

The literal interpreter is a symlink, so availability was checked without
invocation by resolving it and checking the target's regular-file and executable
metadata. Process patterns were assembled from fragments so the metadata shell
could not match its own command line.

Exact read-only command body:

~~~sh
interpreter=/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
if [ -e "$interpreter" ] || [ -L "$interpreter" ]; then
  stat -c "INTERPRETER_LITERAL|%n|type=%F|mode=%a|perm=%A|uid=%u|gid=%g|size=%s" "$interpreter"
  resolved_interpreter=$(readlink -f "$interpreter")
  printf "INTERPRETER_REAL=%s\n" "$resolved_interpreter"
  if [ -n "$resolved_interpreter" ] && [ -f "$resolved_interpreter" ] && [ -x "$interpreter" ]; then
    stat -L -c "INTERPRETER_TARGET_AVAILABLE|%n|type=%F|mode=%a|perm=%A|uid=%u|gid=%g|size=%s" "$interpreter"
  else
    printf "INTERPRETER_TARGET_AVAILABLE=FAIL\n"
  fi
else
  printf "INTERPRETER_LITERAL=ABSENT\n"
fi
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

Exact result:

~~~text
INTERPRETER_LITERAL|/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python|type=symbolic link|mode=777|perm=lrwxrwxrwx|uid=1258|gid=1258|size=10
INTERPRETER_REAL=/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python3.10
INTERPRETER_TARGET_AVAILABLE|/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python|type=regular file|mode=775|perm=-rwxrwxr-x|uid=1258|gid=1258|size=17504824
ACTIVE_E1_B1_E2_PROCESS_MATCHES=0
exit_status=0
~~~

The same metadata shell established the literal working directory:

~~~text
WORKING_DIRECTORY_AVAILABLE|/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source|type=directory|mode=550|perm=dr-xr-x---|uid=1258|gid=1258
WORKING_DIRECTORY_REAL=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source
~~~

Fact 4 interpretation: `VERIFIED`. Transport, cwd, and interpreter availability
are present. The interpreter was not invoked.

## 5. Discarded observations and reserved-identity conservatism

Two non-authoritative metadata observations were discarded:

1. The first process query matched its own shell because literal phase paths were
   present in that shell's command line. Its `NONZERO` line is invalid and is
   superseded by the fragment-assembled query returning exact count `0`.
2. The first interpreter guard required the literal path not to be a symlink and
   therefore returned `FAIL`; that condition was stricter than the queued
   availability fact. It is superseded by the non-invoking literal-link plus
   executable-target metadata above.

One correction attempt had local shell quoting error, exited `127`, produced no
usable fact, and caused no remote mutation. It is not used for admission.

The reserved execution queue was never deliberately opened, parsed, issued,
consumed, or dispatched, and no payload content from it was returned or used.
However, before the SSH fact check, two read-only content searches were scoped
across `.cvpr-pro-lab/**/*.{md,json,jsonl}` to locate registered N16R4 and frozen
revision evidence. Because such a search may open every candidate JSON file,
strict `UNOPENED` status for
`msg-20260812T214652Z-7b2c8a5e61d4` cannot be certified. Therefore:

- reserved queue consumption: `NOT_CONSUMED`;
- reserved queue dispatch: `NOT_DISPATCHED`;
- reserved queue strict unopened fact: `UNVERIFIED`;
- payload content used: `NONE`;
- execution authority: `NOT_GRANTED`.

This uncertainty independently prevents fact 3 from becoming `VERIFIED` even
if the missing shared parent were later corrected.

## 6. Durable disposition

- Fact 1: `BLOCKED`.
- Fact 2: `VERIFIED`.
- Fact 3: `BLOCKED`.
- Fact 4: `VERIFIED`.
- Aggregate PRE_RUN: `PRE_RUN_BLOCKED`.
- PRE_RUN_READY: `NO`.
- Reserved E1 queue: `NOT_CONSUMED`, `NOT_DISPATCHED`, strict unopened status
  `UNVERIFIED`.
- E1/B1/E2/C1: `NOT_EXECUTED`.
- Python/import/compile/test: `NOT_EXECUTED`.
- Fixture content: `NOT_READ`.
- Data/model/checkpoint/metric: `NOT_ACCESSED`.
- GPU/CUDA/Slurm/browser: `NOT_USED`.
- Execution receipt: `NOT_CREATED`.
- Subagents/probes/extra role processes: `NONE`.
- Scientific/protocol change: `NONE`.
- Remote mutation: `NONE`.

EVALUATOR_DECISION: P0_E1_READONLY_PRE_RUN_FACT_CHECK_BLOCKED.
