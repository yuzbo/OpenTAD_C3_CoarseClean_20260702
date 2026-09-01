---
doc_id: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE
version: v004
status: CANDIDATE_NOT_EXECUTED_NOT_PRE_RUN_READY
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T214652Z-3e9d2c4a7f80
reserved_execution_queue_id: msg-20260812T214652Z-7b2c8a5e61d4
reserved_execution_queue_state: reserved_not_dispatched
reserved_execution_queue_consumption_state: NOT_CONSUMED
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v002
parent_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v003
parent_manifest_critic: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V003_FOCUSED_RECHECK-v001
coordinator_admission_candidate: DUCA_P0_E1_PRE_RUN_ADMISSION_CANDIDATE-v001
supersedes: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v003
phase: E1_INDEPENDENT_REFERENCE_FREEZE
evidence_class: p0_identity_e1_admission_candidate
execution_state: NOT_EXECUTED
pre_run_state: NOT_READY
admission_state: PRE_RUN_BLOCKED
---

# E1 independent-reference admission-manifest candidate

This v004 candidate consumes only durable authoring queue
`msg-20260812T214652Z-3e9d2c4a7f80`. It binds the reserved E1 identity
`msg-20260812T214652Z-7b2c8a5e61d4` as an inert literal. The reserved queue is
`reserved_not_dispatched`, was not opened or consumed by this authoring task,
and is not executable from this candidate.

This document does not authorize E1, establish `PRE_RUN_READY`, dispatch a
queue, issue an argv, or create an execution receipt. No local or remote
command, runtime surface, fixture/reference input, data, model, checkpoint,
metric, GPU/CUDA/Slurm service, or browser was accessed.

## 1. Frozen v003 envelope and sole identity binding

The following artifact is incorporated as the complete normative E1 envelope:

`EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v003`.

Its mathematics, `Q=1048576`, exact ordered `18/9/6` identities, fixtures,
absolute paths, eleven-element argv, inline program, exact LF-stripped input
discipline, exhaustive/reference method allocation, candidate order, output
schema, first-fail and one-shot behavior, phase-relative publication rules,
Evaluator/Builder role separation, and every environment/access denial are
preserved exactly.

The effective v004 argv[3] is defined deterministically as the exact UTF-8 bytes
of the v003 inline program with one and only one program-text substitution: its
unique queue-identity assignment is now exactly:

~~~python
QUEUE_ID = "msg-20260812T214652Z-7b2c8a5e61d4"
~~~

No other byte of the incorporated inline program changes. In particular,
`MANIFEST_VERSION = "v003"` and
`AUTHORING_QUEUE_ID = "msg-20260812T213026Z-2c2fa1e59073"` remain part of the
frozen, Critic-passed executable envelope. The v004 document version and its
authoring-queue lineage are recorded separately in this document's frontmatter
and durable receipt.

The former future-queue sentinel is therefore absent from the effective
argv[3]. That replacement resolves identity naming only; it grants no execution
authority and does not alter protocol or science.

## 2. Exact eleven-element argv

The preserved candidate argv still has exactly eleven elements. argv[3] is the
effective incorporated inline program defined in section 1. There is no shell
interpolation.

~~~text
argv[0]  = /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
argv[1]  = -B
argv[2]  = -c
argv[3]  = E1_INLINE_PROGRAM-v003 with the sole literal queue-identity substitution defined in section 1
argv[4]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py
argv[5]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl
argv[6]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json
argv[7]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json
argv[8]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference
argv[9]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production
argv[10] = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison
~~~

The exact environment bindings and denials remain the v003 values. The E1
process, if separately admitted later, remains limited to the registered N16R4
remote CPU environment, the literal interpreter and working directory, four
Evaluator-owned source/input paths, and the phase-relative reference output
contract. No Builder checkout, production adapter/output, adapter, validator,
detector, dataset, model, decoder, checkpoint, metric, Torch/CUDA, training
stack, Slurm, browser, network program, hash, tolerance, second solver,
fallback, repair, retry, fixture change, or route expansion is admitted.

## 3. Literal PRE_RUN admission checklist

Every fact below is deliberately `UNVERIFIED`. Naming the reserved E1 identity
does not satisfy any fact.

| admission fact | literal required condition | state |
|---|---|---|
| phase-specific clean Evaluator binding at frozen revision | `author_role=evaluator`; `phase=E1_INDEPENDENT_REFERENCE_FREEZE`; `project_id=g-p-6a796fef9a00819194024cf1de3bd697`; clean project-local Evaluator binding exactly at revision `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, with no foreign or Builder mutation | `UNVERIFIED` |
| exact sealed 18/9/6 input/root state | the four literal v003 Evaluator source/input paths are present, regular, sealed, non-writable and project-local; `Q=1048576`; ordered positive/negative/mutation memberships and source ordering equal the frozen v003 `18/9/6` contract; the exact shared parent `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001` pre-exists and is separately admitted | `UNVERIFIED` |
| absent E1/B1/E2 output roots and no prior/active receipt | `evaluator-reference`, `evaluator-reference.partial`, `builder-production`, and `evaluator-comparison` are absent; `DUCA_P0_E1_REFERENCE_RECEIPT-v001.json` is absent; no prior or active E1/B1/E2 command, job, result, receipt, retry, or duplicate invocation exists for this identity | `UNVERIFIED` |
| registered remote CPU-service availability | the registered N16R4 remote CPU service, literal interpreter `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`, and literal v003 working directory are available within the frozen project boundary for the separately admitted one-shot CPU batch | `UNVERIFIED` |

Aggregate admission state: `PRE_RUN_BLOCKED`.

Before any argv can issue, a later separately authorized read-only PRE_RUN must
independently verify all four facts. A versioned Coordinator admission record
must then change each literal fact from `UNVERIFIED` to `VERIFIED` and change
the aggregate state. Any mismatch remains `PRE_RUN_BLOCKED` and issues no
command. This candidate cannot perform that transition itself.

## 4. Reserved queue disposition and execution prohibition

- Reserved identity: `msg-20260812T214652Z-7b2c8a5e61d4`.
- Registration state: `reserved_not_dispatched`.
- Payload access: `NONE`.
- Consumption state: `NOT_CONSUMED`.
- Dispatch state: `NOT_DISPATCHED`.
- Executability from v004: `FORBIDDEN`.
- PRE_RUN authority: `NOT_GRANTED`.
- E1 authority: `NOT_GRANTED`.

The reserved identity must not be issued, consumed, or dispatched on the basis
of this v004 candidate. No production/reference invocation, comparison, test,
import, compile, fixture materialization or runtime read, data/model/checkpoint/
metric access, GPU/CUDA/Slurm/browser operation, receipt creation, protocol
change, or scientific change is permitted before the independent admission
transition described above.

## 5. Durable authored-not-run receipt

- Authoring queue consumed: `msg-20260812T214652Z-3e9d2c4a7f80`.
- Reserved execution identity named only: `msg-20260812T214652Z-7b2c8a5e61d4`.
- Reserved execution queue consumed: `NO`.
- Pro decision: `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`.
- v003 Critic closure: `CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V003_FOCUSED_RECHECK-v001` (`PASS`).
- Artifact: `EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v004`.
- Candidate: `AUTHORED_NOT_EXECUTED`.
- PRE_RUN: `NOT_READY` / `PRE_RUN_BLOCKED`.
- Admission facts: `4 UNVERIFIED`, `0 VERIFIED`.
- E1/B1/E2/C1: `NOT_EXECUTED`.
- P0: `BLOCKED_PRE_RESULT`.
- P1: `BLOCKED`.
- Reference/projector/adapter runtime access: `NONE`.
- Fixture materialization or runtime fixture read: `NONE`.
- Local/remote command or SSH: `NOT_EXECUTED`.
- Data/model/checkpoint/metric: `NOT_ACCESSED`.
- GPU/CUDA/Slurm/browser: `NOT_USED`.
- Execution receipt: `NOT_CREATED`.
- Subagents/probes/extra processes: `NONE`.
- Scientific/protocol change: `NONE`.
- Scope deviation: `none`.

EVALUATOR_DECISION: E1_ADMISSION_MANIFEST_V004_RESERVED_IDENTITY_BOUND_AUTHORED_NOT_EXECUTED_PRE_RUN_BLOCKED.
