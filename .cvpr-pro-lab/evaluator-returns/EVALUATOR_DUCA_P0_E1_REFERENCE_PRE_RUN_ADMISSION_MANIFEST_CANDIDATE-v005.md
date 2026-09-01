---
doc_id: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE
version: v005
status: CANDIDATE_NOT_EXECUTED_NOT_PRE_RUN_READY
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T220919Z-9d4e6b8a21f0
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v004
parent_manifest_critic: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V004_FOCUSED_RECHECK-v001
parent_fact_check: EVALUATOR_DUCA_P0_E1_PHASE_SPECIFIC_READONLY_PRE_RUN_ADMISSION_FACT_CHECK-v001
supersedes: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v004
old_execution_identity: msg-20260812T214652Z-7b2c8a5e61d4
old_execution_identity_state: FAIL_CLOSED_INELIGIBLE_STRICT_UNOPENED_UNVERIFIABLE
fresh_reserved_execution_identity: msg-20260812T220919Z-12f7e8a4d9c3
fresh_reserved_execution_identity_state: reserved_not_dispatched_payload_unread
fresh_identity_payload_state: UNREAD
fresh_identity_open_state: UNOPENED
fresh_identity_consumption_state: NOT_CONSUMED
fresh_identity_dispatch_state: NOT_DISPATCHED
fresh_identity_execution_state: NOT_EXECUTED
phase: E1_INDEPENDENT_REFERENCE_FREEZE
fixed_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: p0_identity_e1_admission_candidate
execution_state: NOT_EXECUTED
pre_run_state: NOT_READY
admission_state: PRE_RUN_BLOCKED
---

# E1 independent-reference admission-manifest candidate

This v005 candidate consumes only durable authoring queue
`msg-20260812T220919Z-9d4e6b8a21f0`. It supersedes v004 by binding the fresh
opaque reserved identity `msg-20260812T220919Z-12f7e8a4d9c3` as the sole
effective `QUEUE_ID` literal.

The fresh identity remains
`reserved_not_dispatched_payload_unread`: it was not opened, read, parsed,
consumed, dispatched, or executed. Its identifier is known only because the
authoring queue supplied that literal. This v005 candidate does not authorize
E1, establish `PRE_RUN_READY`, issue an argv, or create an execution receipt.

The prior identity `msg-20260812T214652Z-7b2c8a5e61d4` is permanently
fail-closed and ineligible for E1 because its strict unopened status cannot be
certified. It is not reused, rehabilitated, consumed, dispatched, or executed by
v005.

## 1. Frozen envelope and sole effective identity substitution

The complete normative E1 envelope remains the Critic-passed v003 program and
the v004 incorporation contract. All frozen mathematics, `Q=1048576`, exact
ordered `18/9/6` identities, fixtures, absolute paths, eleven-element argv,
input order, exact LF-stripped input discipline, exhaustive/reference method
allocation, candidate order, output schema and publication discipline,
first-fail and one-shot behavior, Evaluator/Builder role order, evidence class,
environment/access denials, and scientific/protocol boundaries are preserved
exactly.

The effective v005 argv[3] is defined as the exact UTF-8 bytes of the frozen
v003 inline program with one and only one program-text substitution. Its unique
queue-identity assignment is now exactly:

~~~python
QUEUE_ID = "msg-20260812T220919Z-12f7e8a4d9c3"
~~~

No other byte of the incorporated inline program changes. In particular,
`MANIFEST_VERSION = "v003"` and
`AUTHORING_QUEUE_ID = "msg-20260812T213026Z-2c2fa1e59073"` remain part of the
frozen executable envelope. The v005 document version and current authoring
queue are recorded separately in this artifact's frontmatter and receipt.

The v004 effective queue identity is no longer eligible. The fresh substitution
resolves queue naming only; it grants no authority and changes no executable
logic, fixture, path, output, role, evidence, or science.

## 2. Preserved exact eleven-element argv

The candidate argv still has exactly eleven elements. argv[3] is the effective
frozen inline program with only the fresh opaque identity substitution defined
in section 1. There is no shell interpolation.

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

The exact v003/v004 environment bindings and denials remain unchanged. No
Builder checkout, production adapter/output, adapter, validator, detector,
dataset, model, decoder, checkpoint, metric, Torch/CUDA, training stack, Slurm,
browser, network program, hash, tolerance, second solver, fallback, repair,
retry, fixture change, or route expansion is admitted.

## 3. Prior read-only fact outcome incorporated without authority upgrade

`EVALUATOR_DUCA_P0_E1_PHASE_SPECIFIC_READONLY_PRE_RUN_ADMISSION_FACT_CHECK-v001`
returned aggregate `PRE_RUN_BLOCKED`. v005 carries its factual outcome forward
without rerunning, repairing, reinterpreting, or upgrading any authority.

| admission fact | carried state in v005 | required next condition |
|---|---|---|
| 1. Phase-specific clean Evaluator binding at frozen revision | `REPLACEMENT_READ_ONLY_CHECK_REQUIRED` | After Coordinator recovery, a new bounded read-only check must verify the exact registered Evaluator workspace, revision `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, zero porcelain entries, and no foreign/Builder mutation. |
| 2. Exact sealed 18/9/6 Evaluator input/root state | `VERIFIED_BY_V001_READ_ONLY_CHECK` | Preserve the verified project-local non-writable metadata and the sealed-materialization receipt's exact ordered `18/9/6` provenance; this verification alone grants no authority. |
| 3. Shared-parent/output/duplicate/reserved-identity state | `REPLACEMENT_READ_ONLY_CHECK_REQUIRED` | After Coordinator recovery, a new bounded read-only check must verify the admitted shared parent exists, all phase-relative E1/B1/E2 outputs and receipt remain absent, no prior or active phase run/receipt exists, the old identity remains fail-closed, and the fresh opaque identity remains unread, unopened, unconsumed, undispatched, and unexecuted. |
| 4. Registered N16R4 CPU transport, interpreter, and cwd availability | `VERIFIED_BY_V001_READ_ONLY_CHECK` | Preserve the prior non-invoking availability evidence; this verification alone grants no authority. |

Facts 2 and 4 remain infrastructure facts only. They do not compensate for,
waive, or satisfy facts 1 and 3. A later replacement read-only check must be
authorized by a new durable queue and must not infer readiness from this
authoring artifact.

Aggregate admission state remains `PRE_RUN_BLOCKED` with `PRE_RUN_READY=NO`.

## 4. Fresh and prior identity disposition

### Fresh identity

- Identity: `msg-20260812T220919Z-12f7e8a4d9c3`.
- State: `reserved_not_dispatched_payload_unread`.
- Payload: `UNREAD`.
- Open state: `UNOPENED`.
- Consumption: `NOT_CONSUMED`.
- Dispatch: `NOT_DISPATCHED`.
- Execution: `NOT_EXECUTED`.
- Executability from v005: `FORBIDDEN`.
- PRE_RUN authority: `NOT_GRANTED`.
- E1 authority: `NOT_GRANTED`.

### Prior identity

- Identity: `msg-20260812T214652Z-7b2c8a5e61d4`.
- State: `FAIL_CLOSED_INELIGIBLE_STRICT_UNOPENED_UNVERIFIABLE`.
- Eligibility for E1: `PERMANENTLY_INELIGIBLE`.
- Consumption by v005: `NO`.
- Dispatch by v005: `NO`.
- Execution by v005: `NO`.

Neither identity may issue on the basis of v005. Before any candidate argv can
issue, Coordinator recovery, a newly queued replacement read-only fact check,
and a later versioned Coordinator admission record must close every literal
admission fact. Any failed or uncertain fact remains `PRE_RUN_BLOCKED` and
issues no command.

## 5. Authored-not-run boundary

While authoring v005:

- local or remote command: `NOT_EXECUTED`;
- SSH/remote access: `NOT_USED`;
- Python/import/compile/test: `NOT_EXECUTED`;
- fixture/reference content or runtime surface: `NOT_READ_OR_ACCESSED`;
- production/reference/comparison invocation: `NOT_EXECUTED`;
- data/model/checkpoint/metric: `NOT_ACCESSED`;
- GPU/CUDA/Slurm/browser: `NOT_USED`;
- execution receipt: `NOT_CREATED`;
- reserved fresh-identity payload: `UNREAD_AND_UNOPENED`;
- old identity payload: `NOT_ACCESSED_BY_V005`;
- subagents/probes/extra role processes: `NONE`;
- protocol/scientific change: `NONE`;
- scope deviation: `none`.

## 6. Durable authored-not-run receipt

- Authoring queue consumed: `msg-20260812T220919Z-9d4e6b8a21f0`.
- Pro decision: `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`.
- Superseded manifest: `EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v004`.
- Prior fact check: `EVALUATOR_DUCA_P0_E1_PHASE_SPECIFIC_READONLY_PRE_RUN_ADMISSION_FACT_CHECK-v001` (`PRE_RUN_BLOCKED`).
- Artifact: `EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v005`.
- Candidate: `AUTHORED_NOT_EXECUTED`.
- PRE_RUN: `NOT_READY` / `PRE_RUN_BLOCKED`.
- Fact 1: `REPLACEMENT_READ_ONLY_CHECK_REQUIRED`.
- Fact 2: `VERIFIED_BY_V001_READ_ONLY_CHECK`.
- Fact 3: `REPLACEMENT_READ_ONLY_CHECK_REQUIRED`.
- Fact 4: `VERIFIED_BY_V001_READ_ONLY_CHECK`.
- Fresh identity: `OPAQUE_UNREAD_UNOPENED_NOT_CONSUMED_NOT_DISPATCHED_NOT_EXECUTED`.
- Prior identity: `FAIL_CLOSED_PERMANENTLY_INELIGIBLE`.
- E1/B1/E2/C1: `NOT_EXECUTED`.
- P0: `BLOCKED_PRE_RESULT`.
- P1: `BLOCKED`.
- Execution authority: `NOT_GRANTED`.

EVALUATOR_DECISION: E1_ADMISSION_MANIFEST_V005_FRESH_OPAQUE_IDENTITY_BOUND_AUTHORED_NOT_EXECUTED_PRE_RUN_BLOCKED.
