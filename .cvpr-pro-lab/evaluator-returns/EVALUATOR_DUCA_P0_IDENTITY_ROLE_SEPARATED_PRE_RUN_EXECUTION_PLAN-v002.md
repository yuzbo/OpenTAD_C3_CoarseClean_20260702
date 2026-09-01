---
doc_id: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN
version: v002
status: AUTHORED_NOT_EXECUTED
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T205555Z-7f66dacd9f70
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
builder_binding: BUILDER_DUCA_P0_CORRECTED_REMOTE_CLEAN_BINDING-v002
evaluator_binding: EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001
critic_correction: CRITIC_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_PLAN_STATIC_REVIEW-v001
supersedes: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v001
evidence_class: p0_identity_pre_run_execution_plan
execution_state: NOT_EXECUTED
---

# DUCA P0 role-separated PRE_RUN and execution plan

This document consumes only deterministic correction queue
`msg-20260812T205555Z-7f66dacd9f70` and supersedes v001. It applies exactly
`P0-PRE-PLAN-001..004` from the cited Critic static review. It is an authored
execution contract, not `PRE_RUN_READY`, not an execution queue, and not a P0
result. No local or remote command, import, compilation, test, materialization,
projector/reference/adapter runtime read or invocation, data/model/checkpoint/
metric access, SSH, GPU/CUDA/Slurm, or browser operation occurred while
authoring it.

The only scientific authority remains accepted
`PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`. This revision changes no projection
mathematics, fixture, objective, candidate order, typed failure, reference
independence, scientific route, evidence class, claim, split, metric, threshold,
budget, or stop rule.

The four corrections are limited to:

1. the bound production JSONL adapter is B1's sole adapter exception;
2. output-root state requirements are phase-relative;
3. every applicable positive `T<=385` fixture is exhaustively enumerated, while
   independently structured exact DAG/shortest-path is reserved for positive
   `T=767/768` fixtures; and
4. E1/B1 receipts preserve exact LF-stripped UTF-8 JSON line text plus the
   one-based source line, and E2 compares that text byte-for-byte without hashes.

## 1. Frozen bindings

The future gate is bound to these literal identities:

- Builder revision:
  `402e582c4092275877828fd36b5b97a8c1213011`.
- Builder direct parent:
  `df8228072b871adbd8dedb480e80f1f7daaca69e`.
- Scientific base:
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`.
- Builder clean checkout:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_402e582c_20260813`.
- Bound production JSONL adapter and sole B1 implementation entry point:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_402e582c_20260813/tools/bata/run_duca_p0_projection_production.py`.
- Sealed Evaluator root:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001`.
- Sealed fixture stream:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl`.
- Sealed fixture index:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json`.
- Sealed certificate-mutation definitions:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json`.
- Independent reference source, identified but not read in this task:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py`.

The future role output roots are literal and single-use:

- `REFERENCE_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference`
- `PRODUCTION_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production`
- `COMPARISON_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison`

Root-state checks are phase-relative and exact:

- before E1: all three roots are absent;
- before B1: sealed non-writable `REFERENCE_OUTPUT_ROOT` is present, while
  `PRODUCTION_OUTPUT_ROOT` and `COMPARISON_OUTPUT_ROOT` are absent;
- before E2: sealed non-writable `REFERENCE_OUTPUT_ROOT` and
  `PRODUCTION_OUTPUT_ROOT` are present, while `COMPARISON_OUTPUT_ROOT` is absent;
- before C1: all required E1, B1, and E2 outputs are present and sealed
  non-writable.

The owner of each absent phase output creates it exclusively, writes it once, and
makes it non-writable before return. A required-present root that is missing,
partial, writable, foreign, cross-project, or not bound to its accepted prior
phase receipt is a hard block. A required-absent root that exists is also a hard
block. There is no deletion, cleanup, overwrite, resume, append, repair, reuse,
fallback, alternate root, or rerun under this identity.

## 2. Frozen input contract

The fixture stream contains exactly 27 LF-terminated canonical UTF-8 JSON
objects. Every object has full integer arrays and fields in the exact order
`T,K,Q,u,a`; `Q=1048576`. The index binds one-based line number to fixture ID,
category, and frozen typed status. Production and reference must consume the
same original line bytes from this single read-only stream; neither may
regenerate, normalize, reserialize, clip, deduplicate, reorder, add, omit, or
repair input.

For each line, `exact_input_line_text` means the exact original UTF-8 JSON line
with only its single terminal LF removed. No other byte, whitespace, escape,
number, key order, or encoding transformation is allowed. The one-based
`source_line` and `exact_input_line_text` together are mandatory in E1 and B1
receipts. They replace neither the parsed canonical input object nor any semantic
certificate; they bind both implementations to the identical source bytes. No
hash or checksum is introduced.

Positive fixtures, exactly 18 in order:

`G16-U`, `G17-E2`, `G17-EINF`, `G17-E1`, `G17-U1`, `G17-PLEX`,
`G31-U`, `G32-U`, `G383-U`, `G384-U`, `G385-X`, `G767-U`,
`F768-U`, `F768-PERIODIC`, `F768-DISP16`, `F768-CONVEX`,
`F768-CONCAVE`, `F768-ALT`.

Negative fixtures, exactly 9 in order:

`N-T15`, `N-K`, `N-U-LEN`, `N-A-LEN`, `N-U-CANON`, `N-A-END`,
`N-A-ORDER`, `N-INFEASIBLE`, `N-ARITH`.

Certificate mutations, exactly 6 in order:

`M-DUPLICATE`, `M-STRIDE5`, `M-DISP17`, `M-OBJECTIVE`,
`M-SCALAR-TIE-LOSER`, `M-CANDIDATE-ORDER`.

The exact successful key remains
`(E2,E_infinity,E1,U1,p_1,...,p_(K-2))`; candidates are ascending physical
positions, exhaustive sequences are lexicographically ascending, and an
incumbent changes only for a strictly smaller complete exact key. A successful
`p` has length `K`, endpoints `0,T-1`, strides in `{1,2,3,4}`, and maximum
absolute uniform displacement 16. Integer arithmetic is exact; tolerance is
forbidden.

Required negative codes remain, in matrix order:

`INVALID_T_LT_16`, `K_EFF_MISMATCH`, `U_LENGTH_MISMATCH`,
`A_LENGTH_MISMATCH`, `U_CANONICAL_MISMATCH`, `A_ENDPOINT_MISMATCH`,
`A_ORDER_MISMATCH`, `INFEASIBLE`, `INTEGER_RANGE_OR_OVERFLOW`.

Mutation results remain `CERTIFICATE_REJECTED`, except the reversed candidate
sequence, which must be `CANDIDATE_ORDER_VIOLATION`.

## 3. Exact role order

Only the registered long-lived Evaluator, Builder, and Critic roles may perform
their phases. No subagent, probe, substitute role, shared executor, Coordinator
implementation, or third projector is permitted.

### Phase E1 — Evaluator independent-reference freeze

After a dedicated durable E1 execution queue and a passing role-specific
PRE_RUN, Evaluator may invoke only the sealed independent reference against the
sealed 27-line stream and six sealed mutation definitions. It must write only
`REFERENCE_OUTPUT_ROOT`.

E1 must complete and seal all reference expectations before any Builder
production command starts. Evaluator must not read or traverse the Builder
checkout, bound production JSONL adapter, `PRODUCTION_OUTPUT_ROOT`, any adapter,
validator, detector, model, decoder, dataset, checkpoint, metric, Torch/CUDA, or
training environment. The reference may share only the sealed input bytes and
frozen mathematical contract; it may import no production helper, objective,
certificate, candidate generator, selector, adapter, or implementation module.

The independent reference must perform exhaustive ascending enumeration for
every applicable positive fixture with `T<=385`, exactly:

`G16-U`, `G17-E2`, `G17-EINF`, `G17-E1`, `G17-U1`, `G17-PLEX`,
`G31-U`, `G32-U`, `G383-U`, `G384-U`, and `G385-X`.

Singleton feasible sets remain exhaustive results. The independently structured
exact integer DAG/shortest-path solver is reserved for the positive full-scale
fixtures with `T=767/768`, exactly:

`G767-U`, `F768-U`, `F768-PERIODIC`, `F768-DISP16`, `F768-CONVEX`,
`F768-CONCAVE`, and `F768-ALT`.

No positive fixture is assigned to a method outside those two frozen groups.
The receipt must include complete T17 witnesses, exhaustive results for all other
applicable `T<=385` fixtures, the complete 383-candidate `G385-X` witness and
ties, and independent T767/T768 root optima. Reference failure or incomplete
sealing blocks every later phase.

### Phase B1 — Builder one-shot production

Only after E1 is complete, accepted, and `REFERENCE_OUTPUT_ROOT` is present and
sealed non-writable may Coordinator dispatch one dedicated Builder production
queue. Builder may invoke only the bound production JSONL adapter
`tools/bata/run_duca_p0_projection_production.py`, exactly once, from the bound
clean checkout, using the sealed 27-line stream and six sealed mutation
definitions as read-only inputs. That bound adapter is the sole B1 adapter and
implementation-entry exception. It writes only `PRODUCTION_OUTPUT_ROOT`.

Builder must not read, list, traverse, import, or receive any reference source,
reference command, reference receipt, reference expectation, tie/root-optimum
result, or `REFERENCE_OUTPUT_ROOT`. Except for the bound production JSONL
adapter itself and the one production projector path it invokes as its frozen
purpose, B1 may invoke or access no other adapter, validator, detector path,
model path, decoder, implementation call, pytest, fallback, legacy selector,
dataset, checkpoint, metric, Torch/CUDA, Slurm, network, or training environment.
The bound adapter may not broaden into any of those forbidden surfaces. Any
production failure ends B1; the invocation is not repeated.

### Phase E2 — Evaluator comparison only

Only after complete, sealed E1 and B1 receipts may Coordinator issue one exact
Evaluator comparison queue. E2 reads the sealed fixture/index/mutation inputs and
the two sealed role receipts. It invokes neither production nor reference and
does not import either implementation. It writes only `COMPARISON_OUTPUT_ROOT`.

E2 compares in frozen fixture order, stopping at the first discrepancy. Before
semantic comparison for each fixture, it compares E1 and B1 `source_line` and
`exact_input_line_text`. `exact_input_line_text` is compared byte-for-byte as its
UTF-8 encoding; parsed-object equality is insufficient. Any line-number or byte
difference is the first discrepancy and binds the failure receipt to the source
line, fixture ID, E1 exact text, and B1 exact text. No hash is used.

Only after the byte comparison passes may E2 compare typed status, `p`, length,
endpoints, minimum/maximum stride, maximum uniform displacement, `E2`,
`E_infinity`, `E1`, `U1`, interior position vector, global-optimality evidence,
candidate order, exhaustive `T<=385` witness counts and ties, T767/T768 root
optima, and all six mutation codes.

### Phase C1 — Critic final closure

Critic acts only after the complete sealed Builder and Evaluator packages exist.
It performs an independent read-only audit of authority, role separation,
reference independence, closed input order, exact input-line byte binding,
comparison fields, first-failure behavior, forbidden-access attestations, and
scope deviation. It returns exactly `P0_IDENTITY_GATE_PASS` or
`P0_IDENTITY_GATE_BLOCKED`; it does not patch, rerun, recompute, invoke an
implementation, select science, or clean another role's output.

Only after a complete Critic `PASS` may Coordinator intake a candidate global
P0 pass. Coordinator does not itself execute, compare, review, or reinterpret
the result.

## 4. Allowed command envelopes

This plan does not authorize or seal any executable argv. Each phase requires a
new exact durable queue containing one literal argv, working directory,
interpreter, environment, input paths, output root, start condition, and expected
receipt path. Unresolved tokens or more than one implementation invocation are
hard failures.

The only future envelopes eligible for admission are:

| Phase | Working surface | Allowed call | Inputs | Output | Invocation limit |
| --- | --- | --- | --- | --- | --- |
| E1 | Sealed Evaluator source/root only | Independent reference only | sealed fixtures, index, mutations | `REFERENCE_OUTPUT_ROOT` | one reference batch |
| B1 | Bound clean Builder checkout only | Bound production JSONL adapter as sole adapter/implementation entry | same sealed fixtures, index, mutations | `PRODUCTION_OUTPUT_ROOT` | exactly one production batch |
| E2 | Sealed receipts only | comparison-only reader/writer | sealed E1/B1 receipts plus index/mutations | `COMPARISON_OUTPUT_ROOT` | one comparison pass; zero implementation calls |
| C1 | Frozen evidence package | read-only Critic review | complete E1/B1/E2 package | versioned Critic return | no implementation call |

For every remote CPU envelope, `CUDA_VISIBLE_DEVICES` must be empty and user-site
loading must be disabled. No envelope may load CUDA modules, activate a training
stack, install dependencies, use a launcher, connect to a network service, or
read a dataset/model/checkpoint/metric location. The sole B1 adapter exception
does not except any of those prohibitions. SSH used by the owning role only as
command transport is not permission for program network access.

## 5. Receipt schema expectations

Every E1, B1, and E2 receipt must be machine-readable canonical JSON and contain:

- `schema_version`, `artifact_status`, `project_id`, `parent_decision`, exact
  durable queue ID, role, phase, and `scope_deviation="none"`;
- literal input and output paths, working directory, interpreter, argv,
  environment, host, start/completion timestamps, and exit status;
- bound implementation identity: Builder revision and bound production JSONL
  adapter path for B1; sealed Evaluator reference path plus non-importing
  independence attestation for E1; both sealed receipt paths and
  `implementation_invocations=0` for E2;
- matrix declaration `Q=1048576`, positive/negative/mutation counts `18/9/6`,
  and the exact ordered IDs from section 2;
- one ordered record for each of 27 fixtures containing its one-based
  `source_line`, fixture ID, exact LF-stripped `exact_input_line_text`, parsed
  canonical input object `T,K,Q,u,a`, typed status, and no tolerance,
  normalization, or reserialization;
- for E1 and B1, an attestation that `exact_input_line_text.encode("utf-8")` is
  exactly the original source line bytes before the single terminal LF, with no
  hash or checksum;
- for successful fixtures: exact `p`, feasibility fields, `E2`, `E_infinity`,
  `E1`, `U1`, interior position vector, candidate-order evidence, and
  global-optimality evidence;
- for negative fixtures: the exact typed failure and null success-only fields;
- six ordered mutation records with the exact rejection code;
- exhaustive witness method/results for every applicable positive `T<=385`
  fixture and independent exact-DAG root-optimum evidence for positive
  `T=767/768` fixtures where owned by E1;
- first failure/discrepancy or null, zero-forbidden-access attestations, and
  evidence class `P0_PROJECTOR_CONFORMANCE_ONLY`.

The E2 global receipt must compare `source_line` for equality and
`exact_input_line_text` byte-for-byte before any semantic field. It records
`input_line_bytes_identical` for every compared fixture without a hash. The
first-discrepancy structure must contain phase, fixture ID, source line, field,
E1 exact text/value, and B1 exact text/value. For an input-byte discrepancy the
field is exactly `exact_input_line_text`, and both exact texts are included.

After byte identity passes, the E2 receipt contains a per-field equality decision
for every semantic field, reference-independence status, forbidden-access
status, and candidate gate state. It may state
`P0_IDENTITY_GATE_PASS_CANDIDATE` only when all 27 input-line byte comparisons,
all semantic fixture records, all exhaustive/full-scale witness obligations, and
all six mutations agree exactly. It cannot state a final pass before Critic
closure.

The eleven-artifact Pro bundle remains exactly: normative specification, fixture
matrix, independent reference, Builder production receipt, T17 receipt, T385
receipt, T768 receipt, negative/mutation receipt, Evaluator global receipt,
Critic closure, and unresolved-blockers receipt. This PRE_RUN plan and prior
materialization/binding receipts are infrastructure dependencies, not additions
to that scientific gate bundle.

## 6. Stop-on-first-discrepancy

Before a phase starts, any authority, identity, cleanliness, phase-relative root
state, path, permission, membership/order/count, process-isolation, argv,
environment, duplicate-run, or forbidden-access mismatch yields
`PRE_RUN_BLOCKED`; no command issues.

During E1 or B1, the first source-line byte, input, typed-status, certificate,
candidate-order, arithmetic, write, or forbidden-access failure stops that phase
and blocks all later phases. During E2, the first source-line/byte or semantic
discrepancy yields `P0_IDENTITY_GATE_BLOCKED` and a minimal first-discrepancy
receipt containing the exact input-line evidence required above. No later fixture
is used to dilute or reinterpret it.

After any started-phase failure there is no repair, rerun, third implementation,
tolerance, fallback, fixture deletion/addition, expectation update, output-root
reuse, or evidence promotion under the same identity. The failure establishes
only non-conformance or incomplete evidence for this package; it does not decide
density quality, learnability, detector correctness, accuracy, efficiency,
latency, cost, P1, PRE_RUN for a scientific experiment, or a paper claim.

## 7. Remaining admission conditions

All conditions below remain unsatisfied until Coordinator cites literal durable
evidence. No P0 command may issue before the applicable conditions are true:

1. Coordinator accepts this exact v002 plan without changing science or role
   order.
2. An independent focused Critic static recheck closes
   `P0-PRE-PLAN-001..004`, accepts the corrected Builder binding and this PRE_RUN
   separation, and reports no unresolved scientific ambiguity; this is not the
   later final C1 closure.
3. Coordinator records a formal `PRE_RUN_READY` binding the accepted Pro
   decision, this plan v002, Builder binding v002, sealed Evaluator
   materialization, registered role identities, and standing operational
   authorization.
4. A fresh read-only binding check immediately before E1 confirms the exact
   Builder revision/parent/base, zero Builder porcelain entries, the literal
   checkout, the sealed read-only Evaluator root, exact `18/9/6` membership/order,
   and no cross-project path.
5. Phase-relative root states are enforced exactly:
   - before E1, all three roots absent;
   - before B1, sealed reference present and production/comparison absent;
   - before E2, sealed reference/production present and comparison absent;
   - before C1, all required E1/B1/E2 outputs present and sealed.
   Any mismatch blocks the phase without deletion, reuse, overwrite, append,
   repair, alternate root, or retry.
6. E1 receives its own exact durable execution queue and sealed single argv.
   B1 and E2 queues do not become eligible until their preceding sealed receipts
   exist. C1 is dispatched only after E2 completes.
7. Each role-specific queue repeats the literal interpreter, cwd, input paths,
   phase-relative root requirement, output root, environment denials, invocation
   limit, receipt schema, exact-input-line byte obligation, first-fail rule, and
   zero-forbidden-access contract. No implicit default or unresolved placeholder
   remains.
8. Remote CPU service, credentials, and the approved run boundary are available
   without changing host, interpreter, checkout, source, fixtures, or outputs.
9. No dataset, model, checkpoint, metric, validator, detector, Torch/CUDA, GPU,
   Slurm, browser, network-program, training, test, launcher, decoder, or adapter
   access is required by any admitted argv, except that B1 must use the one bound
   production JSONL adapter as its sole adapter/implementation entry. That sole
   exception does not authorize any other forbidden surface. If any other access
   is required, admission fails rather than widening scope.
10. The Coordinator commits to event-driven dispatch of one role at a time and
    no retry under this execution identity.

## 8. Current receipt

- Plan: `AUTHORED_NOT_EXECUTED`
- Deterministic corrections: `P0-PRE-PLAN-001..004_APPLIED`
- PRE_RUN: `BLOCKED_PENDING_FOCUSED_CRITIC_RECHECK_AND_ADMISSION_CONDITIONS`
- P0: `BLOCKED_PRE_RESULT`
- P1: `BLOCKED`
- Production/reference/comparison/Critic closure: `NOT_EXECUTED`
- Projector/reference/adapter runtime surfaces: `NOT_READ_OR_INVOKED`
- Local/remote command, import, compile, test, materialization: `NOT_EXECUTED`
- Data/model/checkpoint/metric: `NOT_ACCESSED`
- SSH/GPU/CUDA/Slurm/browser: `NOT_USED`
- Subagents/probes/extra processes: `NONE`
- Scope deviation: `none`

`EVALUATOR_DECISION: ROLE_SEPARATED_PRE_RUN_PLAN_V002_AUTHORED_NOT_EXECUTED`.
