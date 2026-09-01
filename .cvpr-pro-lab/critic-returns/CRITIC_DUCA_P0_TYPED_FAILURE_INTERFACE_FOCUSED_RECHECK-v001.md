---
doc_id: CRITIC_DUCA_P0_TYPED_FAILURE_INTERFACE_FOCUSED_RECHECK
version: v001
stage: DRAFT_P0_IDENTITY_INTERFACE
author_role: critic
parent_message_id: msg-20260812T193541Z-26663d626ebd
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_builder_return: BUILDER_DUCA_P0_TYPED_FAILURE_INTERFACE_CORRECTION-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
builder_snapshot: C:\Users\skywalker\.codex\worktrees\07c1\OpenTAD_C3_CoarseClean_20260702
verdict: CLOSED_FOR_NEXT_P0_DEPENDENCY
remaining_findings: NONE
scientific_ambiguity: NONE
status: NOT_EXECUTED
evidence_class: BLOCKED_PRE_RESULT
---

# Critic focused recheck — P0 typed-failure interface correction

## Frozen target and review boundary

- Consumed durable queue `msg-20260812T193541Z-26663d626ebd`.
- Authority: accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`.
- Reviewed only `BUILDER_DUCA_P0_TYPED_FAILURE_INTERFACE_CORRECTION-v001`
  and the named Builder changes in:
  - `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`;
  - `tests/test_duca_p0_projection_policy.py`.
- Frozen Critic comparison revision:
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`.
- Review was static/read-only. `NOT_EXECUTED`.

## Attacks and evidence

### 1. Stable typed interface

`DUCAProjectionError` retains its fail-closed exception identity and adds only an
optional machine-readable `code` field; prose remains the exception message
(`pc_ot_mras_prebackbone_frame_selector.py:31-36`). No message-substring status
inference, fallback, retry, or alternate success path was added.

### 2. Closed nine-case negative matrix and precedence

The authored validation order gives the frozen cases their required exact codes:

- admitted-integer validation for `T`, followed by `T < 16`, precedes K or array
  access, so `N-T15` maps to `INVALID_T_LT_16`
  (`pc_ot_mras_prebackbone_frame_selector.py:239-252`);
- admitted-integer validation for `K`, followed by the frozen effective-K rule,
  maps `N-K` to `K_EFF_MISMATCH`
  (`pc_ot_mras_prebackbone_frame_selector.py:253-267`);
- `u` and `a` are materialized separately and their lengths are checked against
  K with `U_LENGTH_MISMATCH` and `A_LENGTH_MISMATCH`; non-integer, negative, or
  greater-than-signed-128 elements fail as `INTEGER_RANGE_OR_OVERFLOW`
  (`pc_ot_mras_prebackbone_frame_selector.py:83-117,268-279`);
- exact canonical-uniform comparison maps `N-U-CANON` to
  `U_CANONICAL_MISMATCH`, after which the canonical witness feasibility check
  maps the frozen `T=1534,K=384` case to `INFEASIBLE`
  (`pc_ot_mras_prebackbone_frame_selector.py:280-294`);
- endpoint validation precedes monotonic-order validation, mapping `N-A-END` to
  `A_ENDPOINT_MISMATCH` and `N-A-ORDER` to `A_ORDER_MISMATCH`
  (`pc_ot_mras_prebackbone_frame_selector.py:295-305`);
- the frozen `N-ARITH` mutation `a[1]=2^127` is rejected by the admitted-range
  check as `INTEGER_RANGE_OR_OVERFLOW` before projection mathematics begins
  (`pc_ot_mras_prebackbone_frame_selector.py:107-117`).

The authored regression enumerates exactly those nine cases, calls the sole
production projector, captures `DUCAProjectionError`, and compares its `code`
field directly (`tests/test_duca_p0_projection_policy.py:141-188`). It does not
derive status from prose.

### 3. Frozen projector and decoder surfaces

The typed correction is confined to exception metadata and pre-projection
validation. The inspected solver still uses the frozen candidate generator,
ascending predecessor/current/candidate traversal, exact integer E2,
E-infinity, E1, U1 and final position-vector ordering, followed by exact
certificate recomputation (`pc_ot_mras_prebackbone_frame_selector.py:307-475`).
No Q, canonical-uniform formula, feasible bound, candidate ordering, objective,
tie rule, certificate, second solver, clipping, tolerance, deduplication, or
fallback change is present in this correction.

C-PROJ-001 remains closed: decoder endpoints are explicit; inverse-CDF lookup is
limited to internal ranks; out-of-range lookup remains typed fail-closed; and no
decoder clamp or replacement repair is present
(`pc_ot_mras_prebackbone_frame_selector.py:478-557`).

### 4. Test and evidence boundary

The added test is code-propagation coverage only. It constructs the frozen
negative inputs and checks exception codes. It adds no reference optimizer,
fixture winner, certificate trace, identity comparison, metric, cost,
performance assertion, execution authorization, or paper evidence
(`tests/test_duca_p0_projection_policy.py:141-188`).

## Verdict

`CLOSED_FOR_NEXT_P0_DEPENDENCY`.

No exact deterministic defect remains in the bounded typed-failure interface, so
there is no remaining `IMPLEMENTATION_CORRECTION`. Resolution requires no change
to route, mechanism, claim, split, metric, threshold, budget, or protocol, so
`SCIENTIFIC_AMBIGUITY: NONE`.

Fairness/leakage verdict: unchanged and clear for this bounded interface change;
it adds no data, GT, teacher, cache, detector, evaluator, metric, or result
surface. No scientific route was created, selected, revised, or advanced.

This verdict closes only the authored typed-failure dependency. It does not start
or pass the P0 identity/optimality gate and establishes no identity, optimality,
metric, cost, efficacy, or paper claim. P0 remains `BLOCKED_PRE_RESULT`.

`NOT_EXECUTED`: no shell/repository command, Python/import, test, validator,
model or projector execution, data/model/checkpoint access, CPU/GPU workload,
Slurm, browser, network, Git mutation, experiment, Pro, Sources, identity gate,
result admission, or evidence promotion was performed. The only write is this
queue-required durable Critic return in the canonical control plane.
