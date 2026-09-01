---
doc_id: EVALUATOR_DUCA_P0_PROJECTION_PROTOCOL
version: v001
status: AUTHORED_NOT_RUN
date: 2026-08-12
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T003436Z-eb863ea3b862
parent_decision: PRO_P0_PROJECTION_POLICY-v001
fixed_commit: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: preparatory_projection_protocol
execution: NOT_AUTHORIZED
pre_run: BLOCKED
---

# Evaluator DUCA P0 projection protocol v001

## Authority and boundary

This artifact consumes only durable queue
`msg-20260812T003436Z-eb863ea3b862`. Policy authority is the accepted intake
`PRO_P0_PROJECTION_POLICY-v001` and its verbatim transcript at
`.cvpr-pro-lab/pro-reviews/runs/duca-projection-policy-v001/raw-response.md`.

No implementation, fixture, test, validator, data, checkpoint, model forward,
local or remote command, CPU/GPU/Slurm workload, metric, browser, Git operation,
or launcher was accessed or executed. No route, claim, or gate was changed. No
subagent, probe, or extra process was used.

This is an authored-not-run projector-identity contract. It cannot produce a
P0 pass or authorize execution.

## Frozen projector

The identity obligation begins from the identical serialized tuple
`(T,K,u,a)`; it does not assert cross-library bit identity for softplus or
floating inverse-CDF computation.

- `K=min(384,16*floor(T/16))`; `T<16` fails closed.
- `Q=2^20=1048576`.
- `u_j=floor((2*j*(T-1)+(K-1))/(2*(K-1)))` for `j=0..K-1`.
- The projector input `a_j` is the exact nonnegative half-up conversion of the
  serialized binary64 inverse-CDF target, with endpoints forced to
  `a_0=0` and `a_{K-1}=Q*(T-1)`.
- Feasible `p` has length `K`, endpoints `0,T-1`, strides in `{1,2,3,4}`, and
  `abs(p_j-u_j)<=16`.
- For internal `j=1..K-2`, `e_j=Q*p_j-a_j`.
- The unique output minimizes the exact lexicographic key
  `(E2,E_infinity,E1,U1,p_1,...,p_{K-2})`, where
  `E2=sum(e_j^2)`, `E_infinity=max(abs(e_j))`,
  `E1=sum(abs(e_j))`, and `U1=sum(abs(p_j-u_j))`.
- Candidate positions and states are visited in the ascending order frozen by
  Pro. Exact arbitrary-precision or checked signed arithmetic of at least 128
  bits is required. No weighted sum, heuristic, tolerance, fallback, clipping,
  deduplication, second decoder, or legacy selector is conforming.
- Malformed input, uniform-witness failure, infeasibility, overflow, comparison
  inconsistency, missing minimizer, constraint failure, certificate failure, or
  production/reference mismatch fails closed with a typed projection error.

## Canonical fixture serialization

Each future fixture is one RFC 8259 JSON object containing these fields:

```json
{
  "fixture_schema": "duca-p0-projection-fixture-v001",
  "fixture_id": "<stable-id>",
  "family": "<declared-family>",
  "status": "AUTHORED_NOT_RUN",
  "T": 0,
  "K": 0,
  "Q": 1048576,
  "u": [],
  "a": [],
  "recipe": "<exact recipe named below>",
  "expected": {
    "p": null,
    "E2_decimal": null,
    "E_infinity_decimal": null,
    "E1_decimal": null,
    "U1_decimal": null
  }
}
```

`T`, `K`, `Q`, `u`, and `a` are JSON integers; no floating number is permitted
in the projector tuple. `u` and `a` each have exactly `K` elements. The tuple
passed to production and reference is the same parsed object, not separately
regenerated arrays. Large objective values are serialized as base-10 strings
in certificates to avoid loss of integer precision. No fixture hash or SHA tree
is required.

For compact recipes below, define nonnegative exact half-up division

`H(n,d)=floor((2*n+d)/(2*d))`, for integer `n>=0,d>0`.

Piecewise interpolation between integer fixed-point anchors `(j0,A0)` and
`(j1,A1)` is

`PL(j)=A0+H((A1-A0)*(j-j0),j1-j0)`.

The future fixture artifact must materialize every `u` and `a` array explicitly;
recipes are normative authoring checks, not runtime generators.

## Exhaustive witness: T=385, K=384

For `T=385,K=384`, every feasible sequence has exactly one stride of 2 and 382
strides of 1. Define `p^(s)` for `s=0..382` by

`p^(s)_j=j` when `j<=s`, otherwise `p^(s)_j=j+1`.

All 383 such sequences are feasible, and there are no others. The canonical
uniform vector is `u=p^(191)`. The independent exhaustive oracle must enumerate
exactly `s=0..382` in ascending order, compute every full key exactly, and
report one unique minimizer.

Two mandatory exhaustive fixtures are frozen:

### `t385_exact_s37`

- `u=p^(191)`.
- `a_j=Q*p^(37)_j` for every `j`.
- Expected unique output: `p=p^(37)`.
- Expected certificate:
  `E2=0`, `E_infinity=0`, `E1=0`, `U1=154`.
- Expected feasible count: `383`; expected minimizer count: `1`.

### `t385_full_key_tie`

- Start with `a_j=Q*u_j`.
- Replace `a_191=Q*191+3*Q/4` and `a_192=Q*192+Q/4`.
- Candidates `p^(190)` and `p^(192)` tie exactly through
  `(E2,E_infinity,E1,U1)`.
- Expected tied prefix:
  `E2=687194767360`, `E_infinity=786432`, `E1=1048576`, `U1=1`.
- Position-vector lexicography selects the expected unique output `p=p^(192)`.
- Expected feasible count: `383`; expected minimizer count: `1`.

The exhaustive receipt must list the first key at which every losing candidate
differs from the winner. Any count other than 383, any smaller key, more than one
full-key minimizer, or production/reference disagreement is `P0_BLOCKED`.

## Full-scale matrix: T=768, K=384

For every fixture below, set `T=768`, `K=384`, `Q=1048576`, and materialize the
same canonical `u`. Unless a recipe states otherwise, endpoints are
`a_0=0,a_383=Q*767`.

### `t768_smooth_monotone`

For internal `j`, set
`a_j=H(Q*767*j*j,383*383)`. This is the smooth monotone nonconstant target.

### `t768_alternating`

Let `b_j=H(Q*767*j,383)`. For internal `j`, set
`a_j=b_j+Q/4` when `j` is even and `a_j=b_j-Q/4` when `j` is odd.

### `t768_single_boundary`

Use exact `PL` interpolation through fixed-point anchors
`(0,0)`, `(128,320*Q)`, `(255,448*Q)`, `(383,767*Q)`.

### `t768_dual_boundary`

Use exact `PL` interpolation through fixed-point anchors
`(0,0)`, `(80,200*Q)`, `(150,270*Q)`, `(233,497*Q)`,
`(303,567*Q)`, `(383,767*Q)`.

### `t768_constraint_saturating`

Define `d_j=0` for `j>=192`. For `0<=j<192`, let `r=j mod 24` and
`d_j=2*r` for `0<=r<=8`, otherwise `d_j=24-r`. Set
`p_sat_j=u_j+d_j` and `a_j=Q*p_sat_j`.

The expected unique output is `p_sat`. Its certificate is
`E2=0`, `E_infinity=0`, `E1=0`, `U1=1536`; it reaches displacement 16 and
stride 4 without violating any bound.

### `t768_fixed_point_half_up`

Set `a_j=Q*u_j` except `a_100=Q*u_100+1`. The authoring source target at index
100 is exactly `x_100=u_100+2^-21`, so exact half-up conversion at `Q=2^20`
produces the required `+1`; host `round` is not admissible.

The expected unique output is `u`, with
`E2=1`, `E_infinity=1`, `E1=1`, `U1=0`.

For the four matrix fixtures without a closed-form expected minimizer, the
future independent reference establishes the expected `p` and exact certificate
only after a separate Pro execution authorization. Authoring does not populate
or infer those results.

## Exact identity and certificate obligations

For each fixture, future production and independent-reference receipts must
separately contain:

- exact returned `p`;
- `E2`, `E_infinity`, `E1`, and `U1` as decimal integer strings;
- internal position vector used by the final lexicographic term;
- length and endpoint checks;
- every stride, minimum/maximum stride, and maximum displacement from `u`;
- canonical-uniform feasibility witness status;
- exact objective-recomputation status;
- typed failure code when no certificate exists; and
- deviation field.

The combined identity receipt must state exact equality of `p`, all four
objective terms, the internal position vector, and the independently recomputed
feasibility certificate. The `T=385` receipts additionally record feasible
sequence count, minimizer count, selected long-stride index, and exhaustive
winner key.

The reference projector may share only the normative specification and fixture
objects. It must not import or call the production solver, comparator, candidate
generator, or backpointer logic. Critic supplies the separate independence
audit; this schema does not certify independence by assertion.

## Authored-not-run receipt state

The companion
`DUCA_P0_PROJECTION_IDENTITY_RECEIPT_SCHEMA-v001.json` defines the exact future
fields. In this version every execution field is literal `NOT_AUTHORIZED`, all
certificate statuses are `NOT_RUN`, and no output, exit status, or comparison
result exists. A later accepted Pro decision must issue a new execution receipt
version before any command may run.

Current status:

- `projection_protocol=AUTHORED_NOT_RUN`
- `projection_identity=PENDING_NOT_RUN`
- `P0=BLOCKED_PRE_RESULT`
- `P1=BLOCKED_NOT_AUTHORIZED`
- `PRE_RUN=BLOCKED`
- `scientific_evidence=BLOCKED_PRE_RESULT`

Even a future complete identity pass cannot establish density usefulness,
gradient validity, trainability, detector/transport closure, accuracy, cost,
efficiency, superiority, P1/P2 readiness, or any paper claim.

`EVALUATOR_DECISION: PROJECTION_PROTOCOL_AUTHORED_NOT_RUN`.

