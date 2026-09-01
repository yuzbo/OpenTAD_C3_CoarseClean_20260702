---
doc_id: EVALUATOR_DUCA_P0P1_PROTOCOL_AMENDMENT
version: v002
status: authored_not_executed
date: 2026-08-12
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260811T194655Z-955f276dfb30
parent_decision: PRO_P0_ROUTE_ADJUDICATION-v002
builder_parent: BUILDER_DUCA_P0_BOUNDED_PATCH_RETURN-v001
base_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: EVALUATOR_DUCA_P0P1_PROTOCOL_AMENDMENT-v001
evidence_class: preparatory_protocol_authored_not_executed
p1_status: BLOCKED
pre_run_status: BLOCKED
---

# Evaluator DUCA P0/P1 protocol amendment v002

## Authority and execution boundary

This artifact consumes only durable queue
`msg-20260811T194655Z-955f276dfb30`, citing
`PRO_P0_ROUTE_ADJUDICATION-v002` and
`BUILDER_DUCA_P0_BOUNDED_PATCH_RETURN-v001`.

No implementation file, fixture, data, checkpoint, model forward, test,
validator, local or remote command, CPU/GPU/Slurm workload, metric, browser,
held-out split, Git state, route, claim, or `PRE_RUN` state was accessed or
changed. The Builder return and governing records were read as documents only.
No subagent, probe, or extra process was used.

The Builder patch remains `authored_not_executed`. This amendment is a future
verification contract, not a P0 pass receipt and not authority to run P1.

## Frozen fixed-K density decoder contract

Future P1 must evaluate the exact accepted candidate
`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`:

1. The sole learned decoder input is the valid prefix of
   `duca_density_logits[b,t]`, produced by the dedicated density-only reader
   from dense `browser_memory` with `selection_unit=1` and an identity physical
   candidate grid.
2. Slot allocation, frame-selection logits, actionness/boundary heads, soft
   transport, top-k/rank, quota, and dynamic-budget policies are not density
   inputs and must not affect decoded positions when density logits are held
   fixed.
3. For prefix length `T_v`,
   `K_eff=min(384,16*floor(T_v/16))`; `T_v<16` fails closed.
4. Finite logits are transformed pointwise as `1e-6 + softplus(logit)`.
   Adjacent point weights form per-sample trapezoidal interval masses.
   Endpoint-inclusive inverse-CDF targets are `j/(K_eff-1)`.
5. The integer output has exactly `K_eff` positions, includes `0` and
   `T_v-1`, is in range, unique, and strictly increasing. Adjacent selected
   positions differ by at most 4. Each output differs by at most 16 positions
   from its canonical uniform counterpart.
6. Non-finite input, infeasible constraints, solver failure, or an unresolved
   tie fails closed. No clipping, deduplication, scaffold, or post-hoc repair may
   convert failure to success.
7. Exactly equal finite logits over the valid prefix use the canonical
   endpoint-inclusive integer-half-up uniform generator with exact equality,
   not a tolerance. Padding outside the valid prefix is irrelevant.
8. Near-constant logits follow the density decoder and must not enter the
   constant specialization.
9. No gradient surrogate, detector-gradient path, contribution teacher,
   dynamic K, or learned-budget behavior is admitted by this P0/P1 contract.

For the canonical specialization,
`u_j=floor((2*j*(T_v-1)+(K_eff-1))/(2*(K_eff-1)))`. At
`T_v=768,K_eff=384`, the vector starts at 0 and ends at 767.

## P0 authored-but-not-executed obligations

The complete Builder diff must receive independent static Critic closure before
any P1 dispatch. P0 is blocked unless that closure establishes all of the
following against the ten paths listed in the Builder return:

- both exact-uniform call sites use one canonical generator;
- constant valid-prefix logits take the exact canonical specialization;
- the density-only reader cannot alias any excluded legacy selector signal;
- count, endpoint, ordering, span, displacement, non-finite, and fail-closed
  semantics are represented by authored fixtures without weakening the frozen
  thresholds;
- raw proposals are tagged `selected_q` and mapped exactly once to
  `physical_dense` at the start of each per-sample
  `SingleStageDetector.post_processing`, before filtering, top-k, IoU, or NMS;
- unknown coordinate state and an attempted second mapping fail before NMS;
- detector/head, losses, NMS callable and configuration, evaluator, split, and
  class map are unchanged; and
- the new config remains explicitly non-launchable and
  `BLOCKED_PRE_RESULT`.

The only admissible static terminal statuses are `P0_STATIC_PASS` and
`P0_BLOCKED`. Neither status is created by this amendment.

## Future remote-CPU P1 receipt obligations

P1 requires a later accepted Pro decision and a new durable Evaluator queue
that names one immutable execution snapshot containing the accepted patch, the
exact focused command, environment, output path, and fixture seed. No command is
authorized here.

The future deterministic receipt must contain these witnesses:

### Decoder identity and geometry

- Valid-prefix cases `T_v={16,17,31,32,383,384,385,767,768}` with expected
  `K_eff={16,16,16,32,368,384,384,384,384}`.
- Invalid cases `T_v={0,1,15}` fail before decoding.
- Constant logits match the canonical vector bit-for-bit in shape, integer
  dtype, order, and values; the `768/384` case ends at 767.
- Near-constant, monotone, alternating, impulse, boundary-heavy, extreme finite,
  and one preregistered seeded-random family satisfy exact count, endpoints,
  range, uniqueness, strict ordering, adjacent span `<=4`, and canonical
  displacement `<=16`.
- Repeated identical inputs and batch permutation/duplication yield identical
  per-sample positions.
- With density logits fixed, perturbing every excluded legacy selector signal
  leaves decoded positions unchanged.
- Non-finite inputs, infeasible constraints, solver failure, and explicit tie
  cases return the declared fail-closed outcome without repaired positions.

### Five deterministic parity boundaries

Using identical synthetic sentinels in the clean-uniform and exact-constant
DUCA paths, record the first mismatch across:

1. selected integer indices;
2. gathered input after identical preprocessing;
3. compact valid mask and selected-axis training segments;
4. raw selected-q proposals, scores, and labels; and
5. physical-dense proposals presented to unchanged NMS and final serialization.

Indices, masks, inputs, scores, labels, and serialized outputs must be exact.
Coordinate comparisons may use only the preregistered `1e-6` dense-unit
tolerance. These are synthetic correctness witnesses, not detector-quality
evidence.

### Coordinate transport and NMS order

- The only accepted transition is `selected_q -> physical_dense`.
- Knots cover selected endpoints `[0,K_eff]` and physical endpoints `[0,T_v]`.
- Knot, midpoint, near-knot, and seeded segment round trips preserve ordering,
  scores, and labels with maximum error `1e-5` in the corresponding coordinate
  units.
- Missing/unknown coordinate state and `physical_dense` passed to the adapter
  fail before filtering or NMS; the negative-case NMS call count is zero.
- For every non-sliding single-class and multiclass branch, event order is
  `raw_selected_q -> map_once -> physical_dense -> filter/top-k -> NMS -> seconds`.
  If NMS is disabled, mapping still occurs exactly once before later coordinate
  conversion.
- NMS receives the mapped physical tensor. Its callable and all configured
  values are identical to the clean control.

### Detector/config invariance

The resolved clean and DUCA configs may differ only in the accepted
density/acquisition path, canonical sampling source, coordinate-state metadata,
and nonsemantic output directory. Any detector, head, assignment, loss, NMS,
evaluator, split, class-map, or augmentation difference blocks P1.

## Minimum durable P1 receipt

The future receipt records, without a hash manifest:

- project ID, new queue ID, fresh Pro decision ID, Builder and Critic receipt
  IDs, base revision, immutable execution revision, command, host/environment,
  output path, and seed;
- one `pass|blocked|not_run` status and first failure for each witness group
  above;
- observed `K_eff`, endpoints, maximum adjacent span, maximum displacement,
  maximum round-trip error, first parity boundary mismatch, coordinate event
  trace, and first unapproved config key;
- attestations that no dataset, checkpoint, GPU/CUDA, metric/evaluator,
  validation/test, held-out, network, or raw-prediction cache was accessed;
- deviation `none` or one explicit deviation that forces `P1_BLOCKED`; and
- terminal `P1_PASS` or `P1_BLOCKED`, always with
  `evidence_class=preparatory_synthetic_correctness` and
  `scientific_evidence_status=BLOCKED_PRE_RESULT`.

Any failed witness, forbidden access, command/snapshot deviation, missing field,
or non-zero exit stops P1 as `P1_BLOCKED`. Do not tune thresholds, alter
fixtures, repair positions, relabel evidence, or self-authorize a scientific
retry. An objective infrastructure retry must reuse the identical snapshot,
command, inputs, and seed and receive a new linked receipt.

## Evidence classification and explicit gates

- Builder return: `implementation_authored_not_executed`.
- This amendment: `preparatory_protocol_authored_not_executed`.
- Future P1 pass: `preparatory_synthetic_correctness` only.
- Model quality, benchmark accuracy, efficiency, cost, held-out performance,
  novelty, and paper claims: no evidence.

Current gate state:

- `P0=AUTHORED_NOT_EXECUTED_PENDING_CRITIC`.
- `P1=BLOCKED_NOT_AUTHORIZED`.
- `PRE_RUN=BLOCKED`.
- `scientific_evidence=BLOCKED_PRE_RESULT`.

A P1 pass would not automatically unlock `PRE_RUN`; a later accepted Pro
decision must adjudicate the returned P0/P1 evidence and select any experiment.

## Protocol ambiguity requiring Pro

`PROTOCOL_AMBIGUITY_REQUIRES_PRO`: for nonconstant logits, the accepted decision
requires a deterministic constrained integer projection but does not freeze its
optimization objective, canonical candidate ordering, or tie-breaking rule.
Several different integer vectors can satisfy the same count/span/displacement
constraints. P1 can verify properties and repeatability, but it cannot certify
cross-implementation decoder identity or decide which tied optimum is canonical.

Before P1, Pro must either freeze the projection objective and complete tie rule,
or explicitly declare nonconstant acceptance to be property-and-repeatability
only and define which situations count as an unresolved tie that must fail.
Evaluator does not choose between those scientific meanings.

## Conclusion

`EVALUATOR_DECISION: PROTOCOL_AMENDED_PRE_RUN_BLOCKED`.

The v002 protocol is authored and unexecuted. It supersedes the prior density
decoder description and any prior future-command template where inconsistent
with `PRO_P0_ROUTE_ADJUDICATION-v002`. No P0 pass, P1 authority, PRE_RUN
readiness, result, or claim is created.
