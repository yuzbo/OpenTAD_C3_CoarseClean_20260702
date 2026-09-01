---
doc_id: CRITIC_DUCA_P0_PROJECTION_POLICY_STATIC_REVIEW
version: v001
stage: DRAFT_P0_STATIC_REVIEW
author_role: critic
parent_message_id: msg-20260812T122929Z-f8afaa83eef5
parent_decision: PRO_P0_PROJECTION_POLICY-v001
parent_builder_return: BUILDER_DUCA_P0_PROJECTION_PATCH-v001
parent_evaluator_protocol: EVALUATOR_DUCA_P0_PROJECTION_PROTOCOL-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
builder_snapshot: C:\Users\skywalker\.codex\worktrees\07c1\OpenTAD_C3_CoarseClean_20260702
verdict: P0_BLOCKED
primary_classification: IMPLEMENTATION_CORRECTION
scientific_ambiguity: NONE
evidence_class: BLOCKED_PRE_RESULT
execution: NOT_RUN
---

# CRITIC_DUCA_P0_PROJECTION_POLICY_STATIC_REVIEW-v001

## Evidence receipt

- Durable queue consumed: `msg-20260812T122929Z-f8afaa83eef5`.
- Accepted Pro decision: `.cvpr-pro-lab/pro-reviews/PRO_P0_PROJECTION_POLICY-v001.md`
  (`status: accepted`, `scientific_decision: CONTINUE`).
- Review surface was restricted to the five Builder paths named by the queue.
- Frozen comparison revision: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`.
- Classification: `IMPLEMENTATION_CORRECTION`; `SCIENTIFIC_AMBIGUITY: NONE`.

## Finding C-PROJ-001 — nonconstant decoder performs forbidden clipping

Classification: `IMPLEMENTATION_CORRECTION`.

The accepted Pro policy states that no clipping is allowed for a nonconstant
input (`PRO_P0_PROJECTION_POLICY-v001.md:51-58`). The Builder-authored normative
spec repeats `clipping` in `forbidden_nonconstant_paths`
(`tests/duca_projection/DUCA_P0_NONCONSTANT_PROJECTION_SPEC-v001.json:81-92`).
Nevertheless, the production nonconstant decoder computes a search interval and
then executes `left = left.clamp(min=0, max=valid_len - 2)`
(`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:454-465`).

This is reached by every nonconstant input. In particular, the last quantile is
exactly total mass, so `searchsorted(..., right=True) - 1` produces the terminal
sample index; the clamp changes that index to the last interval before the code
overwrites the target endpoint. The endpoint overwrite prevents an immediate
output error, but it does not make the implementation literally conformant to
the frozen no-clipping rule.

The authored conformance gate does not detect the violation: it searches the
projector source for `.clamp(`, while the decoder source is checked only for the
projector call, canonical-uniform symbol, constant-input equality, and typed
error (`tests/test_duca_p0_projection_policy.py:397-413`).

Required correction: form the two fixed endpoints explicitly and perform
inverse-CDF interval lookup only for internal quantile ranks, whose masses lie
strictly inside `(0,total_mass)`. Remove the decoder clamp and make the
no-clipping conformance check cover the nonconstant decoder. This is a local
implementation correction; it changes neither the frozen objective nor the
research route.

## Other bounded attacks

No additional finding was identified in the authorized surface:

- Candidate bounds match the frozen `L_j(r)`/`R_j(r)` equations and are visited
  in ascending physical-position order
  (`pc_ot_mras_prebackbone_frame_selector.py:95-115,247-266`).
- The exact lexicographic objective is implemented with arbitrary-precision
  integer stages and is recomputed against the returned certificate
  (`pc_ot_mras_prebackbone_frame_selector.py:117-168,275-415`).
- Binary64-to-`Q=2^20` conversion uses the serialized bit pattern and exact
  rational half-up arithmetic rather than host `round`
  (`pc_ot_mras_prebackbone_frame_selector.py:43-75,469-478`).
- Malformed domain inputs, infeasible states, comparison inconsistencies, and
  certificate mismatches use `DUCAProjectionError`; unexpected decoder failures
  are wrapped fail-closed (`pc_ot_mras_prebackbone_frame_selector.py:171-245,259-268,485-488`).
- The reference imports no production or Torch code and uses independently
  named reverse suffix states/candidate construction
  (`tests/duca_projection/DUCA_P0_PROJECTION_REFERENCE-v001.py:1-8,30-46,121-218`).
- The eight fixture objects preserve the required `T=385` exhaustive cases and
  `T=768` full-scale families. Four deterministic receipts are populated and
  the four execution-dependent receipts remain explicitly null/authored-not-run;
  the tests pass the same parsed fixture arrays to production and reference
  (`tests/test_duca_p0_projection_policy.py:264-333,416-432`).
- No detector, loss, split, metric, budget, evaluator, NMS, checkpoint, claim,
  data, or experimental surface is added by the bounded five-file review.

These positive static checks are not execution evidence and do not discharge
the later cross-implementation identity/optimality gate.

## Governance quarantine

Five probes created before the project-specific three-role restriction was
clarified remain interrupted and quarantined. Stop reason for all: this project
permits exactly Builder/Critic/Evaluator and forbids additional probes. Their
outputs were not read, used, or cited in this review.

- `019fef39-c800-71e3-b9b1-b1373a5c44e7` (`selector_audit`)
- `019fef39-e5f7-74a3-ac9c-5e87667f4a79` (`coordinate_audit`)
- `019fef3a-02e6-7553-ac4d-ee7437db2935` (`cost_audit`)
- `019fef3a-2ba3-7b50-ae56-a93bd1510af3` (`novelty_audit`)
- `019fef3a-50b9-7013-9843-fc5380f54f7c` (`spec_audit`)

## Boundary attestation

`NO_EXECUTION_ATTESTATION`: static text-only inspection. No implementation edit,
test, Python/model execution, data/checkpoint/held-out/metric access, browser,
remote operation, CPU/GPU workload, Slurm, experiment, Git operation, route or
claim change, result promotion, Pro/Sources action, or Builder-dialogue access
was performed. The only write is this requested durable Critic return.

`EVIDENCE_CLASS: BLOCKED_PRE_RESULT`
