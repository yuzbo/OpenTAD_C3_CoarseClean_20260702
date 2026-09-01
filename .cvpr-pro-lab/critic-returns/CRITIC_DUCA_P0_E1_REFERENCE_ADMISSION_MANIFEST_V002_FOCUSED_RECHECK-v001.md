---
doc_id: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V002_FOCUSED_RECHECK
version: v001
status: IMPLEMENTATION_CORRECTION
date: 2026-08-13
author_role: critic
queue_message_id: msg-20260812T212751Z-3c2602e85164
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
reviewed_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v002
prior_critic_review: CRITIC_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_STATIC_REVIEW-v001
execution_state: NOT_EXECUTED
scientific_ambiguity: NONE
external_blocker: NONE
e1_admission_disposition: BLOCKED_PENDING_IMPLEMENTATION_CORRECTION
---

# Verdict

`IMPLEMENTATION_CORRECTION`.

`P0-E1-MANIFEST-002` is closed: v002 requires the exact pre-existing admitted shared parent, rejects a missing/different/unusable parent before any write, and creates only the temporary sibling with `parents=False, exist_ok=False` before atomic promotion (`manifest-v002:35-66`, `386-398`, `615-626`, `660-673`).

`P0-E1-MANIFEST-001` is improved but not closed. The new validator checks presence and basic shape of many fields, yet it can still seal `E1_REFERENCE_COMPLETE` with malformed or semantically non-evidentiary mandatory fields.

## Remaining finding

### P0-E1-MANIFEST-001R — mandatory evidence is not validated to the frozen contract

Classification: `IMPLEMENTATION_CORRECTION`.

Exact evidence:

- `validate_positive` verifies that minimum/maximum stride and maximum displacement are integers, but never verifies the frozen constraints `minimum_stride>=1`, `maximum_stride<=4`, `maximum_uniform_displacement<=16`, or that these values equal the extrema recomputable directly from returned `p` and sealed `u` (`manifest-v002:303-323`). Thus a result can claim infeasible stride/displacement evidence and still complete.
- Candidate-order evidence is accepted whenever merely non-null/nonempty (`manifest-v002:324-330`). A string such as `"unknown"` or unrelated object satisfies this check; it does not establish ascending candidate order as required by Pro and plan v002.
- Exhaustive tie records check only the `through` label and number of entries (`manifest-v002:286-301`). They do not validate that each tied candidate is a full length-`K` integer vector, corresponds to a candidate in the complete witness, has the required equal objective prefix, or matches the frozen T17/G385 tied alternatives. A two-element list of arbitrary values can pass.
- Full-scale root evidence requires `root.p` and `root.objective` to equal the result, but accepts any nonempty `root.certificate` value (`manifest-v002:353-361`). This does not fail closed on a malformed root-optimality certificate and therefore does not establish the mandatory independent T767/T768 root-optimum evidence.

These are not requests for a second solver or objective recomputation. They are direct schema/consistency checks over the already returned result, sealed `u`, and witness objects. Without them, the one-shot E1 package can still be sealed as complete despite malformed mandatory evidence, the precise failure class identified in `P0-E1-MANIFEST-001`.

Minimal correction: validate feasibility values directly against returned `p` and sealed `u`; require candidate-order evidence in the declared ascending form; validate each tie candidate's shape, membership, and objective-prefix equality plus the frozen tied alternatives; and require a defined nonempty root-certificate structure whose required fields are internally consistent with the returned root/result. Fail at the first missing, malformed, or inconsistent field. Do not add a solver, recompute optimality, alter fixtures, or broaden the route.

## Preserved scope

The candidate continues to preserve exact `Q=1048576`, ordered `18/9/6` IDs, frozen objective, exhaustive positive `T<=385` versus independent exact DAG/shortest-path positive `T=767/768` allocation, exact LF-stripped UTF-8 input-line binding, Evaluator reference independence, phase-relative root states, one-shot/first-fail intent, no hashes/tolerance/fallback/rerun, and conformance-only evidence scope.

It remains `AUTHORED_NOT_EXECUTED`, `NOT_PRE_RUN_READY`, and non-authorizing. No E1/B1/E2/C1/P0 execution authority is granted. There is no `SCIENTIFIC_AMBIGUITY` and no `BLOCKED_EXTERNAL` condition.

This was a document-only static review. No runtime reference/projector/adapter surface was read; no command, import, compile, test, materialization, remote/SSH, data, model, checkpoint, metric, GPU/CUDA/Slurm, or browser action was performed.
