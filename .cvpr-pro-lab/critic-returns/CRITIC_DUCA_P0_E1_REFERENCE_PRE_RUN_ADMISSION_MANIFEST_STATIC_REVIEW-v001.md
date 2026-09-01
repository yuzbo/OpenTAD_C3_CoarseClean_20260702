---
doc_id: CRITIC_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_STATIC_REVIEW
version: v001
status: IMPLEMENTATION_CORRECTION
date: 2026-08-13
author_role: critic
queue_message_id: msg-20260812T211251Z-fbaf773600ca
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v002
reviewed_manifest: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v001
execution_state: NOT_EXECUTED
remaining_scientific_ambiguity: NONE
required_gate: P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED
e1_admission_disposition: BLOCKED_PENDING_IMPLEMENTATION_CORRECTION
---

# Verdict

`IMPLEMENTATION_CORRECTION`.

The candidate preserves the accepted scientific contract: exact `Q=1048576`, ordered `18/9/6` IDs, frozen objective and candidate order, exhaustive positive `T<=385` allocation, independent exact DAG/shortest-path allocation for positive `T=767/768`, LF-stripped UTF-8 source-line binding, Evaluator reference ownership, E1-before-B1 separation, phase-relative E1 root state, one-shot/first-fail intent, zero-forbidden-access boundary, and conformance-only evidence class. It also remains explicitly `NOT_EXECUTED`, `NOT_READY`, and non-authorizing.

Two deterministic envelope defects prevent admission. Neither changes the mechanism, fixtures, mathematics, protocol, evidence class, claim, route, or stop rule; therefore no `SCIENTIFIC_AMBIGUITY` exists and no Pro adjudication is required.

## Findings

### P0-E1-MANIFEST-001 — completion status does not validate the mandatory reference-evidence schema

Classification: `IMPLEMENTATION_CORRECTION`.

The inline program checks each fixture only for `typed_status` and, for positives, the reference-method label before appending the opaque `reference_result` (`manifest:248-287`). It then declares `E1_REFERENCE_COMPLETE` whenever those checks and the mutation status checks finish (`manifest:349-409`).

That is weaker than both plan v002 and the manifest's own stated completion contract. A complete E1 receipt must contain, for every successful fixture, exact `p`, feasibility fields, all objective components, interior vector, candidate-order evidence, and global-optimality evidence; it must also contain complete exhaustive T17/T385 witnesses and ties and independent T767/T768 root-optimum evidence (`plan-v002:263-293`; `manifest:469-502`). The program does not fail closed when any of those fields or witness obligations is absent, malformed, incomplete, or inconsistent with the declared method. It likewise does not enforce null success-only fields for negative records before calling the artifact complete.

This is consequential under the one-shot contract: an incomplete E1 package could be sealed as complete and make B1 eligible, after which the missing evidence cannot be repaired or rerun under the same identity.

Minimal correction: before appending/accepting each result and before setting a complete receipt, validate the required result shape and phase-owned evidence obligations. In particular, require all successful-result fields; exact exhaustive witness evidence for every positive `T<=385` fixture, including complete 15-candidate T17 and 383-candidate G385-X evidence and frozen ties; independent root-optimum evidence for each positive `T=767/768` fixture; and null success-only fields for negatives. Any missing or malformed obligation must become the first `E1_REFERENCE_BLOCKED` failure. This is schema/evidence validation only and must not add a second solver, recomputation, tolerance, fallback, or new fixture.

### P0-E1-MANIFEST-002 — publication may write outside the declared E1 boundary

Classification: `IMPLEMENTATION_CORRECTION`.

The manifest states that E1 may write only the new `REFERENCE_OUTPUT_ROOT` and its temporary sibling (`manifest:83-86`). The inline program creates the sibling with `partial_root.mkdir(parents=True)` (`manifest:411-419`) but never requires the shared parent `.../p0/identity-gate-v001` to exist. If that parent is absent, `parents=True` creates it, producing a write outside both declared E1 output paths and implicitly preparing the namespace shared by later B1/E2 roots.

Minimal correction: make the exact shared parent an existing, admitted precondition and create only the absent temporary sibling with non-recursive exclusive creation. If the parent is absent or not the expected project-local directory, fail before the E1 command rather than widening E1's write boundary. Preserve atomic sibling-to-output promotion and the no-cleanup/no-retry rule.

## Static disposition

- Frozen mathematics and fixture/reference allocation: `PRESERVED`.
- Exact input-line byte binding without hashes: `PRESERVED`.
- Reference independence and fairness/leakage boundary: `PRESERVED` in the authored contract; no new scientific leakage found.
- One-shot/first-fail boundary: `PRESERVED IN INTENT`, but incomplete-result admission in `P0-E1-MANIFEST-001` must be closed before execution.
- `P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED`: the previously accepted plan prerequisite is not reopened.
- E1 manifest admission: `BLOCKED_PENDING_IMPLEMENTATION_CORRECTION`.
- `PRE_RUN_READY`: `NOT ESTABLISHED`.
- E1/B1/E2/C1 execution authority: `NOT GRANTED`.
- Scientific ambiguity: `NONE`.

The cheapest closure is one corrected E1 manifest revision addressing only these two findings, followed by one focused static Critic recheck. This review read documents only. It did not read runtime projector/reference/adapter surfaces or perform any shell/Git command, import, compile, test, materialization, remote/SSH, data, model, checkpoint, metric, GPU/CUDA/Slurm, or browser action.
