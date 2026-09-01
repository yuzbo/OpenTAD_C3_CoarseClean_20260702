---
doc_id: CRITIC_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_PLAN_STATIC_REVIEW
version: v001
status: IMPLEMENTATION_CORRECTION
date: 2026-08-13
author_role: critic
queue_message_id: msg-20260812T205219Z-42871ab0b652
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
reviewed_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v001
builder_binding: BUILDER_DUCA_P0_CORRECTED_REMOTE_CLEAN_BINDING-v002
evaluator_materialization: EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001
execution_state: NOT_EXECUTED
required_gate: P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED
gate_disposition: BLOCKED_PENDING_IMPLEMENTATION_CORRECTION
---

# Verdict

`IMPLEMENTATION_CORRECTION`.

The plan correctly preserves the frozen mathematical key, `Q=1048576`, ordered `18/9/6` fixture matrix, typed negative and mutation codes, role-separated E1/B1/E2/C1 sequence, one-shot/fail-closed intent, evidence boundary, and `NOT_EXECUTED` status. It does not authorize E1, B1, or E2 prematurely.

However, four deterministic document-contract defects make the current plan internally inconsistent or weaker than the accepted Pro authority. No finding requires a scientific choice, so there is no `SCIENTIFIC_AMBIGUITY` and no Pro adjudication is requested.

## Findings

### P0-PRE-PLAN-001 — the sole allowed B1 entry point is also forbidden

Classification: `IMPLEMENTATION_CORRECTION`.

The plan binds B1 to `tools/bata/run_duca_p0_projection_production.py` and permits only that production entry point (`plan:42-43`, `138-144`), but then forbids Builder from invoking “adapter ... code” (`plan:146-151`) and requires that no admitted argv access an adapter (`plan:290-293`). The bound entry point is explicitly the production adapter. Under the literal plan, no B1 argv can be admitted.

Minimal correction: name the bound production adapter as the sole B1 exception and forbid only every other adapter, validator, detector/model path, decoder, or implementation call. Preserve the exactly-one production-batch limit.

### P0-PRE-PLAN-002 — global output-root absence is impossible after E1

Classification: `IMPLEMENTATION_CORRECTION`.

The phase contract requires E1 to create and seal `REFERENCE_OUTPUT_ROOT` before B1, and B1 to create and seal `PRODUCTION_OUTPUT_ROOT` before E2 (`plan:61-65`, `116-158`). Yet the admission section says no P0 command may issue until all three output roots are absent (`plan:262-280`). If interpreted for every phase as written, successful E1 makes B1 inadmissible and successful B1 makes E2 inadmissible.

Minimal correction: make root-state checks phase-relative:

- before E1: all three roots absent;
- before B1: sealed reference root present, production and comparison roots absent;
- before E2: sealed reference and production roots present, comparison root absent;
- before C1: all required E1/B1/E2 outputs present and sealed.

No deletion, reuse, overwrite, append, repair, or rerun should be added.

### P0-PRE-PLAN-003 — the reference-method allocation narrows the Pro contract

Classification: `IMPLEMENTATION_CORRECTION`.

The accepted Pro authority requires exhaustive ascending enumeration for the `T<=385` witnesses and independently structured exact shortest-path/DP for the full-scale witnesses (`PRO:60-66`). The plan enumerates only the five T17 tie fixtures and `G385-X`, then assigns every other positive fixture to DP (`plan:131-136`). That includes `G16-U`, `G31-U`, `G32-U`, `G383-U`, and `G384-U`, all within the frozen `T<=385` exhaustive domain.

Minimal correction: require exhaustive ascending enumeration for every applicable positive fixture with `T<=385` (singleton feasible sets remain valid exhaustive results), and reserve the independent exact DAG/shortest-path solver for the full-scale `T=767/768` fixtures. Do not change fixtures, expected winners, counts, objectives, or candidate order.

### P0-PRE-PLAN-004 — receipt obligations do not explicitly bind exact input bytes

Classification: `IMPLEMENTATION_CORRECTION`.

Pro requires production and reference to receive identical canonical JSON bytes and the global receipt to bind those fixture bytes; the first-discrepancy receipt must report them (`PRO:39-43`, `107-118`). The plan's execution prose preserves same-line bytes (`plan:67-74`), but E2 compares only “input object/line identity” (`plan:160-164`) and the receipt schema requires an “exact canonical input object” rather than the exact original JSON line bytes/text (`plan:203-231`). Parsed-object equality does not itself attest byte identity.

Minimal correction: require E1 and B1 to carry the exact original LF-stripped UTF-8 JSON line text (or an equivalently exact byte representation) plus one-based source line; require E2 to compare that representation byte-for-byte and include it in the first-discrepancy/global receipt. No new hash or checksum is needed.

## Static disposition

- Frozen mathematics, objective order, feasibility, `Q`, typed codes, and `18/9/6` ordering: `PRESERVED`.
- Independent-reference ownership and broad role order: `PRESERVED`, subject to `P0-PRE-PLAN-003`.
- Fairness/leakage boundary: `NO NEW SCIENTIFIC LEAKAGE FOUND`; the exact role prohibitions remain appropriate once `P0-PRE-PLAN-001` is made internally consistent.
- One-shot/fail-close boundary: `PRESERVED IN INTENT`, but no phase is admissible until the root-state and adapter contradictions are corrected.
- Scientific ambiguity: `NONE`.
- E1/B1/E2/C1 execution authority: `NOT GRANTED`.
- `P0_IDENTITY_GATE_STATIC_DEPENDENCIES_CLOSED`: `NOT SATISFIED` for this plan revision.

The cheapest closure is one corrected plan revision addressing only the four items above, followed by one focused static Critic recheck. This review performed document/source reading only: no runtime source was read, and no command, import, compile, test, materialization, projector, reference, adapter, remote, data, model, checkpoint, metric, GPU/CUDA/Slurm, or browser action was executed.
