---
doc_id: CRITIC_DUCA_P0_IDENTITY_PRE_RUN_STATIC_DEPENDENCY_RECHECK
version: v001
stage: DRAFT_P0_IDENTITY_PRE_RUN
author_role: critic
parent_message_id: msg-20260812T201650Z-afa60317f081
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_builder_return: BUILDER_DUCA_P0_REMOTE_CLEAN_BINDING-v001
parent_evaluator_return: EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001
base_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
execution_revision: df8228072b871adbd8dedb480e80f1f7daaca69e
classification: IMPLEMENTATION_CORRECTION
scientific_ambiguity: NONE
verdict: PRE_RUN_STATIC_DEPENDENCIES_BLOCKED
status: NOT_EXECUTED
evidence_class: BLOCKED_PRE_RESULT
---

# Critic P0 identity PRE_RUN static dependency recheck

## Frozen target and boundary

- Consumed exactly durable queue `msg-20260812T201650Z-afa60317f081`.
- Authority: accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`.
- Reviewed only the two named durable preparation receipts and the read-only Git
  diff/tree for
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df..df8228072b871adbd8dedb480e80f1f7daaca69e`.
- No production/reference/comparison, test, Python, import, compile,
  materialization, data/model/checkpoint/metric, GPU/CUDA/Slurm, SSH, browser,
  Pro, Sources, or experiment surface was accessed or executed.

## Static evidence that is complete

1. `BUILDER_DUCA_P0_REMOTE_CLEAN_BINDING-v001` identifies a detached clean
   Builder snapshot at exact revision `df8228072b871adbd8dedb480e80f1f7daaca69e`,
   direct parent `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, zero porcelain
   entries, and remote root
   `/data/run01/sczc063/yuzibo/projects/opentad_duca_p0_identity_df822807_20260813`.

2. `EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION-v001` identifies
   a distinct, recursively non-writable evaluator root, the exact three source
   copies, the exact three materialized files, the frozen `18/9/6` membership,
   27 canonical JSONL records, and zero production/reference/comparison access.
   It correctly remains preparation evidence rather than a gate result.

3. Read-only Git inspection confirms the Builder receipt's direct-parent and
   two-path scope: the execution revision modifies only
   `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py` and adds
   `tests/test_duca_p0_projection_policy.py`. The production diff contains the
   frozen integer projector, typed negative codes, exact objective/candidate
   order, certificate, and closed C-PROJ-001 decoder; no data, detector metric,
   checkpoint, launcher, or experiment path is added by this diff.

## Finding P0-PRE-STATIC-001 — clean revision contains a non-self-contained test surface

Classification: `IMPLEMENTATION_CORRECTION`.

The added test module declares these required repository-local inputs and loads
them at module import/collection time:

- `tests/duca_projection/DUCA_P0_NONCONSTANT_PROJECTION_SPEC-v001.json`;
- `tests/duca_projection/DUCA_P0_PROJECTION_FIXTURES-v001.json`;
- `tests/duca_projection/DUCA_P0_PROJECTION_REFERENCE-v001.py`.

Evidence: `tests/test_duca_p0_projection_policy.py:25-43` constructs the three
paths, executes `_load_reference()`, and immediately reads the two JSON files.
Read-only `git ls-tree -r --name-only df822807... -- tests/duca_projection
tests/test_duca_p0_projection_policy.py` returns only
`tests/test_duca_p0_projection_policy.py`; none of the three required support
files is present in the clean execution revision.

On the registered Linux execution environment, the Windows module skip does not
apply (`tests/test_duca_p0_projection_policy.py:16-19`). Consequently, collection
or any attempt to use this tracked test as the clean snapshot's static
conformance/gate interface reaches missing repository files before it can verify
the projector. The evaluator's sealed remote files do not repair this path: they
have different canonical names, live in a separate role-owned root, and the
tracked test is hard-coded to repository-local paths.

This is not a scientific disagreement and does not question the sealed matrix,
reference mathematics, projector policy, fixture membership, route, claim,
metric, split, threshold, budget, or protocol. It is a deterministic packaging
and interface defect in the exact proposed execution revision.

## Minimal correction required before static dependency closure

Builder/Coordinator must provide one internally consistent gate-facing clean
revision and rebind it before PRE_RUN planning is treated as closed. The minimal
correction is either:

1. include the exact tracked support artifacts required by the test, with their
   authority and role boundary consistent with the accepted sealed Evaluator
   package; or
2. remove the unusable test from the formal execution snapshot and name a
   separate bounded production adapter that consumes the sealed Evaluator JSONL
   directly, explicitly forbidding pytest/test-module use in the identity gate.

This choice is implementation-only. It must not revise fixtures, expected
winners, reference logic, projector mathematics, or role order. Any corrected
revision requires a new clean binding receipt; no execution is needed to author
the correction.

## Verdict and remaining NOT_EXECUTED dependencies

`PRE_RUN_STATIC_DEPENDENCIES_BLOCKED` by `P0-PRE-STATIC-001`.

`SCIENTIFIC_AMBIGUITY: NONE`. No Pro/browser action is warranted for this
deterministic correction.

After correction and clean rebind, a later `NOT_EXECUTED` execution-plan
candidate must still bind, without executing:

- the corrected Builder revision/root and sealed Evaluator root;
- exact fixture/index/mutation paths and frozen `18/9/6` order;
- fixed-Q envelope validation and one production call per sealed fixture;
- production/reference role order, exact typed statuses, feasibility/objective/
  candidate-order receipts, stop-on-first-discrepancy behavior, and
  `scope_deviation="none"`;
- distinct output roots and explicit zero data/model/checkpoint/metric/GPU/
  browser access;
- separate durable authorization before any production, reference, or comparison
  command is issued.

`NOT_EXECUTED`. This receipt is static dependency evidence only. It starts no P0
identity/optimality step and establishes no identity, optimality, metric, cost,
efficacy, P1, or paper claim. The only write is this queue-required durable
Critic return in the canonical control plane.
