# Builder bounded implementation correction — P0 typed failure interface

Authority: `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001` and the completed
`BUILDER_DUCA_P0_IDENTITY_PRODUCTION_PLAN-v001`, section 8. This is a direct
`IMPLEMENTATION_CORRECTION`; it does not reopen a scientific route decision.

Working snapshot: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`. Your development
worktree is known dirty. Preserve all pre-existing changes: inspect before
editing, touch only the minimal projector/error surface below, and never reset,
revert or overwrite unrelated work.

Implement the smallest deterministic change in
`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py` so the
existing `DUCAProjectionError` carries a stable machine-readable `code`, and
the existing validation paths can emit only these accepted P0 failure codes:

`INVALID_T_LT_16`, `K_EFF_MISMATCH`, `U_LENGTH_MISMATCH`,
`A_LENGTH_MISMATCH`, `U_CANONICAL_MISMATCH`, `A_ENDPOINT_MISMATCH`,
`A_ORDER_MISMATCH`, `INFEASIBLE`, and `INTEGER_RANGE_OR_OVERFLOW`.

Keep the already frozen solver math, Q, canonical arithmetic, feasible bounds,
candidate ordering, exact objective, tie rule, certificate, decoder closure,
detector, data, configuration, split, loss, metric, budget and claim unchanged.
Do not add a fallback, clipping, deduplication, tolerance, second decoder,
optimizer, cache, framework, config, launcher or unrelated tests.

Add only focused source-level regression coverage sufficient to prove code-field
propagation and the frozen negative-code mapping. Do **not** execute Python,
pytest, production code, data, CPU/GPU, Slurm, browser or any experiment in this
task; the later constrained gate must execute remotely under its separate
PRE_RUN receipt.

Return `BUILDER_DUCA_P0_TYPED_FAILURE_INTERFACE_CORRECTION-v001.md` with the
exact diff paths, mapping from validation branch to code, tests authored but
`NOT_EXECUTED`, all pre-existing dirty paths preserved, and classification.
If the correction would alter any frozen mathematics or needs a second solver,
stop with `NEEDS_ATTENTION` and describe the exact conflict.
