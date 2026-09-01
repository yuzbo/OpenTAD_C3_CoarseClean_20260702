---
doc_id: CRITIC_DUCA_P0_STATIC_PATCH_REVIEW
version: v001
stage: DRAFT_P0_STATIC_REVIEW
author_role: critic
parent_message_id: msg-20260811T194645Z-532a006b6b40
parent_decision: PRO_P0_ROUTE_ADJUDICATION-v002
parent_builder_return: BUILDER_DUCA_P0_BOUNDED_PATCH_RETURN-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
builder_snapshot: C:\Users\skywalker\.codex\worktrees\07c1\OpenTAD_C3_CoarseClean_20260702
verdict: P0_BLOCKED
primary_classification: IMPLEMENTATION_CORRECTION
evidence_class: BLOCKED_PRE_RESULT
---

# CRITIC_DUCA_P0_STATIC_PATCH_REVIEW-v001

## Verdict and binding

`P0_BLOCKED`.

This review consumed durable queue `msg-20260811T194645Z-532a006b6b40`, accepted
decision `PRO_P0_ROUTE_ADJUDICATION-v002`, and the bound
`BUILDER_DUCA_P0_BOUNDED_PATCH_RETURN-v001`. The personally inspected Builder
snapshot was at the declared frozen base
`63a726a4aaf48ecbf6780bb196de43a890c6b4df`; the authored surface matched the
ten declared implementation/config/fixture paths. No Builder dialogue or
quarantined patch was inspected.

Both blockers below are `IMPLEMENTATION_CORRECTION`s. They can be repaired
without changing the accepted density-quantile mechanism, budget, decoder,
scientific route, experiment contract, or claim scope.

## Blocker 1 — effective config changes the detector and bypasses the required adapter

The accepted decision says the route is not a detector change and requires an
exactly-once selected-q to physical-dense adapter before filtering/top-k/NMS,
with detector/head and losses unchanged
(`PRO_P0_ROUTE_ADJUDICATION-v002:31-34,65-68`). The new config instead inherits
`pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py`
(`configs/adatad/thumos/duca_fixedk_bounded_density_quantile_acquisition_actionformer_n16r4.py:1`).
That parent explicitly declares `changes_detector_head=True` and
`changes_loss_assignment=True` and enables the physical-grid ActionFormer head
(`configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py:14-44,195-209`).

The same DUCA config sets `remap_gt_to_selected_axis=False` (`:79-116`). The
selector therefore writes `irregular_native_axis=True`
(`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:3539-3547`),
and `selected_axis_segments_to_dense_axis` immediately returns its input when
that flag is true (`opentad/models/utils/post_processing/utils.py:73-75`). Thus
the authored adapter at the start of `SingleStageDetector.post_processing`
(`opentad/models/detectors/single_stage.py:109-112`) is not active for the
declared DUCA route.

Smallest counterexample: take selected positions `[0,2,10,20]` and an unchanged
selected-axis head proposal `[1,3]`. The required pre-NMS physical proposal is
`[2,20]`; the DUCA config's `irregular_native_axis=True` state leaves it `[1,3]`.
Keeping the inherited physical-grid head avoids that numeric mismatch only by
violating the separate frozen requirement that detector/head and loss
assignment remain unchanged. The active config therefore cannot satisfy both
requirements.

Required correction: inherit the unchanged detector/head/loss stack, use the
selected-axis coordinate state for this route, and add a config-coupled fixture
that proves this exact config sends physical-dense coordinates into NMS. This is
claim-neutral and does not require Pro adjudication.

## Blocker 2 — the configured reader is not dedicated density-only

The accepted decision requires `duca_density_logits[b,t]` from a dedicated
density-only reader over dense `browser_memory`, with legacy slot allocation,
frame-selection/action/boundary signals, soft transport, ranking, quota, and
dynamic budget not serving as density (`PRO_P0_ROUTE_ADJUDICATION-v002:36-41`).
The density projection itself does not alias those returned tensors: it reads
`browser_memory` and applies its own LayerNorm/Linear projection
(`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1319-1335,1405-1425`).
That causal independence is correct.

However, the config instantiates the full legacy `PCOTMRASReader`, not a
density-only reader
(`configs/adatad/thumos/duca_fixedk_bounded_density_quantile_acquisition_actionformer_n16r4.py:107-116`).
The selector calls that full reader before applying the density projection
(`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1399-1425`).
Its frozen forward path computes dense action/boundary-style heads, slot state,
ordered centers/widths/gates, allocation logits, soft allocation and
`acquisition_matrix`, selected tokens, regularizers, and—because the new config
does not disable the default—pair distribution, before returning
`browser_memory` (`opentad/models/selectors/pc_ot_mras_reader.py:375-457`). The
authored fixture only proves output non-aliasing and explicitly asserts the
legacy reader type; it does not prove a dedicated density-only execution path
(`tests/test_c3_physical_grid_actionformer_candidate.py:273-318,321-335`).

Smallest counterexample: one valid-prefix sample reaches the DUCA selector. A
soft `acquisition_matrix` and legacy head/pair outputs are computed before the
sole density tensor is projected, even though the hard decoder later ignores
them. Therefore the effective mechanism and cost surface are not the declared
density-only reader.

Required correction: expose a dedicated browser-memory density path that emits
only the valid-prefix memory/density needed by the frozen decoder and does not
execute the legacy slot/head/soft-transport/pair policy branches. This preserves
the Pro-selected scientific mechanism.

## Static checks that are correct

- The canonical endpoint-inclusive integer-half-up generator is shared by the
  authored route call sites, and constant T=768/K=384 decoding ends at 767
  (`opentad/utils/temporal_positions.py:4-20`;
  `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:39-47`).
- The nonconstant decoder implements finite checks, `1e-6 + softplus`,
  trapezoidal masses, endpoint quantiles, `K_eff`, deterministic projection,
  strict ordering, span/displacement bounds, and fail-closed postconditions
  (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:25-128`).
- The DUCA plan uses hard indices with unit weights and supplies no detector
  gradient surrogate/soft transport weights
  (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1662-1744`).
- The new `SingleStageDetector` call is structurally before score filtering,
  top-k, and NMS (`opentad/models/detectors/single_stage.py:109-142`); blocker 1
  is the route's effective config/state, not this local order.
- No test/held-out/teacher/cache/data/metric evidence is introduced or promoted.

## Governance quarantine

Five probes created before the project-specific three-role restriction was
clarified were interrupted while running and quarantined. Stop reason for all:
the project permits exactly Builder/Critic/Evaluator and forbids extra probes;
their outputs were not read, used, or cited.

- `019fef39-c800-71e3-b9b1-b1373a5c44e7` (`selector_audit`)
- `019fef39-e5f7-74a3-ac9c-5e87667f4a79` (`coordinate_audit`)
- `019fef3a-02e6-7553-ac4d-ee7437db2935` (`cost_audit`)
- `019fef3a-2ba3-7b50-ae56-a93bd1510af3` (`novelty_audit`)
- `019fef3a-50b9-7013-9843-fc5380f54f7c` (`spec_audit`)

NO_EXECUTION_ATTESTATION: static read-only implementation inspection only; no
implementation edit, test, project Python/model execution, data/checkpoint/
metric access, browser, remote operation, project CPU/GPU workload, Slurm,
experiment, Git stage/commit/push, route change, claim change, or result
promotion was performed. Only authorized control-plane read/write/seal/queue
operations were used to return this evidence.

EVIDENCE_CLASS: `BLOCKED_PRE_RESULT`
