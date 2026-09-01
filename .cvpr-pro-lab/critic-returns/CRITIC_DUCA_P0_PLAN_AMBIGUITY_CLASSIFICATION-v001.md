---
doc_id: CRITIC_DUCA_P0_PLAN_AMBIGUITY_CLASSIFICATION
version: v001
stage: DRAFT_P0_PLAN_REVIEW
author_role: critic
parent_message_id: msg-20260811T081502Z-754207b3e3fc
parent_decision: PRO_P0_BLOCKER_DECISION-v001
parent_builder_return: BUILDER_DUCA_P0_MINIMAL_CHANGE_PLAN-v002
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
primary_classification: SCIENTIFIC_AMBIGUITY
plan_viability: BLOCKED_PENDING_FRESH_PRO
evidence_class: BLOCKED_PRE_RESULT
---

# CRITIC_DUCA_P0_PLAN_AMBIGUITY_CLASSIFICATION-v001

## Frozen target and primary classification

The personally inspected Critic worktree was clean and exactly at
`63a726a4aaf48ecbf6780bb196de43a890c6b4df`. The review consumed durable queue
`msg-20260811T081502Z-754207b3e3fc`, `PRO_P0_BLOCKER_DECISION-v001`, and the
bound plan record `BUILDER_DUCA_P0_MINIMAL_CHANGE_PLAN-v002`. It did not inspect
the quarantined Builder worktree or any unbound patch.

**Primary classification: `SCIENTIFIC_AMBIGUITY`.**

The canonical-uniform and pre-NMS coordinate-transport repairs are deterministic
`IMPLEMENTATION_CORRECTION`s, but the P0 plan as a whole is not viable because
the required constant-density hard-forward specialization has no faithful,
named attachment point in the frozen implementation. Choosing or creating that
mechanism would define the model route rather than repair an interface.

## Evidence for the blocking ambiguity

The accepted decision requires positive temporal density, monotone inverse-CDF
positions, and a hard constant-density case that is bit-identical to the
canonical generator (`PRO_P0_BLOCKER_DECISION-v001:21-27,42-60`). The frozen
design describes `rho(t) -> F(t) -> F^{-1}`
(`docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md:55-74`),
but the frozen Python tree contains no density decoder, inverse-CDF decoder, or
constant-density specialization symbol.

The nearest existing symbols are not claim-preserving attachment points:

- `PCOTMRASReader._ordered_centers_and_widths` learns per-slot positive steps
  from a global slot state, while `_allocation` builds slot-specific soft
  allocation from content, local kernels, role signals, and process signals
  (`opentad/models/selectors/pc_ot_mras_reader.py:243-311`). Its `forward`
  returns slot allocation/centers rather than one positive per-time density and
  inverse-CDF hard positions (`:375-457`). Reinterpreting this reader as the
  frozen density mechanism changes the route.
- `PCOTMRASPreBackboneFrameSelector._sparse_transport_plan` hardens a slot
  allocation by per-slot argmax, then deduplicates and fills positions
  (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1510-1748`).
  `_frame_score_transport_plan` instead uses score top-k/global-rank behavior
  (`:2278-2517`). Neither implements density-CDF inversion; selecting one would
  choose different scientific behavior.
- `ActionFormer._pc_ot_mras_eval_override_outputs` and
  `_pc_ot_mras_exact_uniform_positions` provide an exact-uniform evaluation
  override only (`opentad/models/detectors/actionformer.py:406-466`). They have
  no learned density input or hard-decoder wrapper to specialize.

The unresolved choice includes which frozen tensor, if any, is `rho(t)`, how it
is made positive, how inverse-CDF quantiles become strict unique integer
positions under the bounded-warp constraints, and where this hard decoder
enters the pre-backbone selector. Those choices affect the mechanism,
falsifier, and route semantics, so Critic cannot convert them into an
implementation instruction.

## Deterministic surfaces that do not require a scientific change

### Canonical uniform

The integer half-up formula and `K_eff` rule are fully frozen. A shared pure
generator can replace the current duplicated arithmetic without changing the
detector, metric, split, or claim. The named call sites are:

- `LoadFrames._exact_uniform_dense_positions`
  (`opentad/datasets/transforms/end_to_end.py:337-348`);
- `PCOTMRASPreBackboneFrameSelector._uniform_anchor_positions`
  (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:3158-3190`);
- `ActionFormer._pc_ot_mras_exact_uniform_positions`
  (`opentad/models/detectors/actionformer.py:459-466`) if that evaluation
  override remains route-reachable.

The third symbol must either call the one shared generator or be proven outside
the authorized route; otherwise the Pro prohibition on a second generator is
not closed. This is an `IMPLEMENTATION_CORRECTION`, not a Pro ambiguity.

### Pre-NMS selected-to-physical transport

The frozen tree already contains the interpolation semantics in
`_selected_axis_segments_to_dense_axis`
(`opentad/models/utils/post_processing/utils.py:73-106`), but
`SingleStageDetector.post_processing` currently applies non-sliding
`batched_nms` before `convert_to_seconds` invokes that mapping
(`opentad/models/detectors/single_stage.py:107-145`). A minimal attachment point
is immediately after each sample's raw `segments` tensor is extracted at
`SingleStageDetector.post_processing:108-110`, before score filtering, top-k,
IoU, or NMS. A coordinate-state guard can make unknown and double mapping fail
closed, and later seconds conversion can require the already-physical state.
This preserves scores, labels, head/loss, NMS callable/configuration, evaluator,
split, and class map. It is an `IMPLEMENTATION_CORRECTION`.

## Smallest fresh-Pro question

**Which frozen per-time tensor/symbol is the authoritative positive density
input and what exact minimal hard-decoder API is authorized? If no frozen symbol
is authoritative, should P0 authorize a new route-specific density decoder
(including positivity transform, quantiles, strict-integer collision/bound
rules, and selector insertion point), or should constant-density identity be
deferred until the density route itself is separately specified?**

Until Pro answers, Builder may plan the shared canonical generator and the
pre-NMS coordinate adapter, but must not invent, attach, or reinterpret a
density mechanism. The P0 patch gate therefore remains blocked.

NO_EXECUTION_ATTESTATION: read-only source/plan inspection only; no patch, test,
Python/model execution, data access, metric computation, remote operation, GPU,
Slurm, or experiment was performed.

EVIDENCE_CLASS: `BLOCKED_PRE_RESULT`
