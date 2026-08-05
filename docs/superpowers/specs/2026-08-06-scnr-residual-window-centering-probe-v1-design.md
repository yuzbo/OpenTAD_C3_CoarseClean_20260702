# SCNR residual-window centering probe v1 design

## Decision

The next intervention is one independently switchable calibration transform on
the residual role modifier:

`georoute_branch_calibration_mode = "residual_window_center"`.

It is a mechanism repair candidate, not a new selector, role quota, loss, or
performance claim. The existing Scheme-A selector remains

\[
u_{btn}=q^{base}_{btn}+\max(0,\Delta^{roi}_{btn},\Delta^{res}_{btn}).
\]

The user has already approved direct execution once the route and next step are
clear, so this frozen design is the review boundary before implementation. No
additional Pro-model discussion is required for this single-variable probe.

## Evidence that selects this variable

Terminal M2 preserves exact `B=24576`, true ragged execution, masked-zero
carrier, and observes dynamic `K_t=0`, but its selected roles collapse to
context/ROI/residual `0/7/3,342,329` for G1 and `0/0/3,342,336` for G2.

The same-GPU serial OFF/ON pair showed instrumentation neutrality but exposed
ordinary frozen-checkpoint prediction drift. A legacy-backend OFF-A/OFF-B/ON
triplet then showed that even OFF-A and OFF-B drift. The first downstream
nondeterministic operation was memory-efficient CUDA scaled-dot-product
attention after routing. A strict math-SDPA triplet made OFF-A, OFF-B, and ON
prediction files byte-identical in both arms. Because that backend differs from
the historical source, only a separately validated categorical bridge was
opened; continuous score, margin, geometry, prediction, and performance
comparisons remain closed.

The categorical bridge shows that the hard role partition is stable across the
legacy and strict runs for all 136 windows. Before global top-B, residual wins
`99.9767522%` of G1 valid candidates and `99.9914355%` of G2 valid candidates;
context wins zero in both arms. Therefore the first failure is branch-offset
identifiability at role assignment, not top-B eliminating an otherwise diverse
role set.

## Transform

For batch/window `b`, boolean validity mask `m`, and residual modifier
`Delta_res`, compute one differentiable mean over every valid native candidate
in the complete 384-tubelet window:

\[
\mu_b =
\frac{\sum_{t,n}m_{btn}\Delta^{res}_{btn}}
     {\sum_{t,n}m_{btn}},
\qquad
\widetilde{\Delta}^{res}_{btn}
=\Delta^{res}_{btn}-\mu_b\quad\text{when }m_{btn}=1.
\]

Invalid entries are preserved and remain excluded by the existing validity
mask. The mean is not detached: this is an identifiable centered parameterization
whose valid residual gradients have zero-sum coupling. It changes neither
`q_base` nor `delta_roi`; the context modifier remains exactly zero.

The calibrated residual is applied immediately before the unchanged
`select_dynamic_scnr_exact_budget` call, so hard role assignment, global ranking,
exact-B, per-tubelet `K_t`, true-ragged packing, and masked-zero reconstruction
retain their existing implementations.

## Why the scope is the complete window

Per-tubelet centering would force a zero-mean residual competition independently
at every time step and therefore inject a temporal-uniformity prior into dynamic
`K_t`. It is not the first ablation. RMS matching, standardization, temperature,
`tanh`, clipping, and learned scaling alter magnitude as well as offset. An
ROI-conditioned residual complement introduces a second mechanism. Those options
remain separately discussable only if the offset-only probe fails.

## Compatibility and observability

- Default mode is `none`; every historical config and checkpoint keeps old
  behavior without state-dict changes.
- Only dynamic SCNR accepts `residual_window_center`; other routes reject a
  non-`none` setting rather than silently ignoring it.
- The route audit records the mode, valid-candidate mean before calibration, and
  raw versus effective residual modifier without changing formal route fields.
- No fixed context/ROI/residual count, role fraction target, reassignment,
  independent `q_ctx`, teacher, GT, prediction cache, padding token, or new loss
  is introduced.

## Frozen-checkpoint mechanism gate

Run both immutable M2 G1/G2 EMA checkpoints on the exact 136-window development
population with strict math SDPA and the opt-in centering mode. This probe may
write route telemetry but must not evaluate or interpret mAP, choose a geometry
floor, profile cost, train, resume, or open official test.

Hard integrity requirements:

1. exact source config/checkpoint/population receipts and clean runtime;
2. deterministic duplicate replay and exact route/audit self-hashes;
3. selected count exactly `B=24576` per window, valid-only support, unchanged
   ragged and masked-zero invariants;
4. no role quota, `q_ctx`, post-hoc reassignment, GT/teacher/cache input, or
   performance consumer.

The mechanism gate passes only if, in both G1 and G2:

- aggregate valid context and ROI wins are each nonzero;
- aggregate selected non-residual count is nonzero in at least one window;
- residual no longer wins 100% of valid candidates;
- every integrity requirement passes.

These structural `>0` conditions show only that the dormant branches are
reachable. They do not establish useful balance or detector benefit. If the gate
fails, record a negative result and do not train. If it passes, it authorizes a
new matched development-only centering-versus-`none` training protocol under the
same Scheme A and exact budget; it still does not authorize M3, official test,
efficiency, floor selection, complementarity, or a paper claim.

