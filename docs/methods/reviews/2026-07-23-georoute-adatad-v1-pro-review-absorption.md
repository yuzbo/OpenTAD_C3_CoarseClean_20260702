# GeoRoute-AdaTAD v1 External Review: Absorption Record

**Date:** 2026-07-23
**Review status:** `HOLD`
**Project response:** `PARTIALLY_ACCEPTED_WITH_EXPLICIT_REVISIONS`
**Candidate status after absorption:** `designed`; no GeoRoute model code,
deployment, development metric, official-test opening, or paper claim exists.

## Provenance and scope

- Raw review: `2026-07-23-georoute-adatad-v1-pro-review-raw.md`.
- User attachment SHA-256:
  `61A1918B36D811F178152F1E9DE60B464186D9C52678722BA679D617F4468E78`.
- The review accessed historical commit
  `8ebe5f069494dc2efb3d4f9dc1ea3a2fbb51f89c` and correctly reported that the
  GeoRoute design files were absent at that commit. It therefore audits the
  pre-existing Continuous-ROI/AdaTAD baseline and evaluates the GeoRoute
  proposal supplied outside that snapshot. It is not a line-by-line audit of
  nonexistent GeoRoute implementation code.

## Findings accepted without reservation

1. **`HOLD` is correct.** No full training, A-MoD integration, paper claim, or
   official-test use is authorized before a native-token, one-heavy-forward P0
   vertical slice proves semantic and detector-contract parity.
2. **Current-code distinction is correct.** Historical U128 is fixed-output
   `grid_sample` resampling, contains two VideoMAE evaluations, has no learned
   selector/policy, and is not GeoRoute evidence.
3. **Detector-loss correction is correct.** The audited current ActionFormer
   configuration has Focal classification and DIoU regression losses; it has
   no independent quality head or quality loss. All GeoRoute documentation was
   corrected accordingly.
4. **Native tubelet semantics are correct.** A VideoMAE `2 x 16 x 16` token
   must use one absolute spatial patch over its two source frames. Continuous
   ROI geometry can change membership across tubelets but cannot make the two
   frames of one tubelet refer to separate spatial patches.
5. **AdaTAD contract is the right preservation target.** GeoRoute must rebuild
   the dense detector-facing `[B, 384, 768]` sequence and preserve current
   projection, head, target assignment, losses, and NMS unless a new matched
   detector baseline is explicitly declared.
6. **A-MoD must be interval routing.** The required candidate is an initial
   dense prefix followed by `Dense -> MoD -> Dense -> MoD -> ...`; every MoD
   block consumes the immediately preceding dense block's complete
   attention-derived score. A consecutive all-MoD tail is rejected.
7. **SDPA/FlashAttention needs an honest audit.** Full attention maps may be
   materialized only in tiny numerical known-answer tests. A-MoD is excluded
   from the main model unless its score path, `C=1` parity, capacity behavior,
   and end-to-end cost pass a separate P0.
8. **The central falsification is right.** Free TokenSelect-only, ROI-only,
   and ROI-plus-residual must share the exact same budget and detector setup.
   If free selection wins the relevant high-tIoU/cost Pareto, ROI must lose its
   primary paper claim.
9. **Cost claims must be full-stack.** Decode, preprocessing, H2D, scout,
   routing, gather, patch embedding, backbone, adapter, detector, NMS,
   latency, memory, and energy all belong in the ledger. ToMe is an internal
   token-merging comparison, not an equivalent pre-backbone spatial method.
10. **No privileged manual/oracle crop and no early official test.** These are
    fully retained.

## Findings accepted with a different operational decision

### Hard-policy gradient estimator

The review is right that hard exact-K membership has no ordinary pathwise
derivative with respect to continuous ROI geometry, and that straight-through
backpropagation is not the exact gradient of the deterministic hard selector.
This language is now corrected.

We do **not** freeze score-function/REINFORCE as the only final estimator
before evidence. It is an unbiased candidate for a stochastic hard-policy
expectation, but may have unacceptable variance and routing overhead for long
TAD windows. P0 must compare:

1. a dense relaxed pathwise warm-up;
2. a stochastic exact-K score-function candidate with expected-gradient and
   variance known-answer tests; and
3. a clearly labelled biased straight-through optimization surrogate.

The final algorithm is chosen by detector utility, stability, and charged
one-heavy-forward cost. It must never describe a selected estimator as a
direct exact hard-pathwise detector gradient when that statement is false.

### Numerical and systems choices

The review's `K=64`, 48-knot/16-frame cadence, 96-pixel scout, 6-by-8 target
lattice, five-dense-block schedule, 4,800 updates, CPU-pinned source gather,
dense scatter adapter, and fixed numerical latency/mAP thresholds are useful
P0 candidates or audit questions. They are not yet final model constants:

- User requirements favor fine continuous ROI control; the cadence must be
  evaluated at 2/4/8/16 source-frame strides rather than inferred from
  Uni-AdaFocus's distinct `16`-observation/`48`-segment terminology.
- `K`, scout resolution, and Dense-MoD locations need a result-blind geometry
  and cost calibration, then matched development evidence.
- CPU-gather and GPU-gather implementations must both charge synchronization,
  packing, and transfer cost. A CPU path may erase the promised saving.
- Dense-scatter is a parity/reference adapter candidate, not automatically a
  sparse final adapter if it recreates dense memory or compute.
- Thresholds may become pre-registered only after result-blind baseline
  variance and hardware profiling establish that they are meaningful.

### Kill semantics

Semantic violations (non-native resampling, broken two-frame tubelets,
two-heavy-forward execution, detector-contract failure, leakage, or failed
estimator known-answer tests) kill the corresponding claim. A first-pass
gather/adapter performance bottleneck is a `HOLD` and a focused implementation
pivot, not automatic evidence that structured spatial routing is scientifically
false. This distinction preserves the model-first research objective while
retaining strict evidence standards.

## Immediate revised P0 authorization

Only a small native-token, one-heavy-forward vertical slice is eligible next.
It must establish, without full training or A-MoD:

1. native `2 x 16 x 16` source-patch gather and `K=N` numerical parity;
2. fixed two-frame tubelet membership and absolute-coordinate semantics;
3. reconstruction of `[B, 384, 768]` and unchanged Focal/DIoU detector path;
4. mask/coordinate sparse-adapter parity at full support;
5. honest hard-policy estimator known-answer and variance tests; and
6. a component-level cost ledger for competing gather/packing paths.

No development mAP, full three-seed matrix, A-MoD block, official test, or
paper claim follows until this vertical slice is independently reviewed.

## Final position

We agree with the review's diagnosis and staging, but not with every frozen
implementation constant or its implicit assumption that one estimator and one
systems path should be selected before P0. The correct next step is the revised
P0 specification above, not a broader engineering project and not an immediate
full-scale experiment.
