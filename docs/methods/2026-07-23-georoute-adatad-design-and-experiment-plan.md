# GeoRoute-AdaTAD: Final Target and Experiment Plan

> Revision note (2026-07-23): an external implementation review was absorbed
> in `docs/methods/reviews/2026-07-23-georoute-adatad-v1-pro-review-absorption.md`.
> This remains a design document, not an implementation or empirical result.

**Date:** 2026-07-23
**Status:** `designed` -- no implementation, remote deployment, development mAP,
official-test result, cost result, or paper claim exists for this candidate.

## 1. Research Question

Can an offline TAD model reduce real spatial-and-depth compute while preserving
high-tIoU localization by learning a continuous, variable-size spatial support
from the real AdaTAD detection loss?

The question is not whether a manually selected crop can work. Historical S1
and Continuous-RoI S2 artifacts remain diagnostic/training-only evidence and
must not decide whether the learned policy proceeds.

## 2. Final Target Architecture

`GeoRoute-AdaTAD` is a single-heavy-backbone offline TAD model:

```text
full offline source video
  -> lightweight dense global scout
  -> continuous ROI knots: (cx, cy, w, h)
  -> ROI native-token score + residual free-token score + context allowance
  -> exact-K packed native 2 x 16 x 16 VideoMAE tubelets
  -> one shared sparse VideoMAE forward
  -> geometry- and mask-aware sparse temporal adapter
  -> original AdaTAD projection / ActionFormerHead / detector loss / NMS
```

The detector-facing temporal contract stays `[B, 384, 768]` with a matching
mask. The time axis is never sparsified: every detector time position receives
a spatial aggregate from selected native tubelets. The model is offline and may
use full-window scout context; it makes no online or causal claim.

### 2.1 Native continuous ROI

The policy emits bounded continuous `(cx, cy, w, h)` knots and linearly
interpolates them in source-image coordinates. The knot cadence is deliberately
not frozen from Uni-AdaFocus terminology: its `16` global observations and
`48` candidate segments do not mean one ROI per 16 source frames. GeoRoute P0
will support source-frame strides 2, 4, 8, and 16, with stride 4 the initial
high-resolution candidate and stride 16 only a compute control. ROI geometry
is scored against absolute centres of the existing native `2 x 16 x 16`
VideoMAE tubelet lattice.

This is deliberately not a fixed-size `grid_sample` crop. Integer crop
boundaries change a tensor's shape and selected indices discontinuously, so a
literal variable-shape tensor crop cannot provide exact derivatives with
respect to width, height, or centre. The model instead uses continuous geometry
to form a soft support over native tokens, then executes hard exact-K packing
at inference. No selected region is resized, and a tubelet keeps the same
spatial patch across its two source frames.

### 2.2 Hybrid spatial allocation

Each temporal tubelet receives a fixed total input budget:

```text
K_input = K_geo + K_residual + K_context
```

- `K_geo`: tokens favored by continuous ROI support.
- `K_residual`: free token choices driven by the scout, covering disjoint
  actor, object, and scene evidence outside one rectangle.
- `K_context`: a fixed global-context allowance that prevents total loss of
  scene evidence.

ROI is therefore a structured macro-prior, not a claim that one rectangle is
always sufficient. `K_input` is exact and shared with every matched baseline.

### 2.3 One-heavy-forward detector-policy estimator

An exact-K native gather has a discontinuous support set, so it has no ordinary
pathwise derivative with respect to a continuous ROI once membership changes.
GeoRoute must therefore not call a straight-through estimator the exact hard
policy gradient. The P0 vertical slice compares, without adding a second heavy
backbone forward:

1. **Soft-support warm-up:** one dense VideoMAE forward with continuous native
   support. Its detector-loss pathwise gradient is valid for the dense relaxed
   model, but this phase makes no sparse-cost claim.
2. **Hard-policy candidates:** an unbiased stochastic score-function estimator
   for an exact-K policy and a clearly labelled biased straight-through
   surrogate. The former must pass a small expected-gradient known-answer test
   and variance/overhead checks; the latter may remain only as a diagnostic or
   an empirically justified optimization surrogate.

The final estimator is a P0/P1 empirical decision, not a design-time decree.
Training and inference cost must be reported separately. A nonzero gradient
norm is insufficient: policy-enabled routing must be compared with a
stop-gradient policy control and must change routing and development metrics.

### 2.4 Sparse adapter and temporal aggregation

The current dense VideoMAE Adapter assumes a complete regular spatial grid and
cannot be silently reused for packed tokens. The new adapter must accept token
validity, absolute `(t, x, y)` coordinates, and ROI-relative geometry. Its
minimal role is to restore temporal coherence across changing supports before
spatial aggregation. Existing `deform_conv1d` is only a low-level reference:
it lacks spatial coordinates and mask semantics.

Original VideoMAE tubelet-projection weights, detector projection,
ActionFormerHead, current Focal classification loss, DIoU regression loss, and
NMS remain. The audited AdaTAD-derived configuration has no independent quality
head or quality loss. Because the backbone adapter changes, the system is
AdaTAD-derived rather than an unmodified official AdaTAD model.

### 2.5 Alternating Dense-MoD depth routing

A-MoD is a separate depth-compute mechanism. Its intended layout is:

```text
dense prefix -> Dense block -> MoD block -> Dense block -> MoD block -> ...
```

At every MoD block, capacity is derived from the immediately preceding **dense**
block's complete attention-derived importance. This is a Dense-MoD interval
design, not a consecutive all-MoD tail. It keeps the paper-level premise that
each router consumes complete prior attention. A-MoD uses previous-layer
attention column-mean importance, exact capacity, and identity bypass; routing
scores are not multiplied into output features.

The real VideoMAE attention uses SDPA and does not expose maps. The deployed
path must preserve fused attention; a materialized attention matrix is allowed
only in tiny numerical known-answer tests. No author official A-MoD
implementation was verified in this audit, so the local port must be called a
paper-exact reproduction.

## 3. Reference Alignment

| Reference | Reuse faithfully | Deliberate offline-TAD adaptation |
| --- | --- | --- |
| OpenTAD/AdaTAD | data split, source-window contract, VideoMAE initialization, `[B,384,768]`, projection, ActionFormerHead, detector losses, NMS, evaluator | replace only backbone-side spatial/depth computation |
| Uni-AdaFocus | global observer, bounded continuous geometry, temporal policy interpolation, anti-collapse regularization | no fixed-output crop resize, classification proxy, detached local action, temporal segment dropping, or second heavy backbone |
| A-MoD | previous-attention column mean, exact capacity, identity bypass | alternating Dense-MoD blocks, VideoMAE-compatible score path, paper-exact rather than official-code claim |
| ToMe / TokenSelect / MoD | matched token/depth controls | shared source video, detector, update count, K budget, and cost ledger |

## 4. Non-Negotiable Matched Settings

1. Same fit/gate/test split, annotations, source videos, decode order, temporal
   windowing, padding, augmentation, and coordinate mapping.
2. Same VideoMAE checkpoint, AdaTAD projection/head/loss/NMS, optimizer,
   successful-update schedule, AMP, batch size, accumulation, update count,
   EMA policy, and seeds.
3. Same full 768-position detector axis and mask semantics.
4. Same low-resolution scout protocol and charged scout cost for every learned
   policy variant; the heavy route remains native source geometry.
5. Same exact input-token budget, per-tubelet minimum, context allowance, and
   Dense-MoD capacity at a given budget.
6. Gate-only checkpoint selection followed by one sealed official-test opening.
   No validation/test GT, teacher, oracle, manual ROI, raw prediction cache, or
   privileged crop can enter inference.
7. Same full cost scope: decode, preprocessing, H2D, scout, router, gather,
   patch embedding, backbone, adapter, detector, NMS, p50/p95, memory, energy.

## 5. Required Baselines and Ablations

### Primary matched matrix

1. Dense AdaTAD-derived baseline.
2. Random exact-K and uniform spatial exact-K controls.
3. Free native TokenSelect-only.
4. ROI-only native selection.
5. ROI plus residual TokenSelect -- spatial main candidate.
6. ToMe-only internal token-merging baseline.
7. A-MoD-only alternating Dense-MoD baseline.
8. ROI plus residual TokenSelect plus alternating A-MoD -- full candidate.

### Mechanism ablations

- stop detector gradient to ROI/residual policy;
- no residual allocation;
- no absolute coordinates and ROI-relative-only coordinates;
- no geometry regularization;
- ROI knot stride 2/4/8;
- alternating Dense-MoD versus dense-only, random exact-K, and linear MoD;
- matched token/depth budget curve.

ToMe is a useful baseline but not an equivalent pre-backbone crop method:
it normally merges after patch embedding, so total end-to-end cost and internal
token FLOPs must be reported separately.

## 6. Staged Implementation and Deployment

### P0: parallel implementation gates, no mAP claim

- Native gather/projection parity at `K=N`.
- Detector parity at `[B,384,768]`.
- Continuous ROI/residual gradient connectivity, score-function known-answer
  test, variance accounting, and a stop-gradient control.
- Packed sparse-adapter mask/coordinate tests.
- A-MoD numerical KAT, `C=1` parity, and fused-attention preservation.
- Component cost trace availability.

### P1: one-seed development screening

Run dense, free TokenSelect, ROI-only, and ROI-plus-residual under one matched
budget. Do not attach A-MoD unless its P0 gate passes. If free TokenSelect
dominates structured routing at high tIoU and end-to-end cost, ROI cannot remain
the primary paper claim.

### P2: three-seed main matrix

Promote only P1 survivors. Add alternating Dense-MoD after its independent P0
gate. Report Avg-mAP, mAP@0.3--0.7, especially 0.6/0.7, short-action and
boundary diagnostics, exact allocations, spatial-policy stability, and total
cost.

### P3: paper ablations and generalization

Complete mechanism ablations, budget curve, and a second detector or dataset
only after a positive AdaTAD-derived main result. Keep official test sealed
until the development decision is frozen.

## 7. Decision Rules

- A policy does not pass because its gradient exists, its ROI looks plausible,
  or selected-token count is small.
- ROI remains primary only if it improves the high-tIoU/cost Pareto over free
  TokenSelect and matched unstructured controls.
- A-MoD remains only if alternating Dense-MoD lowers measured cost without
  erasing the spatial model's localization advantage.
- No manual or GT-privileged crop result may authorize, reject, or tune the
  learned policy.

## 8. Evidence Status

The previous Continuous-RoI S2 exact-nine receipt is `PASS_TRAINING_ONLY`.
It supplies no development mAP, crop-sufficiency, policy-learning, cost,
official-test, or paper evidence for GeoRoute-AdaTAD.
