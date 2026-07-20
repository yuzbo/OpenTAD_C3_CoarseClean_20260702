---
type: experiment
node_id: exp:native-crop-s2-crop-sufficiency
title: "Continuous-RoI S2 crop sufficiency"
stage: designed
status: fixed_library_protocol_superseded_continuous_roi_redesign_required
outcome: pending
tags: ["offline-tad", "continuous-roi", "crop-sufficiency", "preregistration"]
added: 2026-07-20
---

# Continuous-RoI S2 Crop Sufficiency

## Purpose

Determine whether continuously parameterized source-coordinate boxes
`(cx,cy,w,h)` can preserve registered TAD accuracy and high-tIoU boundary
behavior at lower measured representation-path cost while retaining the full
768-point temporal axis. The center, width, height, scale, and aspect ratio are
not fixed. S2 evaluates deformable crop sufficiency before a deployable learned
policy is claimed.

## Reviewed Protocol

The Pro-authored v1 protocol froze a 21-position static `128x128` crop library.
That protocol is now superseded as the decisive S2 object because the final
method must regress variable center and variable width/height in the
Uni-AdaFocus deformable-patch sense. The 21 candidates may remain a D0
discrete baseline, but cannot establish or reject continuous-RoI sufficiency.

The raw protocol is archived at
`docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-raw.txt`
with SHA-256
`e14abfab41fafa3c3f411df87d3148170872a190c274ed9b7eb2dd44c520c7d5`.

## Accepted

- Crop source pixels before resize and preserve the complete time axis.
- Rebuild the D160 comparator in the same S2 runtime.
- Use one shared VideoMAE parameter instance while accounting for two forward
  computations.
- Keep gate raw inference free of GT, teacher, target cache, and reference IDs.
- Keep official test sealed and use immutable, fail-closed evidence.
- Measure latency, memory, and gross GPU energy; disclose exhaustive reference
  search separately.
- Treat the finite candidate library as library-conditional evidence only.

## Blocking Revisions

1. Replace the fixed-library estimand with continuous normalized
   `(cx,cy,w,h)` geometry, explicit in-bounds parameterization, variable
   scale/aspect constraints, and a matched differentiable crop evaluator.
2. The GT-visible reference remains a heuristic, not a certified
   library or global-mAP upper bound. Its failure cannot kill the library.
3. Crop sufficiency, adaptive-selection headroom, and deployable cost viability
   need independent outcomes. A sufficient fixed crop is not a crop failure.
4. Gate GT artifacts may be created only after raw gate predictions are sealed
   in a separate privileged join stage.
5. Detection and cost require sampling-unit-correct uncertainty families,
   joined by an intersection decision rather than one mixed bootstrap.
6. Continuous-domain coverage/search quality must be separated from
   model-conditioned reachability; a finite restart/search procedure is not a
   certified continuous oracle.
7. The registered crop distribution changes the training distribution; the
   estimand must state this explicitly.
8. Any learned-policy authorization must reserve selector cost or limit its
   claim to representation-path headroom.
9. Numerical margins require a result-blind power/Monte-Carlo feasibility
   audit before freeze.

## Boundaries

No continuous-RoI S2 model, training result, official-test result, measured
cost, crop-sufficiency claim, learned selector, or paper claim exists. S1's
fixed-center CUDA gate validates infrastructure and gradients only.

## Next Gate

Write a new continuous-RoI S2 v2 protocol. It must freeze variable-box
parameterization, differentiable training crop, source-coordinate inference
crop, size-collapse prevention, temporal grouping/smoothing, reference search,
matched baselines, decision states, uncertainty families, and selector-cost
reserve. The prior fixed-library v1/v1.1 must not be implemented as the final
S2 gate. The implementation-targeted Pro review prompt is frozen at
`docs/methods/reviews/2026-07-20-continuous-roi-s2-v2-preregistration-pro-prompt.md`;
it is bound to immutable code/research-state commit
`6118cd50a3601d044dab690427ad9c756ce7d827`.
