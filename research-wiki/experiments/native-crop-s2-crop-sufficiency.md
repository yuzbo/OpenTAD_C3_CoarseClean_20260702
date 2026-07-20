---
type: experiment
node_id: exp:native-crop-s2-crop-sufficiency
title: "Native-Crop S2 crop sufficiency"
stage: designed
status: pro_protocol_reviewed_revision_required
outcome: pending
tags: ["offline-tad", "native-crop", "crop-sufficiency", "preregistration"]
added: 2026-07-20
---

# Native-Crop S2 Crop Sufficiency

## Purpose

Determine whether source-coordinate local crops can preserve registered TAD
accuracy and high-tIoU boundary behavior at lower measured representation-path
cost while retaining the full 768-point temporal axis. S2 evaluates crop
sufficiency before any learned crop selector is implemented.

## Reviewed Protocol

The Pro-authored v1 protocol freezes a 21-position static `128x128` crop
library over the current `320x180` decoded source geometry, a source-letterbox
D160 baseline, global96/local128 shared-VideoMAE paths, three seeds,
development fit/gate separation, official evaluator parity, full-stack cost,
immutable receipts, and a GT-visible detached candidate reference.

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

1. The GT-visible lexicographic reference is a heuristic, not a certified
   library or global-mAP upper bound. Its failure cannot kill the library.
2. Crop sufficiency, adaptive-selection headroom, and deployable cost viability
   need independent outcomes. A sufficient fixed crop is not a crop failure.
3. Gate GT artifacts may be created only after raw gate predictions are sealed
   in a separate privileged join stage.
4. Detection and cost require sampling-unit-correct uncertainty families,
   joined by an intersection decision rather than one mixed bootstrap.
5. Geometry coverage must be separated from model-conditioned candidate-union
   reachability.
6. The registered crop schedule changes the training distribution; the
   estimand must state this explicitly.
7. Any learned-policy authorization must reserve selector cost or limit its
   claim to representation-path headroom.
8. Numerical margins require a result-blind power/Monte-Carlo feasibility
   audit before freeze.

## Boundaries

No S2 model, training result, official-test result, measured cost, crop
sufficiency claim, library KILL, learned selector, or paper claim exists.
S1's CUDA gate validates implementation and gradients only.

## Next Gate

Revise v1 to v1.1 with corrected reference semantics, decision states, GT
artifact ordering, uncertainty families, coverage terminology, and selector
cost reserve. Freeze v1.1 only after a result-blind statistical feasibility
audit. Formal implementation and queueing remain blocked until then.
