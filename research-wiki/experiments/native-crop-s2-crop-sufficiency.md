---
type: experiment
node_id: exp:native-crop-s2-crop-sufficiency
title: "Continuous-RoI S2 crop sufficiency"
stage: designed
status: pro_v2_received_hold_for_p0_revision
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

## Reviewed Protocols

The Pro-authored v1 protocol froze a 21-position static `128x128` crop library.
That protocol is now superseded as the decisive S2 object because the final
method must regress variable center and variable width/height in the
Uni-AdaFocus deformable-patch sense. The 21 candidates may remain a D0
discrete baseline, but cannot establish or reject continuous-RoI sufficiency.

The raw protocol is archived at
`docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-raw.txt`
with SHA-256
`e14abfab41fafa3c3f411df87d3148170872a190c274ed9b7eb2dd44c520c7d5`.

The Pro-authored Continuous-RoI v2 is archived at
`docs/methods/reviews/2026-07-20-continuous-roi-s2-v2-preregistration-pro-raw.txt`
with SHA-256
`9adbd388ad41f79e9323612c25be493332127b226eb2aa968832d14c5446582b`.
Its author verdict is `V2_READY`; the project verdict is
`ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`. The complete absorption is
at
`docs/methods/reviews/2026-07-20-continuous-roi-s2-v2-preregistration-pro-absorption.md`.

## Accepted

- Use continuous `(cx,cy,w,h)` source boxes with analytic in-bounds geometry.
- Use a temporal ROI tube and matched differentiable/runtime bilinear sampling.
- Keep a fixed local output tensor while allowing source box size/aspect to vary.
- Crop source pixels before resize and preserve the complete time axis.
- Rebuild the D160 comparator in the same S2 runtime.
- Use one shared VideoMAE parameter instance while accounting for two forward
  computations.
- Keep gate raw inference free of GT, teacher, target cache, and reference IDs.
- Keep official test sealed and use immutable, fail-closed evidence.
- Measure latency, memory, and gross GPU energy; disclose exhaustive reference
  search separately.
- Treat the finite candidate library as library-conditional evidence only.
- Separate geometry, search adequacy, representation sufficiency, adaptive
  headroom, and cost viability.

## Current Blocking Revisions

1. Do not train a deployable learned ROI policy inside S2 while claiming S2 is
   a pre-policy representation gate; keep S2/S3 separate.
2. Do not derive fixed, random, D0, or location-only controls by overriding an
   adaptive GL checkpoint only at inference. Train decision-critical geometry
   families under matched registered distributions.
3. Do not compare GT-privileged `CR-PREF` against unprivileged `C/R/LC`.
   Variable-size and fixed-size references require the same search and join.
4. Confidence-objective convergence is not spatial-reference adequacy.
   Separate no-GT policy diagnostics from result-independent geometry/support.
5. Repair the formal queue for the audited N16R4 memory policy, actual storage,
   validated 20/100 ms power sampler, and immutable failed namespaces.
6. Remove selector-cost double counting, fix ABBA population arithmetic, use
   independent variance evidence, specify privileged matching completely, and
   add a sufficient-but-no-adaptive-headroom state.

## Boundaries

No continuous-RoI S2 model, training result, official-test result, measured
cost, crop-sufficiency claim, learned selector, or paper claim exists. S1's
fixed-center CUDA gate validates infrastructure and gradients only.

## Next Gate

Freeze a narrow Continuous-RoI S2 v2.1 corrigendum that repairs the S2/S3
boundary, training-matched controls, privilege-matched references, search
adequacy, queue/storage/power contracts, and outcome state machine. No model
implementation or queueing is authorized before the corrected protocol and
its static feasibility validator pass.
