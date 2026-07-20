---
type: experiment
node_id: exp:native-crop-s2-crop-sufficiency
title: "Continuous-RoI S2 crop sufficiency"
stage: tested
status: model_gate_passed_training_runtime_gate_pending
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

The project-authored v2.1 corrigendum is frozen at
`docs/methods/continuous_roi_s2_v2_1_contract.md` with machine-readable core
`docs/methods/continuous_roi_s2_v2_1_protocol.json`. Its protocol SHA-256 is
`ef806b7cd37c704d14a54211b1d4e2f9fb88b75599da918272cc6acad157b3af`.
The static validator passed eight contract families and all 128 outcome-state
assignments; audit SHA-256 is
`5af59b755dd4528fe3e4fd989bb20da71ee40e43ecb5add34083b8ae96057f9d`.
This authorized implementation only, not training or test opening.

## Implementation Evidence

- Commit `61878997adc4ca3d1de7396a804862d4c6943ee8` contains the
  selector-free D160/G96/U128 representation path. Clean remote focused tests
  passed `61`.
- Slurm Job `1177561` completed `0:0` and produced a self-hashed v2 full-model
  one-step Gate. It verified `[1,384,768]`, exact optimizer coverage, finite
  nonzero detector-only gradients through both U128 branches, fusion,
  projection and the AdaTAD-derived head, one shared VideoMAE parameter
  instance, two branch evaluations, and zero official-test access.
- The training runtime now binds the immutable fit-160/gate-40 development
  manifest, all nine family/seed configs, exactly 80 batches per epoch, 4,800
  successful updates, success-only scheduler/EMA advancement, and one
  final-EMA checkpoint. A real-data precheck opens one development training
  sample per family and revalidates the 200-video inventory.
- The integrated runtime authorization reopens every D160/G96/U128 Gate
  checkpoint and sidecar, recomputes their hashes, and revalidates EMA,
  optimizer, scheduler, config, Slurm, and work-directory bindings. The final
  independent read-only audit found no remaining P0/P1/P2 and returned
  `DEPLOY_READY_WITH_GATES`. Local pure-logic tests pass `36`; both Slurm
  launchers pass shell syntax checks.
- These are implementation and Gate facts only. No S2 checkpoint, development
  mAP, reference sweep, cost profile, mechanical outcome, or paper claim
  exists yet.

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

No Continuous-RoI S2 training result, official-test result, measured cost,
crop-sufficiency claim, learned selector, or paper claim exists. The
implemented S2 model and Job `1177561` validate model connectivity only.

## Next Gate

Commit and push the audited training runtime, replay the exact Linux suite in
a clean remote snapshot, and run one integrated Slurm Gate. The Gate must
produce a new full-model one-step certificate, a real-development-data
runtime-precheck certificate, three two-successful-update D160/G96/U128
checkpoints with final EMA sidecars, and an entity-revalidated runtime
authorization. Only that authorization may unlock the D160/G96/U128 x
three-seed development training matrix. Official test, learned ROI, S2
reference sweeps, cost claims, and paper claims remain blocked.
