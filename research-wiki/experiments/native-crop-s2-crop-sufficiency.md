---
type: experiment
node_id: exp:native-crop-s2-crop-sufficiency
title: "Continuous-RoI S2 crop sufficiency"
stage: experiment_running
status: formal_3x3_training_complete_reference_protocol_hold
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
- Runtime commit `6ee8a83` passed its integrated Gate and authorized a first
  formal matrix, but Jobs `1177641-1177646` failed during launcher preflight
  and Jobs `1177647-1177649` were cancelled before allocation. The exact root
  cause was a Windows carriage return appended to `YUZIBO_ROOT`, producing the
  nonexistent path
  `/data/run01/sczc063/yuzibo^M/conda_envs/opentad/bin/activate`. Campaign
  `66cd32ffecf22b5868997bb9f73e8c20befd3a3a668ba96a59901d18a16b43da`
  is immutable deployment-failure evidence and contributes no model result.
- Commits `eea1f906b035dceaa0cb8c17a8271cf5385ca791` and
  `9a61da27e65c2227c8d2a0c547d8f3cb44966738` make the deployment contract
  reject ASCII control characters, whitespace and commas in Slurm export
  values; require the exact canonical YUZIBO root; bind the launcher to the
  expected Git blob; recheck its hash immediately before every `sbatch`; and
  bind the normalized environment into v2 intent, receipt and summary
  evidence. The clean Linux exact suite passed `81`.
- Integrated Gate Job `1177662` completed `0:0`. It reran all `81` exact
  tests, produced full-model Gate SHA
  `c633c3b7bc824c2f65800498621904fe307bc9be674375a3849c8c4e815f8c73`,
  runtime-precheck SHA
  `6c5b8f52fe99c2b865ebf8d58db60b579e62c6258a93f6c7276020a7eb077272`,
  and runtime-authorization SHA
  `62a0fb21809f9b297337fd17d8440c8c557bfca00ab609c934832abc25846a5f`.
  The Gate verified two successful real-data updates for every family,
  nonempty optimizer/scheduler/EMA state, deterministic D160/G96 upsampling,
  U128 auxiliary losses, exact one-GPU Slurm identity, and zero official-test
  access.

## Formal Development Matrix

The sole formal campaign is
`77c2149aa9fe7d6b19c777bebe5a95de710a0a738d89158431e69a9c5e78d066`
under base experiment namespace
`469652821b1ffb984ced17360a333cf1fed2700465ec6f8ac210b578e7dc5de9`.
Deployment intent SHA is
`96a805cafc2f918f8f24518d86744666e2ec39ef77981b954e36ef6db047a264`
and deployment SHA is
`67227c4478dc5bf10ac3fc613aededd603e34bb670bc159a2863f0a69594c204`.

The registered jobs are:

- D160: `1177668/1177669/1177670` for seeds `3407/3408/3409`.
- G96: `1177671/1177672/1177673` for seeds `3407/3408/3409`.
- U128: `1177674/1177675/1177676` for seeds `3407/3408/3409`.

At 2026-07-21 14:25 CST, all nine jobs were `COMPLETED 0:0`. A strict
post-training replay loaded every raw and EMA state from disk and verified the
checkpoint, metadata sidecar, bound config, protocol, campaign and completion
hashes. Every cell has exactly 60 epochs, 80 successful updates per epoch,
4,800 successful updates and a final-EMA-only checkpoint. The bound deployment
and completion contracts prohibit official-test use, and no official-test job,
result, or evidence artifact exists. Historical training did not instrument
syscall-level file access, so this is not a runtime zero-open audit. No
`Traceback`, OOM, non-finite loss, exhausted retry, scheduler/EMA/update-parity
failure, `PytorchStreamWriter`, or failure marker was found. The raw final
epoch-59 losses were:

| Family | Seed | Final loss | AMP skipped attempts | Max retries/batch | Logged GPU memory |
|---|---:|---:|---:|---:|---:|
| D160 | 3407 | 0.2190 | 4 | 2 | 3919 MB |
| D160 | 3408 | 0.2172 | 4 | 1 | 3919 MB |
| D160 | 3409 | 0.2115 | 3 | 1 | 3919 MB |
| G96 | 3407 | 0.2259 | 3 | 2 | 2153 MB |
| G96 | 3408 | 0.2184 | 4 | 2 | 2153 MB |
| G96 | 3409 | 0.2219 | 3 | 1 | 2153 MB |
| U128 | 3407 | 0.2517 | 3 | 1 | 3647 MB |
| U128 | 3408 | 0.2483 | 3 | 1 | 3647 MB |
| U128 | 3409 | 0.2404 | 3 | 1 | 3647 MB |

The saved scheduler states close at successful update `4800`, with the matched
inherited cosine horizon still at `8000` updates and warmup at `400`. Thus the
registered 60-epoch runs are matched truncations of the inherited 100-epoch
schedule, not completed cosine cycles. This is not an evidence-integrity
failure, but convergence claims must retain this limitation.

The exact-nine evidence finalizer is being added as a training-only receipt.
It revalidates live artifacts and Slurm accounting and explicitly records
`reference_sweep_completed=false`, `crop_sufficiency_established=false`, and
`paper_claim_allowed=false`. This remains `experiment_running`: no
development mAP, reference sweep, cost profile, mechanical outcome, official
test result, or paper claim exists.

Three read-only evidence-finalizer review rounds are closed. The final round
returned `NO_P0_P1` after the implementation rejected executable configs,
forced live Slurm accounting, strict-loaded raw/EMA state into real model
interfaces, rejected duplicate/orphan optimizer state, enabled restricted
`weights_only` checkpoint loading, and recursively bound deployment intent,
receipts, configs, checkpoint metadata and tracked validator bytes. Residual
P2 is the missing real D160/G96/U128 Linux integration run; local small-model
tests cannot substitute for the formal training-only finalizer Job.

## Post-Training Reference Audit

The v2.1 reference phase is not executable without changing scientific
meaning. The audit found these protocol-level blockers:

1. The decisive claim says FS and VS share physical center trajectories and
   differ only in area/aspect. The protocol instead shares `sx,sy`, while the
   decoder computes `cx=0.5*w+(1-w)*sigmoid(sx)` and the analogous `cy`.
   Because VS changes `w,h`, equal logits do not imply equal physical centers.
   The current contrast therefore confounds center and scale/aspect.
2. The exact Owen-scrambled Sobol engine, dtype, transform serialization,
   stable-hash byte encoding and known-answer hash are not machine-frozen,
   despite the required exact generator hash.
3. The raw phase forbids a reference ID in the Python object graph while also
   requiring enumerated candidate IDs to be sealed. It must distinguish a
   result-blind enumerated candidate ID from a GT-selected preferred ID.
4. The annotation-free 129-window raw manifest/entrypoint, typed raw sharding,
   privileged CPU join, D0 coordinates/order, tie handling, Short-Q1 and
   bootstrap/max-T details are not fully machine-defined or implemented.

No reference job may be submitted by guessing these definitions. This is a
protocol hold, not a negative result for Continuous-RoI and not permission to
open official test or implement S3.

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

## Remaining Decision Gates

1. Seal the validated exact-nine training-only completion receipt without
   assigning any crop-sufficiency meaning.
2. Correct and re-freeze the FS/VS physical-center contract, exact generator
   identity, raw candidate-ID rule, privileged join and statistical details.
3. Development-only checkpoint selection and the matched fixed/variable
   reference sweep must remain result-blind with equal privilege. Confidence
   convergence alone is not spatial-reference adequacy.
4. Representation sufficiency, adaptive headroom, and deployable cost
   viability remain separate outcomes. S2 must not train or claim the S3
   learned ROI policy.
5. Latency, memory and energy profiling must use trained checkpoints and the
   frozen cost protocol. Selector reserve and ROI-head cost must not be counted
   twice.
6. Official test remains sealed until the complete development decision and
   its immutable evidence pass the v2.1 contract.

## Boundaries

The first valid Continuous-RoI S2 exact-nine training matrix is complete, but
no development detection result, official-test result, measured cost,
crop-sufficiency claim, learned selector, or paper claim exists. Training
completion proves optimization/exposure integrity only.

## Next Gate

Seal the exact-nine training-only receipt, then issue a minimal v2.2 protocol
corrigendum that resolves the four reference blockers before implementing or
queueing development raw inference. Do not open official test, implement
learned ROI, or make crop-sufficiency/cost/paper claims.
