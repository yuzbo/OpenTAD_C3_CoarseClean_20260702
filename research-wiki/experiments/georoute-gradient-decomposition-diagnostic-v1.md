---
type: experiment
node_id: exp:georoute-gradient-decomposition-diagnostic-v1
title: "GeoRoute PL/ST matched gradient decomposition diagnostic v1"
idea: idea:geo-route-adatad
stage: experiment_running
status: two_arm_diagnostic_running
verdict: ACCEPT_WITH_IMPLEMENTATION_CORRECTIONS
confidence: high
updated: 2026-07-30
---

# GeoRoute PL/ST matched gradient decomposition diagnostic v1

## Question

Does the first nonfinite value in the sealed stability-v2 trajectory arise in
the ordered-PL score, the residual-logit gradient, the scout VJP, a shared
detector gradient, or only when the production DDP hook casts a finite scaled
FP32 bucket to FP16?

## Why this is next

Stability-v2 is sealed HOLD and cannot be rerun or reinterpreted. Its telemetry
shows finite forward losses and scaler skips, but observes parameters only after
DDP bucket communication. It cannot distinguish an intrinsically nonfinite
gradient from a finite scaled gradient that overflows during FP16
communication. Repairing PL now would be post-hoc mechanism guessing.

The user-provided Pro review, file SHA-256
`22f5802f62689f687667f56ddd6aacb35e07242c213a591cf93a4e50942c6e83`,
therefore recommended `NEW_MATCHED_DIAGNOSIS_BEFORE_REPAIR`. The project accepts
that verdict with the implementation corrections recorded in
`docs/methods/reviews/2026-07-30-georoute-gradient-decomposition-pro-absorption.md`.

## Frozen design

- study/profile:
  `georoute_pl_gradient_decomposition_diagnostic_v1` /
  `pl_gradient_decomposition_v1`;
- mechanically derived independent seed `7367`;
- matched `residual_pl_rep_off` and `residual_st_rep_off`;
- 64 consumed batches, `B/T/N/K=1/384/220/64`;
- temperature `0.7`, weight `1.0`, baseline momentum `0.95`, temporal mean;
- official-semantics default GradScaler, zero retry/replay, scheduler/EMA per
  consumed batch;
- production FP16 DDP hook remains authoritative;
- separate one-GPU CUDA/DDP KAT before the three-job DAG;
- no checkpoint, prediction, metric, evaluator, NMS, cost, or official test.

PL consumes CUDA RNG for Gumbel sampling while ST does not. The corrected
matchedness gate therefore requires all-batch data and CPU RNG equality plus
batch-zero CUDA RNG equality, while recording expected later CUDA divergence.

Full specification:
`docs/superpowers/specs/2026-07-30-georoute-gradient-decomposition-diagnostic-v1-design.md`.

## Implementation status

The implementation adds:

- an opt-in transient wrapper payload without changing sampling, likelihood,
  policy loss, route, or estimator;
- an event-connected observer for forward, backward, unscale, clipping, scaler,
  scheduler, EMA, and DDP buckets;
- an exact detached shadow of PyTorch's cast-then-divide FP16 hook order while
  returning the real standard hook Future;
- analytic ordered-PL score, expected scaled gradient, actual residual-logit
  hook, and FP64 subset reference;
- compact grouped statistics only, with no raw tensor serialization;
- self-hashed binding, KAT, receipt, stage, deployment, and finalization
  contracts;
- held parallel leaves, `afterany` finalizer, no-resume namespace, storage and
  origin-ref gates.

Exact source `664180b6e2645aa3f9bde8b3a67fc7c224b3915c` passed the clean
N16R4 Linux/Torch suite `161/161`. CUDA/DDP KAT Job `1207467` completed `0:0`
and proved that the observer leaves the authoritative bucket unchanged, the
standard hook Future completes, a finite FP32 value can become nonfinite at the
FP16 cast, and the analytic ordered-PL gradient agrees with autograd.

The first DAG admission attempt then failed before namespace creation or
`sbatch`: the deployer read the parent stability-v2 field as
`failed_batch_indices`, while the sealed v2 schema correctly calls it
`skipped_batch_indices`. The correction is restricted to parent-provenance
serialization, validates a sorted unique nonnegative list, and does not alter
the model, optimizer, observer, frozen protocol, or parent evidence. A new exact
source, full remote suite, and same-commit CUDA KAT are required before
deployment.

The corrected exact source
`33f721be83e0ad7f7a36e853491e7a14f148814b` passed the clean remote
Linux/Torch suite `162/162`; its full HEAD and origin ref match and its tree is
clean. Same-commit CUDA/DDP KAT Job `1207480` completed `0:0` with receipt
self/file SHA-256
`d31d34144e60bdde6103acc36cff42301ba7fbd80a40eb6f04ead63ddb6901b4`
/
`09e2ed0ec6f6e3372871ea00f0aa610027bbedd81d098b60ea6a2529aed0e6f4`.

The fresh no-resume diagnostic root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_pl_gradient_decomposition_v1_33f721be_s7367_20260730_2300`.
PL/ST Jobs `1207484/1207485` were released simultaneously and are running on
separate Slurm nodes; afterany finalizer `1207486` remains dependency-held.
Deployment self/file SHA-256 are
`9e038d872f7869cd184a1835827c61c2c3527fa565bc7928ef17824f905297a2`
/
`12e785c78f1fdff3d39bd83222836548f3d803990cd1c62862c07ecbcd7ecc66`.
Initial stderr files are empty, no performance or checkpoint artifacts exist,
and the storage/capacity gates passed. This raises the experiment only to
`experiment_running`.

## Decision boundary

The finalizer can identify one repair class, HOLD, or declare the diagnosis
incomplete. A unique class authorizes only one class-specific minimal repair and
a new no-performance gate. It never authorizes a performance experiment,
official test, estimator winner, Geometry Zoom, P2/P3, or paper claim.

## Connections

Maintained only in `research-wiki/graph/edges.jsonl`.
