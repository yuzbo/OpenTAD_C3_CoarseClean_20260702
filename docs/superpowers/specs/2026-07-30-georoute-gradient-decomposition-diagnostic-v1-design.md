# GeoRoute PL gradient-decomposition diagnostic v1

## Objective

Determine where the first nonfinite value appears during the frozen
stability-v2 residual-PL/ST execution path, without changing the model,
estimator, optimizer transition, communication hook, data, or any performance
surface.

This is a diagnostic, not a stability PASS and not a model experiment.

## Identity

```text
study_id = georoute_pl_gradient_decomposition_diagnostic_v1
profile = pl_gradient_decomposition_v1
seed = 7367
seed derivation = 1000 + int(SHA256(study_id)[0:8], 16) mod 9000
batches = 64
```

Seed `7367` is disjoint from stability-v2 seed `4417`, pilot seed `3407`, and
reserved paper seeds `3407/3408/3409`.

## Parent and source gates

Deployment requires:

- exact sealed stability-v2 finalization file SHA-256
  `c7f59dbcec609430bdf4aafe99cc5ef3272ef93362b7f44ba74bcbc337c85ab0`;
- exact stability-v2 PL receipt file SHA-256
  `6667c8f0f357984fc19dd18d9bbe466241749dc77b6d5ce434b2e106ef7da277`;
- exact stability-v2 ST receipt file SHA-256
  `cdc3947bb42e63799a5c8591274d54d508c3bdbda99ecbea8d05eb7df01c555d`;
- transitive pilot, matched-diagnosis, stability-v1, and official-reference
  evidence from the sealed v2 deployment;
- exact current full runtime commit, matching origin-tracking ref, and clean
  tree;
- exact hashes for source config, official reference, manifest, annotation,
  class map, pretrained checkpoint, and exact development video-root path;
- a passing, self-hashed CUDA/DDP observer KAT from the same runtime commit.

Every GitHub operation uses the academic proxy from its first request.

## Arms

| Arm | Estimator | Representation | Route |
| --- | --- | --- | --- |
| `residual_pl_rep_off` | ordered score function | off | free residual |
| `residual_st_rep_off` | straight-through | off | free residual |

Both arms freeze:

```text
B=1, T=384, N=220, K=64
temperature=0.7
score_function_weight=1.0
baseline_momentum=0.95
temporal_reduction=mean
pooling=uniform_selected
geometry_side_channel=false
fp16_compress=true
default GradScaler=true
zero retry, zero replay
scheduler and EMA advance once per consumed batch
```

No AMP skip is retried. A scaler-skipped optimizer attempt is consumed and
observed using the same transition cadence as the intended official semantics.

## Matchedness

Required:

- identical immutable inputs and parent chain;
- identical 64-batch data-fingerprint sequence;
- identical 64-batch CPU RNG fingerprint sequence;
- identical CUDA RNG fingerprint at batch 0;
- exactly 64 consumed batches per arm;
- no retry or replay.

PL consumes CUDA RNG for Gumbel sampling and ST does not. Post-batch-zero CUDA
RNG sequence equality is therefore not required. Both sequences are retained so
the divergence remains explicit.

## CUDA/DDP KAT

The KAT runs before the DAG and produces no model artifact. In a one-process
NCCL DDP backward it must prove:

1. the detached observer does not alter the original bucket before the
   authoritative hook;
2. the wrapper returns a Future accepted by DDP;
3. a finite FP32 value outside FP16 range is detected as a cast-introduced
   nonfinite;
4. the analytic ordered-PL gradient and autograd gradient agree and have
   positive dot product;
5. the receipt self-hash and no-performance guards pass.

The two-arm deployer refuses a missing, failed, changed, or different-commit KAT.

## Telemetry

For every consumed batch:

- data descriptor and content fingerprint;
- CPU and CUDA RNG fingerprints at batch start;
- iteration, successful-update index, retry count, and scale;
- detector losses/cost, baseline, advantage, policy loss, and total cost;
- production route metadata and `(B,T,N,K,tau,lambda,S)`;
- analytic ordered PL log probability and score;
- per-slot chosen-probability, remaining entropy, slot score, and cumulative
  score statistics;
- FP64 score reference on preregistered tubelets `0/127/255/383`;
- expected scaled residual-logit gradient;
- actual residual-logit hook gradient, difference, cosine, and sign;
- DDP bucket index, parameter names/groups, FP32 pre-hook statistics,
  hypothetical unscaled statistics, and detached FP16 cast statistics;
- scaled, unscaled, pre-clip, post-clip, and scaler-result grouped gradient
  snapshots;
- scheduler/EMA/optimizer attempt counts.

Only aggregate statistics and parameter names may be serialized. Raw tensors
are forbidden.

## Mechanism classification

Per failed PL attempt:

- `DDP_FP16_CAST_OVERFLOW`: expected and actual logit gradients finite, FP32
  bucket finite, detached FP16 cast introduces the first nonfinite;
- `UPSTREAM_SCORE_NONFINITE`: analytic expected or actual residual-logit
  gradient is nonfinite;
- `SCOUT_VJP_NONFINITE`: logit gradients are finite but a scout bucket is
  already nonfinite before the FP16 cast;
- `SHARED_DETECTOR_BUCKET_OVERFLOW`: detector-only cast overflow is reproduced
  by ST on the matched batch;
- `AMBIGUOUS_MIXED_FAILURE`: evidence does not uniquely satisfy one class.

`REPAIR_CLASS_IDENTIFIED` requires complete matched telemetry, finite forward
losses, at least one PL failed attempt, one unique non-ambiguous mechanism class,
and consistent analytic/actual PL gradient direction. No PL failure, mixed
classes, missing telemetry, or matchedness failure yields HOLD. Missing/invalid
artifacts or execution failure yields INCOMPLETE.

## DAG

```text
independent CUDA/DDP KAT (must PASS)
                  |
                  +-- residual_pl_rep_off, seed 7367, 1 GPU
                  +-- residual_st_rep_off, seed 7367, 1 GPU

afterany(PL, ST) ---- finalizer, site-compliant GPU allocation
```

The two leaves are submitted held, immutable deployment/submission receipts are
written, and both leaves are released together. The finalizer always runs after
both terminal states. There is no resume, replacement arm, or partial
supplementation.

## Artifacts

Allowed:

- bound config, deployment/submission/release receipts;
- KAT receipt;
- train/Slurm logs;
- gradient-decomposition receipt;
- stage result and finalization.

Forbidden:

- checkpoints (`*.pth`, `*.pt`, `*.ckpt`);
- raw tensor dumps;
- predictions, metrics, evaluator/NMS output, official-test output;
- latency, energy, or performance profiles;
- namespace larger than 2 GiB.

At least 10 GiB free storage is required before deployment.

## Authorization boundary

A unique mechanism class authorizes only a separately preregistered minimal
repair for that class and a fresh no-performance gate on a new source, study ID,
namespace, and independent seed.

HOLD or INCOMPLETE authorizes no objective/estimator change and no performance
run. Even a unique class does not authorize PL/ST ranking, paper seeds, formal
accuracy/cost experiments, official test, Geometry Zoom, P2/P3, efficiency
claims, or paper readiness.
