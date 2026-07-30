# GeoRoute gradient-decomposition Pro review absorption

Date: 2026-07-30

## Source identity

The user-provided review is:

- local attachment:
  `C:\Users\skywalker\.codex\attachments\c4c0c744-e2cf-435d-8825-5d4437635410\pasted-text.txt`;
- file SHA-256:
  `22f5802f62689f687667f56ddd6aacb35e07242c213a591cf93a4e50942c6e83`;
- executive verdict:
  `NEW_MATCHED_DIAGNOSIS_BEFORE_REPAIR`.

The review is design and code-audit evidence. It is not a numerical,
performance, official-test, or paper result.

## Project verdict

`ACCEPT_WITH_IMPLEMENTATION_CORRECTIONS`

The central verdict is accepted. The stability-v2 HOLD does not identify whether
the first nonfinite value exists in the FP32 scaled gradient before communication
or is introduced when the DDP hook casts a finite scaled bucket to FP16. No
estimator, objective, temperature, token budget, baseline, official config, or
old receipt may be changed before that mechanism is decomposed.

The following actions are accepted:

1. add a strict opt-in transient probe in the GeoRoute wrapper;
2. observe the production residual-logit gradient and DDP GradBuckets;
3. call the authoritative PyTorch FP16 compression hook unchanged and return its
   exact Future;
4. run a fresh seed-7367 matched PL/ST, 64-consumed-batch, no-performance
   diagnosis;
5. use an `afterany` finalizer and fail closed on missing or invalid evidence;
6. allow the terminal diagnosis to authorize at most one mechanism-specific
   numerical repair followed by another no-performance gate;
7. retain the two-anchor, three-seed, sealed-test publication plan for later.

## Code-grounded findings accepted

- Both stability-v2 arms used the DDP FP16 compression hook during
  `GradScaler.scale(loss).backward()`.
- The hook receives scaled bucket contents before `unscale_()` and clipping.
- The PyTorch 2.0.1 implementation casts the bucket to FP16 before division by
  world size, so a finite FP32 scaled value above the FP16 range can become
  nonfinite.
- Residual-PL's detector cost is detached in the score-function term. Its extra
  policy gradient reaches the scout/residual-logit path, while ST uses a
  hard-forward/soft-backward pathwise surrogate.
- Stability-v2 used temperature `0.7`, `K=64`, temporal mean, weight `1.0`, and
  baseline momentum `0.95`.
- PL's third backoff to `8192` is compatible with dynamic-scaler adaptation but
  violates the already frozen v2 skip and scale-floor rules. The sealed HOLD
  cannot be relabelled.
- ST's arm-only numerical PASS is not estimator-quality or performance evidence.

## Implementation corrections

The review is not accepted verbatim in four implementation details.

### Observer integration

The illustrative observer code was not connected to the existing train-engine
event stream. The implementation connects one observer instance to:

- batch start and bound RNG/data descriptors;
- forward completion;
- scaled backward;
- after unscale;
- pre/post clipping;
- GradScaler result;
- batch completion;
- final scheduler/EMA audit.

Without this connection, the requested receipt could not be produced.

### CUDA RNG matchedness

PL's Gumbel sampling consumes global CUDA RNG while ST does not. After the first
matched batch start, requiring all later CUDA RNG states to remain identical
would require replay, manual resetting, or a changed estimator, all of which are
forbidden.

The corrected frozen rule is:

- identical data fingerprints for all 64 batches;
- identical CPU RNG fingerprints for all 64 batch starts;
- identical CUDA RNG fingerprint at batch 0;
- later CUDA divergence is recorded but is neither hidden nor a failure by
  itself.

No replay or artificial RNG reset is permitted.

### Real CUDA/DDP KAT

Before the two-arm DAG, a separate Slurm GPU KAT is required and receipt-bound.
It executes a world-size-one CUDA DDP backward through the standard FP16
compression hook and verifies:

- the detached observer leaves the original bucket bitwise unchanged before
  invoking the standard hook;
- DDP successfully consumes the standard hook Future;
- a finite FP32 scaled bucket can become nonfinite on the detached FP16 shadow
  cast;
- the analytic and autograd PL residual-logit gradients agree in value and
  direction;
- the receipt is self-hashed and has no performance artifact.

### Parent terminal status

The canonical stability-v2 parent is an
`INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2` HOLD because its PL stage
violated the frozen rule. The new deployer validates that exact terminal state,
the finalization file hash, and both terminal arm receipts. It does not require
or fabricate a parent PASS.

## Frozen mechanism classes

The observer may classify a failed PL optimizer attempt as:

- `DDP_FP16_CAST_OVERFLOW`;
- `UPSTREAM_SCORE_NONFINITE`;
- `SCOUT_VJP_NONFINITE`;
- `SHARED_DETECTOR_BUCKET_OVERFLOW`;
- `AMBIGUOUS_MIXED_FAILURE`.

The finalizer may emit only:

- `PL_NUMERICAL_MECHANISM_LOCALIZED_REPAIR_CLASS_IDENTIFIED`;
- `PL_NUMERICAL_MECHANISM_NOT_LOCALIZED_HOLD`;
- `GRADIENT_DECOMPOSITION_DIAGNOSTIC_INCOMPLETE`.

No winner or estimator ranking is produced.

## Modification boundary

Allowed implementation surfaces:

- opt-in transient payload in `georoute_wrapper.py`;
- opt-in observer/hook wrapper in `tools/train.py`;
- new binder, observer, KAT, stage runner, deployer, finalizer, Slurm launchers,
  tests, and research records.

Frozen and unchanged:

- hard exact-K sampling;
- ordered Plackett-Luce likelihood;
- score-function policy loss;
- straight-through gate;
- temperature, K, weight, baseline momentum, and temporal reduction;
- production FP16 communication hook;
- stability-v1/v2 schemas, thresholds, receipts, and namespaces;
- official AdaTAD config;
- checkpoint, prediction, evaluator, NMS, metric, and official-test surfaces.

## Publication boundary retained

The numerical decision chain remains:

```text
matched gradient decomposition
  -> one uniquely authorized numerical repair
  -> fresh no-performance stability gate
  -> freeze development decisions
  -> three-seed formal study
  -> one sealed official-test opening
```

The later formal study must keep two distinct anchors:

- exact official AdaTAD reproduction for external comparability;
- native-source dense control as the sole causal baseline for native sparse
  routing.

Native-routing deltas cannot be computed against the official 160x160 input
pipeline. The native family must match data, effective batch/update count,
optimizer, scheduler, AMP/EMA, final-EMA checkpoint rule, evaluator/NMS, and
full decode-to-NMS cost. Geometry Zoom remains blocked until same-K learned
residual routing beats fixed and random controls.

## Claim boundary

This absorption authorizes implementation and deployment of a no-performance
diagnosis only. It does not establish NativeTokenSelect effectiveness, PL/ST
superiority, an efficiency gain, official comparability, Geometry Zoom,
P2/P3, official-test access, or paper readiness.
