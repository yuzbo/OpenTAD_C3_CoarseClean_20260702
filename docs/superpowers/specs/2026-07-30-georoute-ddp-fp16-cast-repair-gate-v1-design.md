# GeoRoute DDP FP16-cast repair gate v1

## Decision

Run one fresh, matched, no-performance PL/ST numerical gate after the sealed
gradient-decomposition diagnostic uniquely identified
`DDP_FP16_CAST_OVERFLOW`. The only intervention is to disable DDP FP16
communication compression for both arms. The ordered-PL estimator, ST control,
objective, selector, data, optimizer, AMP, EMA and scheduler semantics remain
unchanged.

This gate cannot produce an accuracy, efficiency, Geometry Zoom, P2/P3 or paper
claim. A pass only authorizes freezing a later official-comparable performance
protocol.

## Immutable parent

- Parent study: `georoute_pl_gradient_decomposition_diagnostic_v1`
- Parent runtime:
  `33f721be83e0ad7f7a36e853491e7a14f148814b`
- Parent decision:
  `PL_NUMERICAL_MECHANISM_LOCALIZED_REPAIR_CLASS_IDENTIFIED`
- Unique repair class: `DDP_FP16_CAST_OVERFLOW`
- Parent finalization file SHA-256:
  `816819086374f964264d3a8bb4810842f97ef554d5661d2ec4a6b85fd135bc9c`
- Parent PL/ST receipt file SHA-256:
  `9f402fc46ff9398e2677130596a5b0bd7bed5a94dcac2e36ba3e6d46ec476a2f`
  and
  `9dfaa40b153af57cd04516ea37d96bf84f91b23037370c2bd8f4421d508d962d`

Deployment must validate the parent finalization, both arm receipts, parent
deployment, canonical self-hashes, runtime commit and immutable inputs before
creating a namespace or submitting any job.

## Frozen study

- Study ID: `georoute_ddp_fp16_cast_repair_gate_v1`
- Profile: `ddp_fp16_cast_repair_no_compression_v1`
- Arms: `residual_pl_rep_off`, `residual_st_rep_off`
- Seed: `2307`, derived before execution as
  `1000 + int(SHA256(study_id)[:8], 16) % 9000`
- Forbidden seeds: `7367`, `4417`, `3407`, `3408`, `3409`
- Real development training population and order: identical to the sealed
  gradient diagnosis
- Shape/budget: `B/T/N/K = 1/384/220/64`
- Ordered-PL temperature/weight/baseline/temporal reduction:
  `0.7/1.0/0.95/mean`
- Default dynamic `GradScaler`
- Initial observed scale: `65536`
- Retry/replay: `0/0`
- Scheduler and EMA advance once per consumed batch, matching the official
  prefix transition cadence
- Batches: `64`
- Metrics, checkpoint, prediction, evaluator, NMS and official test: disabled

## Single registered intervention

| Factor | Parent | Repair gate |
|---|---:|---:|
| `solver.fp16_compress` | `true` | `false` |

No BF16 hook, reduced initial scale, extra clipping, loss rescaling,
temperature/weight/K change, estimator change, selector change, data-order
change or arm-specific treatment is allowed. Both matched arms receive the same
communication change.

The official AdaTAD reference enables FP16 communication compression. Therefore
this gate is intentionally not a full official-recipe reproduction and is not
performance-comparable. It binds the official reference only to preserve the
unchanged AMP, clipping, EMA and scheduler-transition semantics and to identify
the registered difference precisely.

## Same-commit CUDA/DDP KAT

Before the two-arm gate is admitted, one Slurm CUDA KAT at the exact clean
runtime commit must:

1. initialize real world-size-one NCCL DDP over an FP32 parameter;
2. use a `65536` loss scale to create a finite scaled FP32 gradient of `70000`;
3. register no communication hook;
4. show default DDP reduction preserves that finite FP32 gradient;
5. show a detached FP16 shadow cast of the same value becomes nonfinite;
6. unscale to a finite FP32 gradient and complete an optimizer update; and
7. emit only a self-hashed no-performance receipt.

A missing or failed KAT blocks deployment.

## Frozen pass rule

Each arm must satisfy:

- exactly 64 consumed batches, forward attempts and optimizer attempts;
- all forward losses finite;
- at least 62 successful updates;
- at most two skipped updates;
- no consecutive skipped updates;
- no retry or replay;
- minimum and final scale at least `16384`;
- all final 16 batches update successfully;
- exactly 64 scheduler advances and EMA updates; and
- zero forbidden performance artifacts.

The pair must additionally have identical ordered data and CPU-RNG
fingerprints, identical batch-zero CUDA RNG (later divergence is recorded but
not forbidden because PL alone samples Gumbel noise), the same seed and
immutable inputs, skip-count delta at most one, final-scale ratio at most two,
and the identical registered no-compression intervention.

Pass decision:
`DDP_FP16_CAST_REPAIR_GATE_PASS_MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED`.

Any missing/invalid arm, artifact, parent mismatch or threshold violation yields
`DDP_FP16_CAST_REPAIR_GATE_HOLD`. Failure preserves the namespace and forbids
resume, one-arm supplementation, threshold adaptation and performance
inference.

## Downstream boundary

Only a complete pairwise pass authorizes design and freezing of a separate
formal study containing:

1. an exact official AdaTAD reproduction;
2. same-source dense and GeoRoute arms under one matched recipe;
3. at least three preregistered independent seeds;
4. identical split/windows/padding, effective batch and update counts,
   optimizer/scheduler/AMP/EMA, checkpoint selection, evaluator and NMS;
5. one sealed official-test opening after development decisions; and
6. selector-inclusive decode-to-NMS latency, peak memory, energy and actual
   token/route cost.

No diagnostic or repair-gate number may enter a paper performance table.
