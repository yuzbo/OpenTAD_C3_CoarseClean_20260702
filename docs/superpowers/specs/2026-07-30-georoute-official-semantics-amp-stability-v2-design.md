# GeoRoute official-semantics real-data AMP stability v2 design

Date: 2026-07-30

Status: `threshold_exceeded_waiting_fail_closed_closeout`

Scope: numerical stability and official-protocol freeze authorization only.
This design does not authorize accuracy measurement, checkpoint selection,
model promotion, P2/P3, Geometry Zoom, an official-test opening, or a paper
claim.

## 1. Evidence and problem statement

The matched diagnosis at exact source
`861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` localized the historical
residual-PL failure to score-function gradient scale. On an identical real
batch and RNG state, PL failed at scales `65536` through `256` only in
`scout_score_function` and first succeeded at `128`; ST succeeded at `65536`.
That authorized the explicit per-tubelet temporal mean implemented in
`768e1a30`.

Stability-v1 at exact source
`86ff1dde6ddb058ca9250f968972c255f19dab92` deliberately required 32 real
batches with zero skipped optimizer attempts at initial GradScaler scale
`65536`. Repaired PL completed two updates and then backed off on batch 3 in
`scout_score_function`. ST completed twenty updates and then backed off on
batch 21 in the detector head. Both forwards remained finite. The v1 finalizer
correctly emitted `STABILITY_GATE_INCOMPLETE_HOLD`.

The exact official AdaTAD path does not impose the v1 rule. It constructs the
default dynamic `torch.cuda.amp.GradScaler()`, uses zero retries, tolerates a
skipped optimizer update, and advances scheduler and EMA on every batch. Thus
v1 is a useful strict numerical stress test, but it is not an official AdaTAD
comparability predicate. It must remain sealed and failed.

## 2. Competing approaches

### A. Official-semantics bounded-adaptation prefix gate

Run a fresh 64-batch, no-metric PL/ST pair with the official dynamic GradScaler,
zero retries, no replay, skipped updates tolerated, and scheduler/EMA advanced
on every batch. Require bounded backoff and a stable tail.

Advantages:

- matches the intended official AMP, scheduler, and EMA transition semantics;
- distinguishes normal dynamic-scale calibration from the historical PL
  collapse through scale `256`;
- preserves an inexpensive fail-closed preflight before full training;
- can use a diagnostic-only seed and independent data order.

Risk: a 64-batch prefix is not a complete official training run. The receipt
must state this explicitly and cannot be performance evidence.

### B. Lower the fixed initial loss scale and retain zero-skip

Starting at `32768` or `16384` could make both arms pass, but it changes the
official default constructor and would hide rather than measure dynamic
adaptation. Rejected.

### C. Skip the numerical gate and launch full multi-seed training

This matches the eventual workload but risks wasting all confirmatory compute
on an unresolved numerical failure. Rejected.

Decision: implement approach A as a new profile and schema. Do not modify or
reinterpret stability-v1.

## 3. Frozen v2 contract

### 3.1 Identity

- profile: `stability_official_semantics_v2`;
- study: `georoute_real_data_amp_stability_v2`;
- arms: `residual_pl_rep_off`, `residual_st_rep_off`;
- diagnostic-only seed and sampler-order seed: `4417`;
- forbidden future paper seeds: `3407`, `3408`, `3409`;
- token budget: `K=64`;
- temporal score-function reduction: `mean`;
- maximum batches: `64`;
- world size: one Slurm-visible GPU per arm;
- no resume and one fresh immutable namespace.

The two arms use the same source config, initialization, training population,
sampler seed, input files and ordered data fingerprints. Seed `4417` controls
Python, NumPy, CPU Torch, CUDA Torch, distributed sampler order and GeoRoute
sampling. It is not reused in the later paper experiment.

### 3.2 Official AMP transition semantics

Within the bounded 64-batch prefix:

- construct `torch.cuda.amp.GradScaler()` without an explicit `init_scale`;
- require the observed initial scale to be `65536`;
- use FP16 autocast exactly as the production train engine;
- set `max_amp_retries_per_batch=0`;
- set `fail_on_skipped_update=false`;
- never replay a skipped batch;
- set `schedule_and_ema_on_success_only=false`;
- advance scheduler and EMA exactly once per consumed batch, including a
  skipped optimizer update;
- retain gradient clipping `L2=1`;
- retain the GeoRoute successful-update index hook only as method-internal
  deterministic routing state; it is not described as an official AdaTAD
  feature.

The diagnostic `max_train_iters=64`, disabled evaluator and disabled checkpoint
are deliberate prefix-gate truncations. Receipts must say
`full_official_training_claimed=false`. They may say only that AMP,
scheduler, and EMA transition semantics match the official implementation
within the observed prefix.

### 3.3 Frozen pass rule

Each arm must satisfy all of:

1. exactly 64 consumed batches and exactly 64 scaler attempts;
2. all forward loss components and total cost finite;
3. at least 62 successful optimizer updates;
4. at most two skipped optimizer updates;
5. maximum consecutive skipped updates at most one;
6. no retry and no replay;
7. minimum observed scaler value at least `16384`;
8. the final 16 attempts all successful;
9. final scaler value at least `16384`;
10. scheduler-advance count exactly 64;
11. EMA-update count exactly 64;
12. no Traceback, OOM, non-finite forward loss/cost, checkpoint, prediction,
    evaluator, metric, NMS output, official-test access, or paper artifact.

The pair must additionally satisfy:

1. identical ordered data-fingerprint sequences across both arms;
2. both seeds exactly `4417`;
3. absolute skip-count difference at most one;
4. final scales differ by at most one factor-of-two level;
5. exact equality of manifest, annotation, class map, video root, pretrained
   checkpoint and official-reference-config hashes.

The threshold permits at most two ordinary two-fold GradScaler calibrations.
It still rejects the historical PL behavior, which required nine consecutive
backoffs to reach `128`. The final stable tail prevents an early calibration
from masking continuing instability.

The only passing decision is
`OFFICIAL_SEMANTICS_AMP_STABILITY_V2_PASS_PROTOCOL_FREEZE_AUTHORIZED`.
Every incomplete, mismatched or threshold-violating run emits
`OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`.

## 4. Binding and provenance

The v2 deployer must require:

- exact clean runtime commit, origin-ref parity and full SHA;
- the immutable failed-pilot finalization;
- the repair-authorizing matched diagnostic finalization;
- the sealed stability-v1 HOLD finalization;
- exact file SHA-256 and internal self-hash for each parent;
- the current source config plus manifest, development annotation, class map,
  development video root and pretrained checkpoint;
- the exact official AdaTAD reference config
  `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` and its
  SHA-256;
- Slurm Job IDs, one-GPU visibility and collision-free rendezvous identities;
- storage and submission-capacity receipts before namespace creation.

The v2 binding records:

- `execution_seed=4417`, `paper_seed_disjoint=true`;
- default GradScaler constructor and observed initial scale;
- retry, fail-on-skip, replay, scheduler and EMA transition policies;
- official reference path/hash and an aggregate
  `official_prefix_transition_semantics_matched` flag;
- `full_official_training_claimed=false`;
- all no-performance/test/claim guards.

No v1 schema, constant, validator, classifier, receipt or namespace is changed.

## 5. Telemetry

The observer retains the complete ordered event stream and adds summary fields:

- attempts and successful-update counts;
- ordered `scale_before`, `scale_after` and update-success arrays;
- skip count and skipped batch indices;
- maximum consecutive skips;
- minimum and final scale;
- final-16 success count and `stable_tail_all_success`;
- retry/replay count;
- scheduler-advance and EMA-update counts;
- ordered data, CPU RNG and CUDA RNG fingerprints;
- failed-attempt non-finite groups and parameters;
- partial-provenance metadata on failure.

Every receipt is atomic, canonical and self-hashed. A failed or partial receipt
remains valid failure evidence but cannot enter the passing classifier.

## 6. Implementation boundaries

Add a separate v2 profile to the existing diagnostic stack:

- protocol constants, schemas and classifier;
- profile-aware binding and validation;
- profile-aware GradScaler and fail-on-skip selection in `tools/train.py`;
- update-audit counters in the production train engine;
- profile-aware seed propagation through bound config, torchrun, rendezvous,
  deployment, stage and finalization receipts;
- v2 parent validation and official-reference binding;
- a separate focused v2 test file.

The disabled observer path and all v1 profiles must remain behavior preserving.
No optimizer, model architecture, loss weight, estimator, data split or
pretrained initialization changes are authorized.

## 7. Verification and deployment

Before remote deployment:

1. test v1 receipts and decisions unchanged;
2. test default GradScaler constructor selection for v2;
3. test skip tolerance, no replay, scheduler/EMA advancement and the exact
   64-batch threshold;
4. test rejection of seed `3407/3408/3409` for v2;
5. test ordered data-sequence mismatch, excessive skip, consecutive skip,
   scale below `16384`, unstable tail, parent tampering and artifact leakage;
6. run Python compilation, focused GeoRoute tests, required C3 regressions and
   `git diff --check`;
7. proxy-sync an exact clean source to N16R4 and repeat the Linux/Torch suite;
8. check Slurm capacity, GPU admission and storage;
9. submit PL and ST in parallel plus one `afterany` finalizer.

Any failure preserves the namespace and blocks all performance experiments.
There is no resume or single-arm replacement.

## 8. Official-comparability boundary after v2

A passing v2 authorizes only freezing the separate paper experiment. It does
not make v2 itself publishable performance evidence.

The later paper experiment must contain:

1. an exact official AdaTAD reproduction at the repository's official
   preprocessing and training recipe;
2. a same-recipe native-source dense control;
3. native-source fixed and random `K=64` controls;
4. matched residual-ST and repaired residual-PL `K=64` arms;
5. identical official split, window/padding, effective batch, optimizer,
   schedule, AMP/EMA, checkpoint selection, evaluator and NMS contracts;
6. at least seeds `3407/3408/3409`, reported as mean and standard deviation;
7. one sealed official-test opening only after development decisions are frozen;
8. Avg-mAP and mAP@0.3:0.7, emphasizing 0.6/0.7;
9. selector-inclusive decode-to-NMS latency, peak memory, energy and actual
   requested/effective/executed token-route cost.

Geometry is tested only after NativeTokenSelect defeats fixed/random at high
tIoU and costs less than native-source dense. No development pilot, diagnostic
or stability-gate value enters a paper performance table.
