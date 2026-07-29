# GeoRoute real-batch AMP diagnostic and paper-comparability design

Date: 2026-07-29

Status: `approved_design_pending_implementation`

Scope: numerical diagnosis and repair authorization only. This design does not
authorize a performance claim, model selection, P2/P3, Geometry Zoom, or an
official-test opening.

## 1. Evidence that motivates this design

Exact source `c822add335c38a9f6c63e609237c4bfa9b9f468d` passed both a
standalone full-graph CUDA AMP gate and all six fresh per-arm P0 gates. The
fresh six-arm namespace nevertheless failed because residual-PL Job `1204309`
could not complete its first real training batch after eight AMP retries at
loss scales `32768, 16384, 8192, 4096, 2048, 1024, 512, 256`.

This establishes only that the synthetic full-graph P0 is not a sufficient
real-batch stability certificate. It does not establish that PL is
scientifically worse than ST, that geometry representation helps, or that any
GeoRoute variant is efficient.

The historical job did not persist sample indices, the complete input batch,
sampler state, or all CPU/CUDA RNG states before its first attempt. Therefore
an exact bitwise replay of Job `1204309` is impossible. The next run must be
described honestly as a deterministic, same-config, first-real-batch
reproduction on the same data path, not as a replay of the historical tensor.

## 2. Competing approaches

### A. Instrument the production training call path

Add an optional diagnostic observer to `train_one_epoch`, retain the ordinary
dataset, model, autocast, GradScaler, optimizer and retry implementation, and
run matched residual-PL and residual-ST arms.

Advantages:

- smallest execution drift from the failing path;
- directly distinguishes scaled-backward overflow, unscaled non-finiteness,
  clipping effects and skipped optimizer updates;
- keeps the diagnostic disabled by default.

Risk: a callback inside the train engine must be rigorously tested so that its
disabled path is behavior preserving.

### B. Build an independent one-batch reproducer

This is simpler to package, but it can silently drift in data loading, DDP,
autocast scope, loss construction or optimizer coverage. It is rejected as the
primary diagnosis.

### C. Rerun a complete six-arm pilot with more AMP retries

This spends training compute before identifying the cause and changes the
numerical protocol without evidence. It is rejected.

Decision: implement approach A. A small wrapper may configure the run, but the
forward/backward/step must remain the exact production `tools/train.py` and
`train_one_epoch` path.

## 3. Frozen diagnostic questions

Two arms run in parallel with the same source commit, bound config, seed
`3407`, pretrained initialization, dataset, sampler, batch size, optimizer and
first real training batches:

1. `residual_pl_rep_off`;
2. `residual_st_rep_off`.

The diagnostic answers:

1. At which stage and parameter group does the first non-finite gradient
   appear?
2. Does residual-ST remain finite on the matched input?
3. Does residual-PL produce a successful update at scale `128`, `64`, or below
   when diagnosis-only retries continue beyond the historical eight-attempt
   limit?
4. Are detector losses finite before the score-function term is added?
5. Is the failure caused by an intrinsically non-finite loss/gradient or by a
   finite unscaled gradient whose scaled FP16 representation overflows?

No accuracy metric, prediction, evaluator, NMS result, checkpoint or test split
is produced.

## 4. Binding and no-drift contract

Each arm must bind:

- exact clean runtime commit and origin-ref parity;
- Slurm Job ID, host, CUDA device visibility and rendezvous identity;
- exact bound-config SHA, pretrained SHA, annotation SHA, class-map SHA and
  video-root identity;
- seed, world size one, batch size and sampler configuration;
- source branch, arm name and representation switches;
- `workflow.disable_checkpoint=true`, one epoch, and a bounded number of real
  batches;
- no resume, no config override, no test/evaluator invocation and no official
  test path;
- a fresh immutable run root and atomic self-hashed receipts.

The paired arms must have matching population descriptors and input-tensor
fingerprints for every compared batch. If stochastic estimator sampling makes
post-input RNG state arm-specific, the shared pre-forward RNG hash and input
hash must still match.

## 5. Diagnostic event schema

The observer writes an atomic append-style logical event stream. Every
publication rewrites a complete canonical JSON object and recomputes its
self-hash; no partially written event is valid.

### 5.1 Run and batch identity

- schema version, runtime commit, Job ID, arm and config hashes;
- epoch, batch index and retry index;
- recursively summarized sample metadata;
- tensor shape, dtype, device and SHA-256 for the model input;
- CPU RNG SHA-256 and every visible CUDA RNG SHA-256 before the first attempt;
- optimizer-step count and GradScaler scale.

The raw video tensor is not persisted. Its cryptographic hash is sufficient to
prove paired input equality without duplicating the dataset.

### 5.2 Forward and loss state

- each detector loss name, dtype, scalar value and finite flag;
- detector-loss sum before the policy term;
- policy estimator, policy weight, baseline and advantage;
- ordered log-probability min, max, sum, dtype and finite flag;
- policy loss dtype, value and finite flag;
- route-logit min, max, mean, dtype and finite flag;
- total loss dtype, value and finite flag.

These values are detached telemetry and must not change the graph.

### 5.3 Gradient checkpoints

For every required trainable parameter, and for grouped detector,
score-function/scout, adapter and backbone summaries, record:

1. immediately after scaled backward;
2. immediately after GradScaler unscale;
3. immediately after gradient clipping;
4. after scaler step/update.

Each summary includes parameter name, dtype, shape, finite count, non-finite
count, maximum absolute value and L2 norm. The event also records scale before,
scale after, whether the optimizer update executed, and whether the attempt was
replayed.

## 6. Retry policy for diagnosis

The historical experiment remains bound to eight retries. The diagnostic may
use a separately named, diagnosis-only limit sufficient to observe scales
below `256`, beginning at the same default initial scale. This is not a
training-policy repair and cannot be inherited by a performance experiment.

Interpretation:

- finite unscaled gradients plus success at `128/64` supports an AMP-scale
  compatibility diagnosis;
- non-finite unscaled gradients localize a mathematical or graph problem;
- a finite detector-only sum but non-finite score-function contribution
  localizes the PL path;
- matched PL and ST failure indicates a shared detector/data mechanism rather
  than an estimator-specific mechanism;
- input-fingerprint mismatch invalidates the paired comparison.

## 7. Repair authorization

The diagnostic finalizer may emit only:

- `ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`;
- `ROOT_CAUSE_NOT_LOCALIZED_HOLD`;
- `DIAGNOSTIC_INCOMPLETE_NO_REPAIR`.

A repair is authorized only when the failing stage and parameter group are
identified and the PL/ST contrast excludes an obvious shared-data failure.
The repair must be minimal and hypothesis-matched. Examples include an
explicit FP32 construction of a detector-cost sum that was accidentally formed
under autocast, or a justified GradScaler initialization/floor change. Loss
normalization, weight changes, clipping changes, estimator changes or extra
model components require a new scientific design and are not authorized here.

After a repair, a fresh real-data stability gate must:

- execute the production training path;
- include the residual-PL arm and matched residual-ST control;
- cover the first 32 real batches or the complete observed stress prefix,
  whichever is longer and feasible;
- complete without a per-batch retry exhaustion, non-finite loss/cost/gradient,
  OOM or traceback;
- publish no metric, prediction, checkpoint or test evidence.

Only this gate can authorize a new scientific experiment namespace.

## 8. Official-comparability boundary

The existing 20-epoch, one-seed GeoRoute pilot is development-only. Even a
successful replacement would provide exploratory contrasts, not a paper-table
result.

Before any confirmatory run, freeze an official-comparability manifest with:

- an exact reproduction arm for the repository's official AdaTAD THUMOS14
  recipe and pretrained initialization;
- a matched native-source dense control that differs from the proposed method
  only where native-token execution requires it;
- identical train/validation/test splits, temporal windows, padding policy,
  label mapping and official evaluator;
- identical optimizer, learning-rate schedule, warmup, total optimizer updates,
  effective batch size/accumulation, AMP policy, EMA and checkpoint selection;
- identical detector head and post-processing, including the effective merged
  NMS configuration;
- disjoint development-selection and confirmatory seeds, with no post-result
  threshold tuning;
- final-EMA, preregistered model selection and a sealed official-test opening
  only after the method and analysis are frozen;
- decode-to-NMS p50/p95 latency, peak memory and energy measured on matched
  hardware with the selector/scout included;
- requested, effective, unique, padded and actually executed token budgets;
- no validation/test GT, teacher, oracle or raw-prediction cache in routing.

Because source-native token execution changes preprocessing, it must not be
described as an unmodified official AdaTAD run. The paper comparison needs both
the exact official reproduction and the matched native-source dense control:
the former establishes external benchmark comparability; the latter isolates
the routing intervention.

A result is paper-eligible only if the manifest validates every item. Failed,
diagnostic, single-seed, partial, evaluator-mismatched or model-only cost
results remain explicitly excluded from paper tables.

## 9. Verification

Local and clean remote checks must cover:

- disabled observer path equivalence;
- event ordering and exact scale transitions;
- scaled, unscaled and clipped gradient summaries;
- lower-scale success and retry-exhaustion receipts;
- self-hash and tamper rejection;
- no checkpoint/evaluator/test output;
- PL/ST input-fingerprint equality;
- source/Slurm/config/pretrained/data binding;
- existing GeoRoute and required C3 regression suites.

GitHub fetch/push/clone/ls-remote operations use the `RTK.md` academic proxy
from the first request and must finish with full HEAD, origin-ref and clean-tree
parity.
