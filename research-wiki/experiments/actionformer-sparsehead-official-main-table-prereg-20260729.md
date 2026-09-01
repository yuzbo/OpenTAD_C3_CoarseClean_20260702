# ActionFormer SparseHead Official Main-Table Preregistration

Date: 2026-07-29

Status: `designed`

## Purpose

This protocol prevents the current official matched single-seed screening from
being promoted post hoc into a paper result. It freezes the causal comparison,
seed population, endpoints, uncertainty, cost boundary, failure handling,
factorial attribution and claim rules before Job `1205584` produces a metric.

The target method is a fixed-budget native-grid K384 sparse-head intervention,
not dynamic temporal sampling and not an observation-budget method. It keeps
the full official I3D observations, ActionFormer backbone/FPN geometry,
point/GT mapping, decoder, Soft-NMS and evaluator. It changes:

1. head-query execution to deterministic stratified-uniform K384 native-grid
   centers with the declared physical dependency halo and final scatter; and
2. training loss support to `selected_native_grid_queries`.

The mAP difference therefore estimates this complete method intervention. The
isolated CUDA kernel result and the selected-loss effect must be attributed
separately.

## Immutable Sources and External Anchor

- Official source commit/tree:
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`.
- Candidate source commit/tree:
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907`.
- Sealed official released anchor: Avg-mAP `66.833392`, mAP@0.3--0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`.
- The released anchor is an external protocol-sealed reference only. The causal
  delta is always sparse minus a dense model retrained from the same candidate
  commit and paired seed.
- Historical `63.61`, old OpenTAD/AdaTAD numbers and the frozen v16
  decode-cross results are forbidden matched comparators.

## Stage S0: Single-Seed Screening

Job `1205584` uses seed `1234567891`, the official 413-file I3D dataset,
official 5-warmup + 30-epoch execution, no resume, explicit
`epoch_035.pth.tar:state_dict_ema`, identical post-processing/evaluator and a
pinned-official independent recomputation of both raw prediction files.

It is always `paper_main_table_eligible=false`.

Screening may unlock the predeclared five-seed study only when all conditions
hold:

1. both arms complete without a model-valid non-finite failure;
2. source, environment, data, checkpoint, raw-prediction, evaluated-video and
   evaluator receipts validate;
3. native metrics and the independent official recomputation agree within the
   frozen numeric tolerance;
4. sparse minus dense Avg-mAP is at least `-1.00` percentage point; and
5. sparse minus dense mAP@0.6 and mAP@0.7 are each at least `-1.50`
   percentage points.

An engineering failure is repaired under a new exact commit/runtime/root and
the same scientific contract. A legal model result below any screening bound
stops automatic multiseed deployment and triggers the frozen Pro-level
negative-result analysis. The budget, thresholds, seed, data, checkpoint,
loss, decoder and evaluator may not be changed to rescue it.

## Stage S1: Paired Five-Seed Main Study

### Seed generation

The paired seed set is fixed before S0 metrics:

`[1234567891, 1423812477, 737690612, 1788897292, 1322022747]`.

Seed zero is the official seed. For `i=1..4`,

`seed_i = int(SHA256("actionformer-sparsehead-official-matched-v1:i")[:8], 16) & 0x7fffffff`.

The four full derivation digests are:

- `d4dda77d30f39a5becf99f048fad80bb61a57f41c1ac7c76d8a62aceb17488b9`;
- `abf843f471f9eaf4ce090e2432f25960d5c0821ca0d28c2db4f24310973a862e`;
- `6aa0680cee0cdf516e69f0fe4e06e0b43c2e766d7e604bfa59ffba3a447df545`;
- `cecc775b98e2e3521a31fc148a1e9c68eae5d1a5ff6625e487b52a05f2a32ddf`.

The canonical seed-set JSON SHA-256 is
`a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
No reseeding, replacement or performance-based subset is allowed.

### Paired arms

Each seed runs exactly two primary training arms:

- `D`: dense full native-grid ActionFormer head;
- `S`: deterministic stratified-uniform K384 native-grid sparse head with
  `selected_native_grid_queries` training-loss support.

Both arms use the same candidate commit, feature/annotation manifests,
dataloader semantics, optimizer, batch size, official 5+30 schedule, terminal
EMA rule, evaluator and Slurm GPU class. Arm order alternates by seed to expose
thermal/order effects; accuracy training order itself is not a statistical
replicate.

### Runtime receipts added before S1

Every allocation must fail closed on:

- a live content rehash of all 413 expected feature IDs, file SHA-256, shape
  and dtype, with exact annotation-to-feature alignment;
- exact runtime effective-config JSON and SHA-256 for each arm;
- exact CLI, environment, candidate/audit Git identity and cleanliness;
- dataloader train/test split identities and counts;
- optimizer/scheduler/EMA state in explicit terminal epoch 35;
- no-resume and exactly 35 completed epoch markers;
- exact 212-video raw-prediction/evaluator coverage; and
- native versus pinned-official independent metric agreement.

Counting `*.npy` files or merely pinning an old manifest receipt is
insufficient for S1.

## Accuracy Endpoints and Statistics

The statistical unit is the paired seed, not a video, proposal, window or
repeated evaluator call.

- Primary accuracy endpoint: Avg-mAP, the mean of mAP@0.3--0.7.
- Ordered high-IoU safety endpoints: mAP@0.6, then mAP@0.7.
- Secondary endpoints: each remaining tIoU, per-class AP, short/medium/long
  action buckets, center/start/end error, pre-NMS proposal recall, post-NMS
  recall and calibration.

For each endpoint, report every seed and arm, arm mean plus sample SD, all five
paired deltas `S-D`, paired-delta mean/SD and a two-sided 95% Student-t
interval over the five seed deltas. A hierarchical seed/video bootstrap is a
robustness interval, not a replacement that turns videos into independent
training replicates. Re-running the evaluator is a reproducibility check and
does not increase sample size.

No checkpoint or hyperparameter is selected on THUMOS test results. Only the
predeclared terminal EMA is evaluated. Missing pairs are never replaced:
superiority or non-inferiority requires all five valid pairs. Engineering
retries retain the same seed and scientific commit; a model-valid failure is
reported as a failed outcome and blocks a positive claim.

## Accuracy Claim Rules

An accuracy-preserving efficiency claim requires:

1. lower 95% confidence bound for mean `Delta Avg-mAP` at least `-0.20`
   percentage point;
2. lower 95% confidence bounds for mean `Delta mAP@0.6` and
   `Delta mAP@0.7` each at least `-0.50` percentage point; and
3. the full cost rule below.

An accuracy-improvement claim additionally requires the lower 95% confidence
bound for mean `Delta Avg-mAP` to be strictly above zero, with both high-IoU
safety rules satisfied.

If an interval crosses its boundary, the result is neutral/inconclusive for
that claim. If the Avg upper bound is below zero, the result is negative for
this frozen K384/selected-loss intervention, not for every possible
SparseHead method.

## Stage S2: Preregistered 2x2 Attribution

The same two trained checkpoint families are cross-evaluated without model
selection:

| Training support | Evaluation queries | Interpretation |
|---|---|---|
| full | dense | matched dense control |
| full | K384 | execution/representation-only sparsification |
| selected K384 | dense | selected-loss/optimization effect |
| selected K384 | K384 | complete proposed intervention |

Before evaluation, a focused compatibility gate must prove that switching
query execution changes no state-dict key or protected decoder/evaluator
field. These rows use the same terminal EMA checkpoints and exact official
evaluator. They are attribution rows, not extra independent seeds and may not
be used to select K, halo, loss normalization or checkpoint.

## Stage S3: Detector-Pipeline Cost

The primary cost boundary is the official precomputed-feature TAD detector
pipeline: feature loading, preprocessing/H2D, ActionFormer backbone/FPN,
selector, head/gather/scatter, decoder/Soft-NMS and result serialization. It
must not be described as raw-video end-to-end cost because official I3D feature
extraction is upstream and unchanged. A second preloaded model-only boundary
may be reported separately.

For every terminal seed pair:

- use the same Slurm GPU class and logical `cuda:0`;
- pin driver/CUDA/PyTorch/environment and power/performance state;
- pair identical videos and valid lengths;
- stratify short (`T<=384`), medium (`384<T<=768`) and long (`T>768`);
- run at least 5 discarded warmups and 30 retained measurements per stratum;
- alternate dense/sparse order across rounds;
- synchronize before and after CUDA-event regions;
- retain CPU wall time, total GPU time, selector/head/scatter times, decoder/NMS
  time, peak allocated/reserved VRAM and RSS;
- retain every raw timing sample and mark, but do not silently delete,
  outliers; and
- require stage accounting to agree with total wall time within 5%, otherwise
  fail the cost receipt.

Report paired video/bootstrap 95% intervals for absolute latency and the
dense/sparse ratio. The detector-pipeline efficiency claim requires aggregate
median speedup at least `1.05x`, lower 95% confidence bound strictly above
`1.00x`, and no duration stratum whose speedup interval includes or falls
below `1.00x`. The existing `1.571574x` selector-inclusive isolated-head gate
is reported only as an engineering microbenchmark.

## Negative-Result Analysis Contract

Every legal negative or neutral result preserves all raw values and tests at
least these competing explanations:

1. selected-query positive/negative assignment coverage and changed EMA loss
   normalizer;
2. physical halo occupancy/FPN-level/duration-dependent representation loss;
3. optimization and classification/regression gradient-scale drift;
4. pre-NMS query/proposal recall and boundary regression bottleneck; and
5. score calibration and Soft-NMS interaction.

For each explanation the analysis records supporting evidence, counterevidence,
a falsifiable prediction and a minimal decisive diagnostic. Diagnostic sweeps
use the training split or a separately sealed training-internal subset and may
not tune the official test result. Any new training arm is separately
preregistered and remains diagnostic until it independently satisfies this
protocol.

## Paper Boundary

S0 is screening. S1 accuracy without S3 cost can support a matched accuracy
finding but not an efficiency claim. S3 cost without S1 accuracy cannot
support an accuracy-preserving claim. `paper_ready` requires S1, S2, S3,
independent recomputation, complete receipts and an explicit result analysis.

