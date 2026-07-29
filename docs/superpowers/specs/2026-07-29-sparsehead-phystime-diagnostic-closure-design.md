# SparseHead / PhysTime Diagnostic Closure and Publishable Benchmark Design

Date: 2026-07-29
Status: designed
Exact base commit: `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98`
Exact base tree: `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`

## 1. Objective

Close the remaining SparseHead / PhysTime evidence gaps without turning an
internal diagnostic into a paper claim:

1. independently recompute the v16 frozen-tensor decode, cross-window Soft-NMS,
   and THUMOS14 AP on CPU in NumPy/float64;
2. audit 64 pre-sealed training windows for SDPQ assignment and support
   observability;
3. establish a fail-closed ActionFormer/THUMOS14 comparability manifest;
4. run a publishable matched control before selecting at most one structural
   method change.

No core algorithm, threshold, checkpoint, data split, or evaluator may be
changed before the no-training diagnostics and protocol classification finish.

## 2. Evidence already fixed

The v16 four-condition replay is a tested, frozen-checkpoint mechanistic result:

- selected/online: uniform `41.2566`, physical decode `50.1536`;
- selected/EMA: uniform `41.2830`, physical decode `50.0979`;
- physical/online: uniform `40.1077`, physical decode `57.5556`;
- physical/EMA: uniform `40.2965`, physical decode `57.6087`.

It proves that physical-time decoding is coordinate-correct and materially
better for the same frozen tensors. It does **not** prove a trained SparseHead
gain, a compute saving, an official ActionFormer reproduction, or superiority
over the historical `63.61` run.

The historical `63.61` result belongs to commit `3ac93a1`, jobs
`1159491-1159495`, and an older raw-video K384/J192 experiment family. Its
record does not bind a seed, checkpoint hash, NMS/evaluator hash, or all fields
needed for a strict current-protocol comparison. It is retained as historical
evidence, not a main-table control.

## 3. Official ActionFormer protocol boundary

The official release and ECCV paper define the canonical THUMOS14 reference:

- official repository:
  <https://github.com/happyharrycn/actionformer_release>;
- pinned official commit:
  `61ea7eb9308a568b0cf45e3804830836e30061de`;
- pinned tree:
  `7b06c5261ba244788c942a0d73e304581bc35154`;
- pinned official config:
  <https://raw.githubusercontent.com/happyharrycn/actionformer_release/61ea7eb9308a568b0cf45e3804830836e30061de/configs/thumos_i3d.yaml>,
  SHA-256
  `73f8aeaf7deef93aba57259badd4c454990ec1e0ce6eaa7c3434db44baaeeaf0`;
- pinned README SHA-256:
  `bdee4eb088a74e190935097742c7dbfaf254eb912f79729dccd73b9b36b33db8`;
- official THUMOS archive MD5:
  `375f76ffbf7447af1035e694971ec9b2`;
- ECCV paper:
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf>.

Required official-family fields:

- THUMOS14 validation split for training and test split for evaluation;
- the released two-stream I3D Kinetics features and annotations;
- 16-frame clips, frame stride 4, about 0.1333 seconds per feature;
- 2048 input channels, maximum sequence length 2304;
- ActionFormer identity FPN, local attention window 19;
- AdamW, learning rate `1e-4`, weight decay `0.05`, batch size 2;
- 30 training epochs for the official config;
- center sampling radius 1.5;
- EMA checkpoint evaluation;
- no external classification-score fusion unless every compared arm uses it;
- pre-NMS threshold `0.001`, top-k 2000, maximum 200 final segments,
  multiclass Soft-NMS sigma `0.5`, NMS IoU threshold `0.1`, and voting
  threshold `0.7`;
- THUMOS14 AP at tIoU `0.3,0.4,0.5,0.6,0.7`.

The official release reports two versioned expectations: the initial code
reproduction around `62.6` average mAP and the current released checkpoint
`82.13/77.80/70.95/59.40/43.87`, average `66.83`. A result must record the
official Git commit and config hash before either expectation is used.

## 4. Evidence strata

Every result receives exactly one protocol stratum.

### 4.1 `official_reproduction`

Unmodified official ActionFormer code, released I3D features, official config,
official evaluator, and a recorded official repository commit. This is the
versioned external anchor.

### 4.2 `matched_method_control`

Baseline and method share the same:

- dataset split and annotation bytes;
- feature/backbone bytes and temporal stride;
- observation budget and selection positions;
- crop/window sequence;
- optimizer, schedule, epoch selection, seed policy, AMP and EMA policy;
- candidate density, post-processing, NMS, evaluator, and tIoU thresholds.

Only the declared method variable may differ. Such rows are publishable in a
clearly named matched-protocol table. They must not be labelled as an
unmodified official ActionFormer reproduction.

### 4.3 `external_reference_only`

A published or historical number whose full executable receipt is unavailable,
or whose feature/training/post-processing family differs. It may appear with
an explicit protocol label but cannot be used to calculate the method delta.

### 4.4 `diagnostic_only`

Frozen-checkpoint cross-decode, assignment/support audits, oracle analyses, and
counterfactual replays. These explain mechanism and cannot be headline
benchmark rows.

## 5. Fail-closed comparability manifest

A new fail-closed validator emits
`actionformer_thumos_comparability_v2`. It has no skip-hash mode and never
infers comparability from a method name or a close mAP value.

Required records:

- protocol family and evidence stratum;
- source repository URL, commit, tree/config SHA-256;
- annotation/class-map/data-manifest SHA-256;
- split, class count, video count, blocked videos;
- feature/backbone name, checkpoint SHA-256, input dimension, clip length,
  feature/frame stride, temporal resolution;
- maximum sequence/window length and observation budget;
- model/head/projection identifiers and effective-config SHA-256;
- optimizer, schedule, epochs, batch size, seed, AMP, EMA and checkpoint rule;
- pre-NMS threshold/top-k, NMS type/sigma/voting/max segment count,
  rounding policy;
- evaluator implementation/hash, tIoU thresholds, raw prediction count;
- result artifact/hash and all metrics;
- official evaluation log, raw `eval_results.pkl`, independent official
  evaluator recomputation, and a metric attestation binding all three.

Hard failures:

- absent required field;
- unmatched split, annotation, feature bytes/stride, evaluator, tIoU set, score
  fusion, NMS, or checkpoint policy inside a claimed matched comparison;
- a candidate labelled `official_reproduction` that differs from the official
  manifest;
- a main-table delta computed across protocol families;
- test labels, GT, teacher output, raw-prediction cache, or oracle information
  influencing inference or sampling.

## 6. Independent frozen-artifact recomputation

### 6.1 Independence boundary

The new validator must not import:

- `replay_phystime_decode_cross`;
- `validate_phystime_decode_cross_replay`;
- `apply_sliding_window_nms`;
- OpenTAD NMS bindings;
- OpenTAD evaluator or AP helpers.

It may read their immutable artifact schemas. All geometry, ranking, Soft-NMS,
duplicate-annotation removal, tIoU, TP/FP matching, interpolated AP, and
aggregation are implemented locally with NumPy.

### 6.2 Inputs

For each formal v16 completion:

- formal `DECODE_CROSS_COMPLETE.json`;
- capture manifest and capture NPZ;
- uniform and physical `decoded_candidates.npz`;
- uniform and physical pre-cross/result/metrics artifacts;
- THUMOS14 annotation JSON;
- a small explicit NMS/evaluation policy JSON extracted from the frozen
  effective config.

Every input path, size, and SHA-256 is checked against the completion.

### 6.3 Numeric semantics

- geometry is recomputed in NumPy float64;
- score values preserve the sealed source dtype until the threshold/top-k
  boundary, then are represented in float64;
- descending ranking uses a stable sort with original sequence index as the
  tie breaker;
- Gaussian Soft-NMS is independently reimplemented from the published
  algorithm and the repository C++ semantics;
- all invalid, zero-duration, thresholded, and rounded candidates are counted;
- AP uses float64 VOC2011 interpolated precision-recall.

Outputs report both exact artifact equality where possible and numeric deltas.
The validator fails if:

- masks or candidate traversal differ exactly;
- decoded proposals differ by more than `1e-4` seconds;
- native direct/replay metrics differ by more than `1e-10` in fractional mAP;
- independent-vs-stored metrics differ by more than `1e-4` in fractional mAP
  (0.01 percentage point);
- the sign of any physical-minus-uniform tIoU delta changes;
- tie handling changes average mAP by more than 0.01 percentage point.

A failure reopens implementation/evaluation diagnosis and blocks model claims.

## 7. Sealed 64-window assignment/support audit

### 7.1 Window sealing

Before any audit metric is read, create a deterministic manifest of exactly 64
training windows from seed 42. It stores sample/video/window identifiers,
selected-index hashes, GT hashes, annotation/data/config/checkpoint hashes, and
the ordered manifest hash. Re-running with the same inputs must reproduce the
same manifest byte-for-byte.

### 7.2 Runtime path

Under `torch.no_grad()` and `model.eval()`:

1. run the frozen backbone;
2. align native tubelet geometry;
3. run `PhysTimeMeasureProjection`;
4. build SDPQ query points;
5. call the frozen SDPQ target assignment once;
6. record geometry and assignment state without calculating losses or updating
   the loss normalizer.

### 7.3 Required metrics

Per window and in aggregate:

- domain-valid query count;
- evidence-covered query count;
- assignment-eligible query count;
- null-evidence query count;
- positive query count;
- positive uncovered/low-coverage count;
- GT count and GT with zero eligible query;
- GT with zero assigned query;
- reservation collisions;
- support observability mean and quantiles;
- physical cell widths and coverage ratios;
- GT duration strata: `<1s`, `[1s,4s)`, `[4s,16s)`, and `>=16s`;
- per-stratum eligible, assigned, no-eligible, no-assigned and coverage rates.

This artifact is `diagnostic_only`. It may justify one subsequent structural
change but is never an accuracy result.

## 8. Experiment matrix

### Phase A: no-training closure

- independent v16 recomputation for all four conditions;
- 64-window SDPQ support/assignment audit;
- protocol-manifest classification of v16 and historical `63.61`.

### Phase B: official anchor

Run the unmodified official ActionFormer release with the released I3D
THUMOS14 data and exact official config. Record official repo commit, data MD5
and SHA-256, environment, EMA checkpoint, evaluator and metrics. A modern
official release is expected near `66.8`; the documented old-code `62.6`
expectation is not mixed with it.

### Phase C: matched budget table

Use one declared family at a time.

Preferred first family is released I3D features, because it is closest to the
official benchmark:

1. dense/unmodified detector control;
2. fixed GT-free random K384 observation control with the unchanged official
   head (`selection_budget` intervention only);
3. the physical-query/SDPQ method with the exact same K384 observations
   (`head_projection` intervention only).

The matched K384 rows share the same features, window sequence, observation
positions, seed, schedule, EMA, NMS and evaluator. If the method changes both
selection and head, split this into separate one-variable stages rather than
claiming a single causal delta.

The raw-VideoMAE K384/J192 route may form a separate end-to-end compute table,
but it must never use the official I3D ActionFormer number as its matched
baseline.

### Phase D: one structural decision

After Phase A-C, run one bounded Pro/adversarial review. Competing hypotheses
must include at least:

1. training assignment/support observability is the dominant bottleneck;
2. representation loss from sparse raw observations is dominant;
3. optimization/checkpoint selection is dominant;
4. the historical `63.61` gap is primarily protocol drift.

For each, record supporting evidence, counter-evidence, falsifiable prediction,
and the smallest decisive experiment. Select at most one structural change.
No multi-variable rescue sweep is allowed.

## 9. Deployment and receipts

Every remote run uses:

- a new clean exact-commit runtime;
- a new non-overwriting run root;
- full-content preflight and focused tests;
- a Slurm allocation for every GPU operation;
- logical `cuda:0` without replacing `CUDA_VISIBLE_DEVICES`;
- immutable config/data/checkpoint/manifest hashes;
- a formal completion artifact and hard-failure log scan.

Engineering failures are preserved and repaired until a valid model result is
obtained. Negative valid model results are preserved and analyzed; they are not
relabelled as engineering failures and do not authorize silent retuning.

## 10. Claim boundary

Before all gates pass, the strongest allowed claims are:

- physical-time decode is the coordinate-correct counterfactual for the frozen
  v16 G1a tensors;
- online and EMA decode behavior is descriptively stable;
- the current result remains below an older, non-strictly-comparable `63.61`
  historical run;
- the cause of the remaining gap is unresolved pending independent evaluation,
  assignment/support audit, and matched protocol controls.

`empirically_supported` requires a valid matched comparison. `paper_ready`
additionally requires the official anchor, complete receipts, cost accounting,
and the final Pro claim audit.
