---
type: experiment_plan
node_id: plan:native-crop-paper-experiments
title: "Native-Crop paper experiment roadmap"
stage: designed
status: s2_preregistration_revision
outcome: pending
tags: ["offline-tad", "native-crop", "paper-plan", "claim-driven"]
added: 2026-07-20
---

# Native-Crop Paper Experiment Roadmap

## Paper Thesis

The paper is not about reducing full-frame resolution. Its intended method
selects source-coordinate local views while retaining the complete temporal
axis:

> A low-cost global view can guide source-native local spatial computation so
> that an offline TAD detector preserves high-tIoU localization at a better
> measured accuracy-cost tradeoff than dense full-frame processing.

This thesis has two claims:

| Claim | Minimum convincing evidence |
|---|---|
| C1: source-native crop representation is sufficient | A no-leak, preregistered finite-library reference or fixed crop is non-inferior to the same-runtime dense comparator on Avg-mAP, mAP@0.7, short actions, and boundary error, with measured representation-path cost headroom |
| C2: task-aware learned selection is useful | A deployable selector trained without inference-time GT beats fixed/random crops, recovers registered S2 headroom, preserves the C1 accuracy constraints, and remains cheaper after selector cost |

The required anti-claims are:

- gains are not full-frame downsampling;
- gains are not explained by an easier training schedule or extra compute;
- a fixed center/random crop is not already sufficient when claiming adaptive
  selection;
- reference GT or cached selection does not enter deployable inference;
- theoretical pixels/FLOPs are not substituted for measured full-stack cost.

## Current Position

| Stage | Object | State | What it proves | Paper role |
|---|---|---|---|---|
| R0 | Dense160/224/256 full-frame resize | historical diagnostic | sensitivity to whole-frame resolution only | control/appendix; never crop evidence |
| S1 | Native-Crop vertical slice and CUDA Gate `1174671` | `tested` | source-coordinate crop, shared backbone, detector contract, gradients, provenance, and no-leak are executable | infrastructure evidence |
| S2-P | Crop-sufficiency preregistration v1.1 | **current stage: `designed`, revision required** | only defines a valid experiment and decision semantics | prerequisite, not an empirical result |
| S2-E | Formal crop-sufficiency experiment | blocked | whether registered native crops are sufficient; whether adaptive headroom and cost headroom exist | mechanism/oracle diagnostic |
| S3 | Learned Native-Crop policy | blocked by S2 | whether a deployable selector can recover S2 headroom through TAD supervision | final method candidate |
| S4 | Primary official benchmark | blocked by S3 | final method accuracy-cost benefit on THUMOS14 official test | main paper Table 1/Figure 1 |
| S5 | Detector/dataset generalization and ablations | blocked by S4 anchor | robustness, mechanism, simplicity, and transfer | main Tables 2-3/appendix |
| S6 | Statistical, cost, and claim freeze | blocked by exact evidence closure | whether the complete claim is paper-ready | final paper gate |

The current S2-P stage has no training job and no crop mAP. It is not the
paper's main experiment.

## What S2 Must Establish

S2 is a falsification and sufficiency gate. It must answer three questions
separately on development fit/gate:

1. **Crop sufficiency:** can a registered source-native crop representation
   remain equivalent to the same-runtime dense baseline?
2. **Adaptive headroom:** is there a measurable gap between fixed/random crops
   and a GT-visible heuristic reference that justifies learning a selector?
3. **Cost viability:** does one global plus one local representation leave
   enough measured latency/energy headroom after reserving selector cost?

S2 cannot prove:

- a learned selector can be optimized;
- a deployable method beats dense TAD;
- official-test performance;
- detector or dataset generality;
- SOTA;
- continuous-crop optimality.

### S2 Decision Routing

| S2 state | Authorized next action |
|---|---|
| `SUFFICIENT_AND_POLICY_HEADROOM` | Freeze and implement S3 learned-policy protocol |
| `SUFFICIENT_FIXED_CROP_ONLY` | Do not build an adaptive selector; assess a simpler fixed global-local paper only |
| `SUFFICIENT_BUT_COST_NOT_VIABLE` | Stop the efficiency claim; do not hide cost with FLOPs |
| `REFERENCE_RULE_INSUFFICIENT` | No learned-policy authorization; improve the reference proof only under a new preregistration |
| `INCONCLUSIVE_GEOMETRY_OR_SUPPORT` | Repair geometry/support evidence without interpreting model performance |
| `NO_DECISION_INVALID_EVIDENCE` | Repair infrastructure and rerun in a new immutable campaign |

## End-To-End Stages

### P0: Freeze S2 v1.1

- Correct reference semantics, outcome states, GT/cache order, uncertainty
  units, coverage terminology, selector-cost reserve, and result-blind power
  audit.
- Validate every proposed config, script, statistic, and Slurm dependency
  against the current immutable Git tree.
- Output: one versioned protocol, one static validator, one no-GPU known-answer
  test suite.
- GO: `V1.1_READY_FOR_IMPLEMENTATION`.
- KILL/HOLD: any unresolved P0/P1 or any numeric decision left as TBD.

### P1: Implement And Run S2 Sufficiency

- Implement only the registered dense, fixed/random, and finite-library
  sufficiency cells.
- Keep official test closed.
- Run three seeds and the registered fit/gate inference, detached reference
  join, detection statistics, and ABBA cost profile.
- First report every raw seed/cell result, then corrected intervals and the
  mechanical S2 state.
- Output: S2 evidence commit and exactly one decision receipt.
- GO to S3 only on `SUFFICIENT_AND_POLICY_HEADROOM`.

The v1 draft proposed 21 training jobs. The final job count is not authorized
until v1.1 freezes the training-distribution control and decision-critical
matrix.

### P2: Freeze And Implement The Learned Policy

This stage starts only after S2 policy headroom passes.

The final method contract must be:

```text
decoded source frames
-> low-cost global view / scout
-> runtime crop decision without GT, teacher, or prediction cache
-> source-coordinate local crop
-> registered shared/global-local representation
-> TAD detector
-> joint detector-aware training
```

Development discipline:

- derive an inner training/calibration split from the S2 fit population;
- use only that inner split for architecture and hyperparameter work;
- cap exploratory variants before the held-out Gate is touched;
- freeze the selector, loss weights, crop budget, schedule, checkpoint rule,
  and cost reserve before one formal Gate evaluation;
- inference must generate crop decisions at runtime;
- selector cost is included in every final profile.

The S3 formal comparison must include:

- learned policy;
- center crop;
- deterministic random/balanced crop;
- fit-selected best fixed crop;
- global-only representation;
- same-runtime dense comparator.

GT-visible heuristic results remain development diagnostics and never become a
deployable official-test row.

### P3: Primary THUMOS14 Official Experiment

After S3 Gate passes:

1. Freeze the method and all thresholds.
2. Retrain the final method and core baselines on the complete 200-video
   development training population.
3. Open the 211-video official test once under an immutable certificate.
4. Run three seeds for the learned method and key matched baselines.

**Main Table 1: Accuracy And Cost**

Required rows:

- same-runtime dense160;
- dense224/dense256 accuracy ceilings;
- global-only;
- center local crop;
- deterministic random crop;
- fit-selected best fixed crop;
- final learned Native-Crop method.

Required columns:

- Avg-mAP and mAP@0.3/0.4/0.5/0.6/0.7;
- short-action recall at high tIoU;
- start/end boundary error;
- warm-serial full-stack p50/p95 latency;
- peak GPU memory;
- gross GPU energy;
- measured selector cost;
- parameters and FLOPs as secondary descriptors only.

**Main Figure 1: Accuracy-Cost Pareto**

- At least three crop-budget operating points must be frozen before official
  test.
- All points include decode, preprocessing, H2D, scout, crop, backbone,
  detector, post-processing, and selector overhead.
- A point is Pareto-valid only if its matched accuracy constraints pass.

### P4: Generalization

Paper-ready evidence requires both a second detector head and a second dataset.

**Main Table 2A: Detector generality on THUMOS14**

- primary: AdaTAD-derived ActionFormerHead;
- secondary: TriDetHead;
- optional appendix: TemporalMaxerHead.

Use the same crop policy interface and spatial budgets. Any head-specific
retraining must be disclosed; cached decisions from another head cannot be
called task-aware inference.

**Main Table 2B: Dataset generality**

- target: ActivityNet-1.3;
- first run an auditable raw-video availability and geometry census;
- if the frozen completeness gate fails, select FineAction under a new
  pre-result dataset amendment rather than silently using an incomplete
  ActivityNet subset;
- report the dataset's official temporal localization metrics plus the same
  full-stack cost schema.

Published dense-method numbers may provide context, but direct cost claims use
locally reproduced, matched-runtime baselines.

### P5: Mechanism And Simplicity Ablations

**Main Table 3**

Must isolate only the claims needed by the final method:

- source-coordinate native crop versus whole-frame resize at matched input
  cost;
- learned versus center/random/best-fixed selection;
- global context removed;
- local view removed;
- detector-gradient path removed;
- selector auxiliary/reference supervision removed, if present in the frozen
  method;
- selector overhead included versus representation-only cost accounting.

Do not add ablations for components that are absent from the final method.

**Mechanism diagnostics**

- crop-position distribution and entropy;
- per-class and short-action effects;
- start/end boundary errors;
- selected region trajectories on representative successes and failures;
- camera-motion/background-motion failure cases;
- stage-wise latency and energy breakdown.

Qualitative figures are explanatory evidence, never substitutes for paired
metrics.

### P6: Paper Evidence Freeze

Paper status can become `paper_ready` only when:

- S2 has one valid mechanical decision;
- the final deployable method beats fixed/random controls under its registered
  accuracy-cost criterion;
- primary official-test results have complete three-seed evidence;
- measured end-to-end cost includes selector overhead;
- one second head and one second dataset support the same minimal claim;
- main ablations isolate crop selection rather than resolution reduction or
  extra compute;
- all tables trace to immutable checkpoints, predictions, configs, hashes,
  evaluator receipts, and profile traces;
- claims are limited to the evidence actually closed.

## Paper Experiment Blocks

| Block | Claim defended | Evidence | Placement | Priority |
|---|---|---|---|---|
| B0 S2 sufficiency | Native crop has representational and selection headroom | development-only finite-library/fixed-reference study | motivation or appendix | must run before method |
| B1 primary result | learned crop improves measured accuracy-cost tradeoff | THUMOS14 official test, three seeds | Main Table 1 | must run |
| B2 Pareto/cost | benefit survives real system accounting | at least three budgets, full-stack ABBA profile | Main Figure 1 | must run |
| B3 generality | method is not tied to one head/dataset | TriDet plus ActivityNet-1.3 or preregistered FineAction fallback | Main Table 2 | must run for paper-ready |
| B4 mechanism | gain comes from task-aware crop selection | minimal deletion/baseline ablations and boundary diagnostics | Main Table 3/Figure 2 | must run |

## Execution Order And Stop Rules

| Order | Action | Stop rule |
|---:|---|---|
| 1 | Obtain and freeze S2 v1.1 | stop on unresolved protocol P0/P1 |
| 2 | Implement static tests and one CUDA Gate | stop on leakage, gradient, evaluator, or cost-schema mismatch |
| 3 | Run formal S2 matrix | only `SUFFICIENT_AND_POLICY_HEADROOM` unlocks S3 |
| 4 | Freeze S3 learned-policy protocol | stop if policy consumes GT/cache or selector cost erases headroom |
| 5 | Run inner-fit development and one S3 Gate | stop if learned policy does not beat fixed/random under registered intervals |
| 6 | Freeze and open THUMOS14 official test once | stop paper claim if primary accuracy-cost gate fails |
| 7 | Run second head, second dataset, and decisive ablations | remain non-paper-ready if generality or mechanism claim fails |
| 8 | Seal result-to-claim map and write paper | no new tuning after evidence freeze |

## Current Unique Next Step

Obtain a corrected, code-audited S2 Preregistration v1.1. Do not implement the
S2 matrix, learned selector, official-test evaluation, cross-head integration,
or cross-dataset experiment before that protocol is frozen.
