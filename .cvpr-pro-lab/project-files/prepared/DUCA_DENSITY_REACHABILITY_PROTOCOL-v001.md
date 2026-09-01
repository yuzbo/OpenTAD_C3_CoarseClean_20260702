---
doc_id: DUCA_DENSITY_REACHABILITY_PROTOCOL
version: v001
status: FROZEN_AUTHOR_ONLY_NOT_EXECUTED
date: 2026-08-14
author_role: evaluator
evaluator_process_id: 019febf2-690b-7093-bcf5-8eb69636770e
project_id: g-p-6a796fef9a00819194024cf1de3bd697
parent_decision: PRO_DUCA_DENSITY_REACHABILITY_DECISION-v001
parent_decision_status: accepted_local
parent_scientific_decision: REVISE
pro_nonce: DUCA-ARIS-DENSITY-v001-20260813T222019Z
protocol_revision: a6bdc084cc145c80b6b2c68d0a38f0deea3e8518
evidence_class: TRAINING_POPULATION_MECHANISM_REACHABILITY_PROTOCOL_ONLY
execution_state: NOT_EXECUTED
pre_run_state: NOT_READY
official_validation_state: INACCESSIBLE_FORBIDDEN
---

# DUCA density reachability protocol v001

## 1. Purpose and claim boundary

This document freezes the single training-population-only, video-disjoint
`FIT/CAL` U/O/R reachability gate authorized by
`PRO_DUCA_DENSITY_REACHABILITY_DECISION-v001`. It is an author-only protocol
specification. It does not implement, launch, or evaluate the gate.

The sole active question is mechanism reachability:

> On detector-unseen training-side calibration videos, can a deploy-visible
> density-only reader produce bounded `K=384` physical acquisition that improves
> the same frozen detector over canonical exact-uniform sampling without a
> material loss at tIoU 0.6 or 0.7?

This gate is not an official-validation result, held-out result, deployability
proof, final training objective, benchmark result, cost result, or paper claim.
An `ADVANCE` outcome authorizes only consideration of later full learned-density
training under a new decision and PRE_RUN. It does not establish efficacy or
paper readiness.

The existing density decoder/reader work is an untracked dirty-worktree
prototype with compile/fixture and incomplete geometry-probe evidence only. It
is inadmissible as production code, PRE_RUN evidence, evaluator evidence, or an
experiment result. This protocol does not promote, execute, import, test, or
otherwise validate that prototype.

## 2. Non-negotiable population firewall

### 2.1 Allowed population

Only the official **training population** may participate. Before any training,
checkpoint creation, inference, or metric access, a deterministic immutable
manifest must partition that population by whole video into exactly two sets:

- `FIT`: detector fitting and reader fitting only;
- `CAL`: one reachability evaluation only.

The manifest must be generated without metric, detector prediction, duration,
class-frequency, boundary-density, difficulty, or any other result-dependent
selection. Its construction algorithm, input training-population identity,
partition parameters, deterministic seed, exact ordered video IDs, and one
digest per immutable manifest artifact must be frozen before data access by any
gate process.

Every training-population video must occur in exactly one partition. The video
ID intersection `FIT ∩ CAL` must be empty. Windows, clips, augmentations, cached
features, annotations, predictions, and derivatives inherit their source
video's partition; no derivative may cross the firewall.

No `CAL` video or derivative may enter:

- detector fitting, checkpoint choice, early stopping, or hyperparameter choice;
- reader fitting, loss choice, target choice, checkpoint choice, or stopping;
- normalization/statistics fitting that affects U/O/R inference;
- threshold, oracle-kernel, split, seed, wrapper, NMS, evaluator, or decision-rule
  selection.

`CAL` is consumed exactly once after the full prelaunch package passes. No
resplit, repair, exclusion, addition, rerun, or post-output tuning is permitted.

### 2.2 Official-validation deny boundary

Official validation is inaccessible. It must not be mounted, listed, read,
copied, linked, constructed, inferred from a manifest, evaluated, or used to
derive any target, checkpoint, statistic, threshold, route decision, or claim.

The future prelaunch package must contain a validation-deny receipt that binds:

- the allowed training-population roots and exact FIT/CAL manifest;
- the literal launch mounts and process-visible data roots;
- an explicit absence of every official-validation root, manifest entry,
  annotation, video, feature, prediction, cache, and derivative;
- zero official-validation access by detector FIT, reader FIT, U/O/R inference,
  evaluator, bootstrap, and postprocessing.

Any official-validation accessibility or uncertainty is
`REACHABILITY_PRE_RUN_BLOCKED`. Any access after launch makes the entire package
`INADMISSIBLE_LEAKAGE`; it produces no `ADVANCE`, `HOLD`, or `KILL` conclusion.

## 3. Frozen detector and reader fitting order

The fitting order is exact:

1. Train one canonical exact-uniform `K=384` detector on `FIT` only through the
   tracked production wrapper.
2. Select the terminal checkpoint only. Best-checkpoint selection,
   `CAL`-driven stopping, and metric-driven checkpoint choice are forbidden.
3. Freeze the detector checkpoint and all detector parameters, buffers,
   preprocessing, coordinate mapping, head, filtering, NMS, class map, and
   evaluator identities.
4. Fit the density reader on `FIT` only while the detector remains frozen and
   detector gradients are disabled.
5. Select the reader terminal checkpoint only, without reading any `CAL` output.
6. Freeze the reader and evaluate U/O/R once on the same `CAL` manifest.

The historical uniform checkpoint is inadmissible unless a complete immutable
receipt proves exact compatibility in code, config, coordinate semantics,
training population, terminal-checkpoint rule, wrapper, and evaluator identity.
Absent that receipt, a new FIT-only terminal detector checkpoint is mandatory.

## 4. Frozen U/O/R arms

All three arms use the same tracked production code and identical CAL videos,
ordered windows, valid masks, preprocessing, effective-`K` sequence, frozen
detector checkpoint, density-to-mass conversion, inverse-CDF path, frozen integer
projection, selected-q-to-physical-time mapping, detector, filtering, top-k,
IoU, NMS, class map, evaluator, and output schema. They may differ only in the
source of the density logits supplied to the common decoder.

Exactly-once selected-q-to-physical-time transport must occur before filtering,
top-k, IoU, or NMS. Selected-rank coordinates must never be passed as physical
time. Every arm must record requested `K`, effective `K`, unique `K`, padded
detector `K`, and actual backbone input length for every evaluated window.

### 4.1 U — exact-uniform control

`U` replaces density logits with an exactly constant valid-prefix vector and
runs it through the canonical production density decoder. Its projected
positions must be bit-identical to the canonical exact-uniform `K=384` decoder
for every admitted shape and mask.

The reader/wrapper path remains otherwise present so that U and R do not differ
in preprocessing or detector execution. U may not use GT, teacher output,
cached prediction, detector result, or adaptive logits.

### 4.2 O — privileged boundary diagnostic

For each valid physical time index `t`, let `B` be the physical GT start/end
boundaries in the current valid window, clipped to that window. The sole frozen
target is:

~~~text
h_t = max_{b in B} max(0, 1 - |t - b| / 16)
rho_gt(t) = 1 + h_t
~~~

`O` converts this exact density to the same production logits/mass
representation and then uses the common decoder and wrapper. No alternate
kernel, width, amplitude, clipping rule, repair, or post-output target change is
allowed.

O is `PRIVILEGED_GT_BOUNDARY_DIAGNOSTIC_ONLY`. It is not deploy-visible, not a
learned arm, not an upper bound over all acquisition policies, and never by
itself supports a learned-density or deployment claim.

### 4.3 R — deploy-visible reader

`R` obtains `duca_density_logits[B,T]` only from valid-prefix
`browser_memory`. At R inference, the reader and every transitive input must be
free of:

- GT labels, segments, starts, ends, boundaries, density targets, or derivatives;
- teacher outputs or teacher features;
- cached/raw detector predictions, logits, proposals, scores, or results;
- evaluator/NMS outputs, metrics, oracle choices, or counterfactual ledgers;
- checkpoint-selection signals or CAL-derived statistics.

R is fitted on `FIT` only to match the normalized trapezoidal mass implied by
the frozen `rho_gt`. The detector is frozen and detector gradients are disabled.
This supervision is a mechanism diagnostic and is not authorized as the final
paper-training objective.

`O-U` measures privileged boundary-aligned headroom. `R-U` measures whether
deploy-visible evidence reaches useful density behavior. O cannot substitute
for R.

## 5. Prelaunch freeze and audit package

Before any Builder integration, data access, training, inference, or metric
process, the following package must exist, be immutable, and pass independent
Critic review:

1. **Authority seal:** this protocol, the accepted Pro decision, project ID,
   nonce, role identities, and authorized evidence class.
2. **Clean tracked implementation seal:** one tracked production patch based on
   clean revision `a6bdc084cc145c80b6b2c68d0a38f0deea3e8518` (or an explicitly
   admitted descendant), with no dependence on the untracked prototype.
3. **FIT/CAL split seal:** source training-population identity, deterministic
   construction rule, parameters/seed, exact ordered membership, video-disjoint
   proof, counts, and one digest per manifest artifact.
4. **Validation-deny seal:** exact allowed mounts/manifests and proof that
   official validation is absent and inaccessible to every phase.
5. **Detector FIT seal:** exact config/argv/environment/resources, FIT-only
   membership, terminal-checkpoint rule, final checkpoint identity, and frozen
   detector state.
6. **Reader FIT seal:** exact reader source/config/loss/argv, FIT-only
   membership, normalized trapezoidal-mass target, frozen-detector and
   zero-detector-gradient attestations, terminal reader checkpoint identity.
7. **U/O/R tuple seal:** literal arm identities and the exact only-permitted
   difference in density-logit source; constant-logit bit identity with canonical
   uniform; exact oracle target; R deploy-visible input allowlist and forbidden
   source denials.
8. **Fair-wrapper seal:** identical CAL videos, window order, masks,
   preprocessing, requested/effective/unique/padded K, backbone input,
   checkpoint, coordinate mapping, physical-time transport, filtering, top-k,
   IoU, NMS, class map, evaluator, and output schema.
9. **Evaluator seal:** official evaluator source/revision, config, tIoU grid
   `0.3:0.1:0.7`, class map, NMS identity, score/tie behavior, pooled-prediction
   semantics, and bootstrap implementation identity.
10. **Statistic seal:** nonce-derived seed algorithm, RNG algorithm, 10,000
    paired video-cluster resamples, percentile-bound rule, MWE, all ADVANCE/KILL/
    HOLD predicates, and no per-video-AP averaging.
11. **Launch seal:** literal argv, cwd, environment, mounts, output root,
    resource request, phase order, first-fail rule, and zero access to forbidden
    surfaces.
12. **Output seal:** absent fresh output root, exclusive creation, per-phase
    completion receipts, metric embargo, final atomic publication, and no reuse,
    append, overwrite, repair, or rerun.

The production package must include identical U/O/R evaluation entry points,
one exact-uniform detector-FIT config, one frozen-detector reader-FIT config, one
three-arm CAL evaluator config, and exact future launchers. Production code,
configs, launchers, and tests are Builder-owned and are not created by this
protocol.

The independent Critic must verify at minimum:

- no CAL video or derivative entered detector or reader fitting;
- no official-validation surface was accessible;
- GT and every forbidden signal are absent from R inference;
- U/O/R differ only in density-logit source;
- all arms share effective-K sequences and physical-time pre-NMS mapping;
- the bootstrap recomputes dataset-level official AP on pooled predictions;
- no result-dependent change is technically possible;
- the production projector has a durable identity/optimality receipt at the
  current tracked patch identity.

Critic returns exactly `REACHABILITY_PRE_RUN_PASS` or
`REACHABILITY_PRE_RUN_BLOCKED`. Only a complete PASS plus a later explicit
Coordinator PRE_RUN admission can authorize execution.

## 6. Output seal and result embargo

The future run uses one new literal output root that is absent at PRE_RUN. The
owner creates it exclusively. No prior output, partial tree, prediction, metric,
or receipt may be reused.

The execution order is frozen as detector FIT, detector freeze, reader FIT,
reader freeze, CAL U, CAL O, CAL R, official evaluation/bootstrap, decision. A
phase starts only after its predecessor's required immutable receipt exists.

Raw U/O/R predictions are written under separate arm roots with identical
schemas. Each arm root is sealed non-writable after completion. No arm metric,
cross-arm difference, bootstrap summary, or decision statistic may be displayed,
read, exported, or used for tuning until:

- all three arm prediction roots are complete and sealed;
- all phase receipts and forbidden-access attestations are complete;
- split, checkpoint, coordinate, effective-K, NMS, class-map, and evaluator
  identities agree across arms;
- the metric process receives all three sealed arm roots simultaneously.

The metric process reads only the complete sealed package, runs the unchanged
official evaluator and frozen bootstrap once, writes one final result receipt,
seals it, and atomically publishes the final result root. The first mismatch or
failure blocks publication and records one first-failure field. There is no
cleanup, retry, partial-result interpretation, threshold change, split change,
checkpoint change, oracle change, arm omission, repair, or rerun under the same
identity.

## 7. Frozen statistic

### 7.1 Point estimates

On the full immutable CAL set, run the official evaluator on each arm's pooled
predictions. Let `Avg-mAP` be the official mean over tIoU thresholds
`0.3, 0.4, 0.5, 0.6, 0.7`.

Report percentage-point paired differences:

~~~text
Delta_OU_Avg = Avg-mAP(O) - Avg-mAP(U)
Delta_RU_Avg = Avg-mAP(R) - Avg-mAP(U)
Delta_RU_06  = mAP@0.6(R) - mAP@0.6(U)
Delta_RU_07  = mAP@0.7(R) - mAP@0.7(U)
~~~

No per-video AP or average of per-video AP is an admitted statistic.

### 7.2 Deterministic seed

The bootstrap seed is derived only from the Pro nonce
`DUCA-ARIS-DENSITY-v001-20260813T222019Z`:

~~~text
seed_material = UTF8("DUCA_DENSITY_REACHABILITY_BOOTSTRAP_V1\n" + pro_nonce)
seed_digest   = SHA256(seed_material)
seed_u64      = unsigned_big_endian(seed_digest[0:8])
rng           = PCG64(seed_u64)
~~~

The nonce, prefix, UTF-8 encoding, newline, byte order, digest slice, integer,
and RNG algorithm are immutable. No result, clock, host, process, or run ID may
affect the seed.

### 7.3 Paired video-cluster bootstrap

Use exactly 10,000 resamples. If CAL contains `N` videos, each resample draws
exactly `N` video indices with replacement from the ordered CAL video list using
the frozen RNG. The same index multiset and multiplicities apply to U, O, R,
annotations, windows, and all evaluator inputs.

The video is the indivisible cluster: every prediction and annotation belonging
to a sampled video is retained together. A video selected multiple times is
duplicated with matching prediction/annotation multiplicity under a unique
in-memory bootstrap namespace so the unchanged evaluator treats copies as
distinct sampled clusters without changing scores, labels, segments, or class
IDs.

For every resample, recompute the official evaluator from the resampled **pooled
predictions** for U, O, and R. Recompute all four admitted paired differences.
Never compute AP per video and never average per-video AP values.

For each difference, sort the 10,000 bootstrap values ascending. Freeze the
one-sided 95% percentile bounds as nearest-rank order statistics:

~~~text
LCB95 = sorted_value_at_1_based_rank(500)
UCB95 = sorted_value_at_1_based_rank(9500)
~~~

Report point estimate, `LCB95`, and `UCB95` in percentage points. No alternate
confidence method, interpolation, studentization, seed, resample count, or
multiple-comparison adjustment may be introduced after any CAL output is read.

## 8. Frozen decision rule

Minimum worthwhile effect:

~~~text
MWE = +0.50 percentage points
high_tIoU_floor = -0.20 percentage points
~~~

Before applying scientific thresholds, every authority, population, split,
leakage, identity, checkpoint, arm, coordinate, effective-K, metric, bootstrap,
and output seal must pass. A missing, failed, or uncertain seal yields
`REACHABILITY_EVIDENCE_INADMISSIBLE` / `PRE_RUN_BLOCKED`, not ADVANCE, HOLD, or
KILL.

### 8.1 ADVANCE

Return `ADVANCE_TO_CONSIDERATION_OF_FULL_LEARNED_DENSITY_TRAINING` only if all
conditions hold simultaneously:

- `Delta_OU_Avg >= +0.50 pp`;
- `LCB95(Delta_OU_Avg) > 0`;
- `Delta_RU_Avg >= +0.50 pp`;
- `LCB95(Delta_RU_Avg) > 0`;
- `Delta_RU_06 >= 0`;
- `LCB95(Delta_RU_06) > -0.20 pp`;
- `Delta_RU_07 >= 0`;
- `LCB95(Delta_RU_07) > -0.20 pp`;
- every seal in this protocol passes.

### 8.2 KILL

Return `KILL_BOUNDARY_FOCUSED_DENSITY_ACQUISITION` if any admitted predicate is
true:

1. `UCB95(Delta_OU_Avg) < +0.50 pp`; or
2. after O passes its headroom gate
   (`Delta_OU_Avg >= +0.50 pp` and `LCB95(Delta_OU_Avg) > 0`),
   `UCB95(Delta_RU_Avg) < +0.50 pp`; or
3. `UCB95(Delta_RU_07) < -0.20 pp`.

### 8.3 HOLD

Every other admissible outcome is `HOLD`. HOLD is not KILL and not ADVANCE. A
nonpositive lower bound alone is never a KILL predicate. No threshold, oracle
shape, split, seed, checkpoint, statistic, arm, exclusion, or interpretation may
change after any CAL output is read.

## 9. Evidence-class labels

The following labels are mandatory and non-interchangeable:

- This document:
  `TRAINING_POPULATION_MECHANISM_REACHABILITY_PROTOCOL_ONLY` /
  `AUTHORED_NOT_EXECUTED`.
- Current untracked prototype and prior compile/fixture/geometry probes:
  `INADMISSIBLE_PROTOTYPE_INFRASTRUCTURE_EVIDENCE_ONLY`.
- U and R CAL results after all seals:
  `TRAINING_POPULATION_CAL_MECHANISM_REACHABILITY_EVIDENCE`.
- O and O-U:
  `PRIVILEGED_GT_BOUNDARY_DIAGNOSTIC_ONLY`.
- An output with any firewall, identity, fairness, or seal failure:
  `REACHABILITY_EVIDENCE_INADMISSIBLE`.
- Any official-validation access:
  `INADMISSIBLE_LEAKAGE_QUARANTINED`.
- ADVANCE:
  `MECHANISM_REACHABILITY_GATE_ADVANCE_ONLY`, never paper evidence.
- HOLD:
  `MECHANISM_REACHABILITY_GATE_HOLD_ONLY`.
- KILL:
  `BOUNDARY_FOCUSED_DENSITY_ROUTE_FALSIFIED_BY_FROZEN_GATE`.

No label authorizes official-validation access, P1, training expansion, paper
claims, deployment claims, or result promotion without a new Pro decision.

## 10. Required receipt schema

Every future machine-readable receipt must be canonical JSON and include the
fields below. Additional descriptive fields may not change their meaning.

~~~text
schema_version: "duca-density-reachability-receipt-v001"
artifact_status: PRE_RUN_BLOCKED | RUNNING | COMPLETE | FAILED | INADMISSIBLE
execution_state: NOT_EXECUTED | EXECUTED
evidence_class: one literal label from section 9
project_id
parent_decision
protocol_id
protocol_version
pro_nonce
role
phase
code_revision
clean_binding_status
tracked_patch_status
untracked_prototype_used: false

population:
  source_training_population_id
  fit_cal_manifest_path
  fit_cal_manifest_digest
  split_construction_rule
  split_seed
  fit_video_ids
  cal_video_ids
  video_disjoint: true
  cal_used_for_fit_or_tuning: false
  official_validation_mounted: false
  official_validation_manifested: false
  official_validation_access_count: 0

detector_fit:
  config_id
  argv
  fit_only: true
  terminal_checkpoint_only: true
  checkpoint_id
  checkpoint_frozen_for_reader_and_cal: true

reader_fit:
  config_id
  argv
  fit_only: true
  target_id: "rho_gt_v001_normalized_trapezoidal_mass"
  detector_frozen: true
  detector_gradients_enabled: false
  terminal_checkpoint_only: true
  checkpoint_id

arm_freeze:
  arm_order: [U, O, R]
  sole_difference: density_logit_source
  U_source: constant_logits_canonical_exact_uniform
  O_source: privileged_gt_boundary_rho_gt_v001
  O_evidence_class: PRIVILEGED_GT_BOUNDARY_DIAGNOSTIC_ONLY
  R_source: deploy_visible_browser_memory_only
  R_gt_access: false
  R_teacher_access: false
  R_cached_prediction_access: false
  R_detector_result_access: false
  identical_fair_tuple: true

fair_tuple:
  ordered_cal_videos
  ordered_windows
  masks
  requested_effective_unique_padded_K
  actual_backbone_input
  detector_checkpoint_id
  preprocessing_id
  density_decoder_id
  projector_identity_receipt
  physical_time_mapping_id
  nms_id
  class_map_id
  official_evaluator_id

bootstrap:
  resamples: 10000
  cluster_unit: video
  paired: true
  pooled_official_evaluator_recomputed_each_resample: true
  per_video_AP_averaged: false
  seed_material
  seed_digest
  seed_u64
  rng: PCG64
  bound_rule: nearest_rank_500_9500_of_10000

results:
  Delta_OU_Avg: {point_pp, LCB95_pp, UCB95_pp}
  Delta_RU_Avg: {point_pp, LCB95_pp, UCB95_pp}
  Delta_RU_06:  {point_pp, LCB95_pp, UCB95_pp}
  Delta_RU_07:  {point_pp, LCB95_pp, UCB95_pp}

seals:
  authority
  split
  validation_deny
  detector_fit
  reader_fit
  arm_tuple
  fair_wrapper
  coordinate
  effective_K
  evaluator
  statistic
  output
  metric_embargo

decision: ADVANCE | HOLD | KILL | NOT_APPLIED
decision_rule_version: "DUCA_DENSITY_REACHABILITY_PROTOCOL-v001"
first_failure: null | {phase, field, expected, observed, disposition}
scope_deviation: "none" | literal_description
output_paths
started_at
completed_at
exit_status
~~~

Before execution, receipts must state `execution_state="NOT_EXECUTED"`, omit
result numbers, set `decision="NOT_APPLIED"`, and keep
`artifact_status="PRE_RUN_BLOCKED"` until the later Coordinator admission. A
receipt that claims metrics or a scientific decision while `NOT_EXECUTED` is
invalid.

## 11. Stop and ownership rules

Any missing authority, dirty binding, untracked production dependency,
population overlap, official-validation accessibility, CAL contamination,
checkpoint drift, arm asymmetry, GT/teacher/cache/result access by R, coordinate
mismatch, effective-K mismatch, evaluator drift, output reuse, metric embargo
breach, or result-dependent change blocks the gate at the first failure.

There is no repair, fallback, best-run selection, arm deletion, threshold tuning,
resampling change, or rerun under the same frozen identity. The Coordinator
routes deterministic implementation work but does not reinterpret the protocol.
Builder owns the future tracked production/config/launcher package. Critic owns
independent PRE_RUN review. Evaluator owns only the preregistered evaluation
after a later admission. Pro alone decides any scientific revision or later
route.

## 12. Current authoring receipt

~~~text
artifact: .cvpr-pro-lab/evaluator-returns/DUCA_DENSITY_REACHABILITY_PROTOCOL-v001.md
author_role: evaluator
evaluator_process_id: 019febf2-690b-7093-bcf5-8eb69636770e
canonical_workspace: C:/Users/skywalker/.codex/worktrees/30f3/OpenTAD_C3_CoarseClean_20260702-eval-a6bdc084
HEAD: a6bdc084cc145c80b6b2c68d0a38f0deea3e8518
pre_authoring_binding: CLEAN_EVALUATION_ONLY_PASS
task_class: AUTHOR_ONLY_PROTOCOL_FREEZE
execution_state: NOT_EXECUTED
production_code_or_config_changed: false
test_or_python_invoked: false
fixture_or_data_accessed: false
official_validation_accessed: false
remote_or_slurm_or_gpu_used: false
metric_or_result_computed: false
browser_or_sources_or_pro_used: false
untracked_prototype_promoted: false
evidence_state: BLOCKED_PRE_RESULT
pre_run_state: NOT_READY
scope_deviation: none
~~~

EVALUATOR_DECISION:
`DUCA_DENSITY_REACHABILITY_PROTOCOL_V001_FROZEN_AUTHORED_NOT_EXECUTED`.
