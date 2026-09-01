---
doc_id: EVALUATOR_DUCA_DENSITY_P0P1
version: v001
status: designed_not_executed
date: 2026-08-11
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260811T050602Z-5ea4bb06fd29
parent_artifact_id: art-20260811T045704Z-2d13c25e41
parent_decision: PRO_INITIAL_REVIEW-v002
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: preparatory_preregistration
---

# Evaluator DUCA density P0/P1 preregistration

## 1. Authority, boundary, and receipt

This artifact executes only durable queue message
`msg-20260811T050602Z-5ea4bb06fd29`. Its sealed parent is
`art-20260811T045704Z-2d13c25e41`, the accepted
`PRO_INITIAL_REVIEW-v002` decision at Git revision
`63a726a4aaf48ecbf6780bb196de43a890c6b4df`.

Authoritative raw Pro source:
`.cvpr-pro-lab/pro-reviews/runs/duca-serial-pro-5ad209b3c128fbdd4455da0a7783b203/raw-response.md`.

No test, model code, dataset, metric, GPU, remote host, or Slurm job was run.
No held-out file, prediction, checkpoint result, or metric was accessed. No
training code/config was changed, no route was frozen, and no Git action was
taken. All checks were performed personally by the registered Evaluator task;
`extra_processes: none`, `subagents: none`, `probes: none`.

Current verdict:

- P0/P1 witness contract: `DESIGNED_NOT_EXECUTED`.
- P0 gate: `BLOCKED` by unresolved canonical-uniform identity and current
  post-NMS inverse mapping.
- P1 gate: `NOT_RUN` under the current no-local-CPU boundary.
- P2 `PRE_RUN_READY`: `BLOCKED`; a fresh concrete Pro instruction, human
  authorization, passed P0/P1 receipts, and the fields in Section 7 are required.
- Scientific/model evidence: unchanged at `BLOCKED_PRE_RESULT`.

## 2. Frozen-snapshot static findings

These are repository facts, not experimental results.

1. The intended clean detector truth is rooted at
   `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` and
   inherits `configs/_base_/models/actionformer.py`. Its NMS is soft-NMS with
   `sigma=0.7`, `max_seg_num=2000`, `multiclass=true`, and
   `voting_thresh=0.7` (`e2e_thumos_videomae_s_768x1_160_adapter.py:139-146`).
2. The existing file named exact-uniform is not an admissible clean baseline:
   `pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py:28-30`
   declares `input_sampling_plus_head_temporal_geometry_control`.
3. The two current uniform constructors are statically inconsistent for the
   normal `T=768, K=384` case. `LoadFrames._exact_uniform_dense_positions`
   uses `round(j*T/K)`, ending at 766
   (`opentad/datasets/transforms/end_to_end.py:337-348`), whereas
   `_uniform_anchor_positions` uses rounded `linspace(0,T-1,K)`, ending at 767
   (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:3158-3190`).
   Therefore clean/wrapper uniform identity is presently unproved and, under
   endpoint preservation, fails closed until one shared canonical generator is
   accepted.
4. Current generic non-sliding post-processing invokes `batched_nms` before
   `convert_to_seconds` (`opentad/models/detectors/single_stage.py:138-145`).
   The selected-to-dense inverse map occurs inside `convert_to_seconds`
   (`opentad/models/utils/post_processing/utils.py:73-117`). Thus this path does
   not satisfy “raw proposal inverse mapping before every NMS.”

## 3. Coordinate and budget conventions

The property harness must use one serialized canonical fixture for every
supported `(T_v, K_eff)` pair. Until Question Q1 is answered, no implementation
may silently choose a rounding/tie rule.

- Valid physical frame indices: integers `[0, T_v-1]`.
- Requested normal-window budget: `requested_K=384`.
- Effective budget:
  `K_eff=min(384, 16*floor(T_v/16))`; `T_v<16` fails closed.
- Selected positions: `p=(p_0,...,p_{K_eff-1})`.
- Canonical uniform positions: `u=canonical_uniform(T_v,K_eff)` from the single
  accepted generator/fixture.
- Segment boundary coordinates are provisionally end-exclusive on `[0,T_v]`;
  the terminal knot is `(q=K_eff,t=T_v)`. This convention is not executable
  until Question Q3 is answered.
- `max_unselected_run=max_j(p_{j+1}-p_j-1)`.
- `max_adjacent_span=max_j(p_{j+1}-p_j)`; the literal Pro bound
  `max_unselected_run<=3` is equivalent to `max_adjacent_span<=4`.
- `max_uniform_displacement=max_j(abs(p_j-u_j))` in dense candidate positions.

## 4. Smallest P0 static/parity witnesses

Every witness fails closed. “Exact” means identical shape, dtype, values, and
ordering; floating comparisons are allowed only where a tolerance is stated.

### P0-U1 — one canonical uniform identity

Input: the accepted canonical fixture over normal and short-window cases,
including at least `T_v={16,17,31,32,383,384,385,767,768}`.

Pass conditions:

1. Clean data path and wrapper path call the same canonical generator or bind
   the same generator hash.
2. Both emit exactly the serialized vector `u` for every case.
3. `u[0]=0`, `u[-1]=T_v-1` when `K_eff>1`, length is `K_eff`, values are unique,
   strictly increasing, and in range.
4. Constant density decoding emits a bit-identical vector to `u`.

Falsifier: any value, endpoint, count, ordering, dtype, device-independent
serialization, or tie result differs. P0 stops; no tolerance and no repair.

### P0-U2 — unchanged-detector configuration identity

Input: fully resolved clean-uniform and density-wrapper configurations.

Pass condition: canonical serialization hashes are identical for backbone
architecture/interface, projection, prior generator, assignment, head,
classification/regression losses, optimizer/scheduler exposure, NMS,
post-processing after the external coordinate adapter, evaluator, class map,
split, block list, and augmentation policy. Only acquisition/scout and the two
external coordinate adapters may differ.

Falsifier: any unapproved downstream difference, physical-grid head enablement,
different NMS/evaluator, or unresolved inheritance. P0 stops.

### P0-U3 — uniform wrapper identity surface

With identical synthetic dense RGB sentinels, masks, physical segments, fixed
weights, and canonical indices, capture both paths at five boundaries:

1. selected integer indices;
2. gathered RGB after identical preprocessing;
3. compact valid mask and mapped training segments;
4. raw detector proposals/scores/labels before inverse mapping;
5. physical proposals presented to unchanged NMS and final serialized output.

Pass condition: indices/RGB/masks/raw scores/raw labels are bit-identical;
mapped segment/proposal coordinates have maximum absolute difference `<=1e-6`
dense units; unchanged NMS receives identical physical inputs and produces
identical outputs.

Falsifier: first divergence at any boundary. The divergence is attributed to
the wrapper, not learned acquisition; P0 stops.

## 5. Smallest P1 property and synthetic witnesses

These are future remote CPU-only checks. No local execution is authorized by
the current queue item.

### P1-D1 — density degeneration and determinism

For each fixture case, decode constant logits/density twice, under batch sizes
1 and 2, with sample permutation and duplication.

Pass conditions: every output is bit-identical to canonical `u`; a sample's
positions are invariant to batch size, permutation, and duplication.

Falsifier: any mismatch or batch-dependent position.

### P1-D2 — exact geometry

For constant, monotone, alternating, impulse, boundary-heavy, and seeded random
finite density vectors, including extreme finite logits:

- `len(p)=K_eff`;
- `p_0=0` and `p_{K_eff-1}=T_v-1`;
- all positions are integer, in range, unique, and strictly increasing;
- `max_unselected_run<=3` (`max_adjacent_span<=4`);
- `max_uniform_displacement<=16`;
- no non-finite value and no hidden interpolation counted as a heavy frame.

Falsifier: one violation. No post-hoc deduplication, clipping, uniform scaffold,
or threshold weakening may convert a failure into a pass.

### P1-C1 — physical/selected coordinate round trip

Use each accepted position vector as piecewise-linear knots
`(t=p_j,q=j)` plus the terminal boundary knot `(t=T_v,q=K_eff)`.
Test all knots, all interval midpoints, `0`, `T_v`, values within `1e-6` of each
knot, and seeded valid intervals.

Pass conditions:

- physical `t -> q -> t` maximum absolute error `<=1e-5` dense units;
- selected `q -> t -> q` maximum absolute error `<=1e-5` selected units;
- every selected frame knot round-trips exactly;
- mapped segment starts/ends remain ordered and within declared domains;
- labels and proposal scores are untouched.

Falsifier: tolerance breach, interval reversal, implicit endpoint convention,
or use of GT/teacher/cache/result payload during inference.

### P1-N1 — raw inverse mapping before unchanged NMS

Binding witness: instrument the external adapter boundary, not NMS internals.
Capture the raw selected-axis proposals, inverse-mapped physical proposals, and
the actual first argument passed to the unchanged `batched_nms`. The latter must
be bit-identical to the inverse-mapped physical proposals; the clean NMS callable
identity/config hash must equal the wrapper NMS identity/config hash. Assert no
NMS/top-k/overlap suppression occurs between detector raw output and inverse map.

Order-sensitive diagnostic: with legal positions `[0,4,5,9]`, same-class raw
segments `[[0,2],[1,3]]`, and scores `[0.9,0.8]`, selected-axis IoU is `1/3`
while physical-axis IoU is `1/9`. A harness-only hard-NMS threshold `0.30`
therefore yields different survivors. This diagnostic proves the test is
sensitive; production pass/fail still uses the unchanged clean soft-NMS config.

Falsifier: NMS sees selected-axis coordinates, any local NMS precedes mapping,
the NMS callable/config changes, or output is mapped only after suppression.

## 6. Evidence classes and required artifacts

Evidence classification is fixed as follows:

- P0 static/source/config reconciliation: `preparatory`; no model-quality claim.
- P1 property/synthetic integration: `preparatory`; no model-quality claim.
- P2 training-side mechanism admission: `mechanism_admission_preparatory`; it
  can admit or falsify a gradient path but cannot support an accuracy claim.
- P3 one-seed complete pilot: `pilot`; requires separate authorization and
  cannot support a final paper claim.
- P4 three-seed complete matrix: `formal_candidate`; still requires independent
  seals and a later human result-promotion gate.

Required return artifacts before P2 can become ready:

1. `DUCA_CANONICAL_UNIFORM_SPEC-v001.json` — rounding/tie rule and serialized
   supported-case fixtures.
2. `DUCA_P0_RESOLVED_CONFIG_DIFF-v001.json` — detector/evaluator/NMS identity.
3. `DUCA_P0_UNIFORM_WRAPPER_PARITY-v001.json` — five-boundary parity receipt.
4. `DUCA_P1_DENSITY_PROPERTIES-v001.json` — per-case exact geometry receipt.
5. `DUCA_P1_COORDINATE_ROUNDTRIP-v001.json` — knot/interior error maxima.
6. `DUCA_P1_PRENMS_ORDER-v001.json` — call trace, NMS identity, sensitivity case.
7. `DUCA_P1_REMOTE_CPU_RECEIPT-v001.json` — exact revision, patch/config IDs,
   command argv, environment, remote result path, exit status, zero dataset/GPU
   access, and deviation receipt.
8. Accepted Builder patch receipt and Critic P0/P1 closure receipt.

Broad per-file hash manifests are not required. Each receipt must bind exact Git
revision, patch/config identity, command, environment, output path, and deviation
receipt, consistent with `PRO_INITIAL_REVIEW-v002-intake.md:53-57`.

## 7. Exact future `PRE_RUN_READY` fields for P2 only

The following keys are mandatory. `status` must remain `BLOCKED` if any required
value is null, any P0/P1 gate is not `pass`, or any fresh-Pro question in Section
9 is unresolved.

```json
{
  "schema_version": "duca-pre-run-ready-v001",
  "record_id": null,
  "status": "BLOCKED",
  "stage": "P2_MECHANISM_ADMISSION",
  "project_id": "g-p-6a796fef9a00819194024cf1de3bd697",
  "created_at": null,
  "created_by_role": "evaluator",
  "authority": {
    "current_parent_artifact_id": "art-20260811T045704Z-2d13c25e41",
    "fresh_pro_instruction_artifact_id": null,
    "fresh_pro_instruction_message_id": null,
    "human_authorization_id": null,
    "authorized_experiment_id": null,
    "authorized_stage": "P2_ONLY",
    "authorization_expires_at": null,
    "run_equivalent_cap": null,
    "spend_cap": null
  },
  "code": {
    "git_revision": "63a726a4aaf48ecbf6780bb196de43a890c6b4df",
    "accepted_patch_artifact_id": null,
    "accepted_patch_sha256": null,
    "resolved_config_path": null,
    "resolved_config_sha256": null,
    "mechanism_harness_path": null,
    "mechanism_harness_sha256": null,
    "canonical_uniform_spec_id": "DUCA_CANONICAL_UNIFORM_SPEC-v001",
    "canonical_uniform_spec_sha256": null,
    "deviation_receipt_id": null
  },
  "p0_p1_gates": {
    "uniform_wrapper_parity_artifact_id": null,
    "uniform_wrapper_parity_status": "not_run",
    "detector_invariance_artifact_id": null,
    "detector_invariance_status": "not_run",
    "density_properties_artifact_id": null,
    "density_properties_status": "not_run",
    "coordinate_roundtrip_artifact_id": null,
    "coordinate_roundtrip_status": "not_run",
    "pre_nms_mapping_artifact_id": null,
    "pre_nms_mapping_status": "not_run",
    "critic_p0_p1_closure_artifact_id": null,
    "critic_p0_p1_closure_status": "not_run"
  },
  "data": {
    "scope": "training_side_video_disjoint_utility_only",
    "dataset_identity": null,
    "training_media_manifest_sha256": null,
    "training_annotation_sha256": null,
    "utility_set_manifest_path": null,
    "utility_set_manifest_sha256": null,
    "main_training_exclusion_receipt_id": null,
    "video_count": null,
    "minimum_video_count": 64,
    "window_count": null,
    "minimum_window_count": 128,
    "stratification_spec_sha256": null,
    "video_group_key": "video_id",
    "validation_media_access": false,
    "validation_annotation_access": false,
    "test_media_access": false,
    "test_annotation_access": false,
    "forbidden_payload_attestation_id": null
  },
  "models": {
    "fixed_detector_checkpoint_path": null,
    "fixed_detector_checkpoint_sha256": null,
    "fixed_detector_config_sha256": null,
    "fixed_selector_checkpoint_path": null,
    "fixed_selector_checkpoint_sha256": null,
    "checkpoint_provenance_receipt_id": null
  },
  "sampling": {
    "dense_candidate_length": 768,
    "requested_k": 384,
    "effective_k_rule": "min(384,16*floor(T_v/16));T_v<16=fail",
    "endpoint_rule": "first_and_last_valid_physical_positions",
    "max_unselected_run": 3,
    "max_adjacent_span": 4,
    "max_uniform_displacement": 16,
    "exact_unique_strictly_increasing": true,
    "perturbation_spec_id": null,
    "perturbation_spec_sha256": null,
    "perturbation_families": [
      "single_frame_swap",
      "dispersed_5pct_swap",
      "dispersed_10pct_swap",
      "contiguous_swap",
      "global_density_step_0.25",
      "global_density_step_0.5",
      "global_density_step_1.0"
    ]
  },
  "statistics": {
    "outcomes": [
      "total_detector_loss_change",
      "classification_loss_change",
      "boundary_regression_loss_change"
    ],
    "bootstrap_group": "video_id",
    "bootstrap_replicates": null,
    "bootstrap_seed": null,
    "confidence_level": 0.95,
    "single_frame_spearman_lower_bound_gt": 0.0,
    "five_percent_spearman_lower_bound_gt": 0.0,
    "direction_accuracy_lower_bound_gt": 0.5,
    "top_decile_vs_random_test": null,
    "top_decile_vs_random_threshold": null,
    "ten_percent_inverse_relation_test": null,
    "ten_percent_inverse_relation_threshold": null,
    "global_inverse_relation_test": null,
    "global_inverse_relation_threshold": null,
    "component_alignment_rule": null,
    "failure_action": "stop_direct_detector_gradient_route"
  },
  "execution": {
    "mode": "remote_only",
    "local_execution_allowed": false,
    "remote_cluster": null,
    "slurm_partition": null,
    "hardware_model": null,
    "gpu_count": null,
    "cpu_count": null,
    "memory_limit": null,
    "walltime_limit": null,
    "environment_lock_path": null,
    "environment_lock_sha256": null,
    "working_directory": null,
    "command_argv": null,
    "output_root": null,
    "job_name": null,
    "concurrent_jobs_forbidden": true
  },
  "output_and_access": {
    "allowed_outputs": [
      "training_side_loss_deltas",
      "registered_perturbation_identities",
      "registered_alignment_statistics",
      "resource_and_failure_receipts"
    ],
    "official_evaluator_invoked": false,
    "official_metrics_computed": false,
    "map_fields_forbidden": true,
    "held_out_predictions_forbidden": true,
    "raw_prediction_cache_for_selection_forbidden": true,
    "result_path": null,
    "result_seal_path": null,
    "access_log_path": null
  },
  "ready_decision": {
    "all_required_fields_non_null": false,
    "all_p0_p1_gates_pass": false,
    "all_fresh_pro_questions_resolved": false,
    "no_scope_deviation": false,
    "ready": false
  }
}
```

`deviation_receipt_id` must be the literal string `"none"` when there is no
deviation; null is not a ready value. P2 must receive a new run identity after
any code/config/data/checkpoint/command/environment/authority change.

## 8. Remote-only future executable checks

The following may be executed only after their respective future authority:

1. P1 remote CPU, synthetic only: canonical uniform, constant-density,
   endpoint/count/order/uniqueness/span/displacement, batch invariance,
   coordinate round trip, production-call pre-NMS capture, and resolved-config
   identity. Dataset loading and GPU initialization must be actively denied.
2. P1 remote CPU, sentinel parity: clean/wrapper synthetic RGB, masks, segments,
   raw proposals, inverse mapping, and unchanged NMS parity. No real media.
3. P2 remote Slurm mechanism admission: fixed detector/selector checkpoints and
   the predeclared training-side video-disjoint utility manifest; exact-K legal
   perturbations only; record loss changes and registered statistics only.

No current authorization permits any of these executions.

## 9. Fresh-Pro questions required before P2

- Q1 — Canonical uniform: confirm the exact endpoint-inclusive integer formula,
  floating/integer implementation, and tie-breaking rule. This resolves the
  static 766-versus-767 conflict.
- Q2 — Span terminology: confirm that the frozen bound means three unselected
  positions (`adjacent span <=4`), not the legacy test's `adjacent gap <=3`.
- Q3 — Coordinate domain: confirm end-exclusive segment boundaries on `[0,T_v]`
  with terminal knot `(K_eff,T_v)`, while frame indices remain `[0,T_v-1]`.
- Q4 — NMS boundary: name the one canonical pre-NMS adapter hook covering local
  and sliding-window/global NMS without changing clean NMS behavior.
- Q5 — P2 top-decile rule: define the statistic, random-control replication,
  effect threshold, and confidence rule for “top 10% outperform random.”
- Q6 — P2 inverse-relation rule: numerically define “no stable inverse
  relationship” for 10% swaps and global density steps.
- Q7 — P2 component rule: specify whether classification and boundary regression
  must both pass independently or at least one must pass, and give the threshold.
- Q8 — P2 bootstrap: freeze replicate count, bootstrap seed, handling of ties,
  missing/failed perturbations, and aggregation across videos/windows.
- Q9 — P2 authority/resources: name the concrete experiment, training-side
  utility-set construction, fixed checkpoint provenance, remote Slurm resources,
  run-equivalent cap, spend cap, command, and output seal/access policy.

Until Q1-Q4 are answered and P0/P1 pass, P2 is not scientifically or
operationally ready. Until Q5-Q9 are answered, no `PRE_RUN_READY` record can set
`status=READY`.

## 10. Final evidence receipt

`EVALUATOR_DECISION: PREREGISTRATION_COMPLETE_GATES_UNPASSED`.

This artifact defines falsifiable P0/P1 witnesses and the exact P2 readiness
record. It supplies no pass receipt, no execution authorization, no metric, and
no evidence for a paper claim.
