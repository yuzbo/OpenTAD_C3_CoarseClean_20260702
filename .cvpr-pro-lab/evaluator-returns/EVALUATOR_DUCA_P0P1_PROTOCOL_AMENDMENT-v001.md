---
doc_id: EVALUATOR_DUCA_P0P1_PROTOCOL_AMENDMENT
version: v001
status: designed_not_executed
date: 2026-08-11
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260811T061218Z-cdd2294b327a
parent_decision: PRO_P0_BLOCKER_DECISION-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: EVALUATOR_DUCA_DENSITY_P0P1-v001
evidence_class: preparatory_protocol
pre_run: BLOCKED
execution: prohibited
---

# Evaluator DUCA P0/P1 protocol amendment

## 1. Authority and no-execution receipt

This artifact executes only durable queue
`msg-20260811T061218Z-cdd2294b327a` under the confirmed fresh Pro decision
`PRO_P0_BLOCKER_DECISION-v001`. It amends the protocol only; it does not admit
or authorize P1.

No dataset, model checkpoint/weight, model forward, local CPU command, remote
command, GPU, Slurm, metric, validation/test split, or held-out material was
accessed or executed. Read activity was limited to the authorized frozen source,
config, role, queue, Pro-decision, and prior protocol records needed to write
this amendment.
No code/config, Git state, route, claim, or `PRE_RUN` status was changed. All
checks were performed personally by the registered Evaluator task:
`extra_processes=none`, `subagents=none`, `probes=none`.

## 2. Frozen P0 semantics inherited from Pro

The future P1 harness must bind these literals without reinterpretation:

- Prefix-contiguous valid length `T_v` and
  `K_eff=min(384,16*floor(T_v/16))`; `T_v<16` fails closed.
- Canonical positions
  `u_j=floor((2*j*(T_v-1)+(K_eff-1))/(2*(K_eff-1)))`.
- `u[0]=0`, `u[-1]=T_v-1`, exact length `K_eff`, integer dtype, in-range,
  unique, and strictly increasing.
- Exactly constant density must use the canonical specialization and be
  bit-identical in shape, dtype, values, and order to `u`. Near-constant
  density must not enter that specialization.
- Raw proposal endpoints arrive tagged `selected_q` in end-exclusive
  `[0,K_eff]` and are mapped exactly once through strictly increasing knots to
  tagged `physical_dense` in `[0,T_v]` at entry to each per-sample
  `SingleStageDetector.post_processing`, before filtering, top-k, IoU, or NMS.
- Scores, labels, detector/head/losses, NMS callable/config, evaluator, split,
  and class map remain unchanged. Unknown or repeated coordinate mapping fails
  closed.

## 3. Preconditions for any future P1 command

All of the following must be literal, non-null fields in a later fresh Pro P1
decision and its durable queue. Their presence in this amendment is not
authority to run them.

1. An accepted Builder patch receipt derived from base revision
   `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, plus a clean, immutable remote
   snapshot identity containing exactly that accepted patch.
2. A Critic closure receipt of `P0_STATIC_PASS` for the complete accepted diff.
3. Exact paths for the canonical fixture, clean-uniform resolved config,
   DUCA-density resolved config, P1 harness, receipt schema, remote working
   directory, and output root.
4. A fresh Pro decision explicitly admitting only remote-CPU synthetic P1 and
   a durable Evaluator queue citing it.
5. Active denials for dataset/media/annotation/checkpoint access, CUDA device
   visibility and initialization, evaluator/metric invocation, networking,
   and raw-prediction cache use.
6. A clean process table/run lock showing no duplicate P1 job. Coordinator
   monitoring is read-only and must not launch a second run.

Any missing or mismatched precondition yields `P1_AUTHORITY_OR_INPUT_BLOCKED`;
no command below may start.

## 4. Exact future remote-only P1 command contract

The later Pro decision must replace every angle-bracket token with one literal
value and seal the resulting argv before execution. An unresolved token is a
hard failure. The remote snapshot is pre-staged; these commands do not clone,
pull, patch, install, load modules, traverse data, or initialize CUDA.

Required environment for every argv:

```text
CUDA_VISIBLE_DEVICES=""
PYTHONNOUSERSITE="1"
DUCA_P1_DENY_DATASET="1"
DUCA_P1_DENY_CHECKPOINT="1"
DUCA_P1_DENY_CUDA="1"
DUCA_P1_DENY_METRICS="1"
DUCA_P1_DENY_NETWORK="1"
DUCA_P1_DENY_RAW_PREDICTIONS="1"
```

The executable is the already installed remote interpreter
`/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`. The sealed future
argv list is exactly:

```text
# C0: static authority/input/access preflight; no test collection
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -B \
  <P1_HARNESS> preflight \
  --project-id g-p-6a796fef9a00819194024cf1de3bd697 \
  --parent-decision <FRESH_PRO_P1_DECISION_ID> \
  --base-revision 63a726a4aaf48ecbf6780bb196de43a890c6b4df \
  --snapshot-revision <ACCEPTED_P1_SNAPSHOT_REVISION> \
  --accepted-patch-receipt <BUILDER_PATCH_RECEIPT_ID> \
  --critic-receipt <CRITIC_P0_STATIC_PASS_RECEIPT_ID> \
  --canonical-fixture <CANONICAL_FIXTURE_JSON> \
  --clean-config <CLEAN_UNIFORM_RESOLVED_CONFIG_JSON> \
  --duca-config <DUCA_DENSITY_RESOLVED_CONFIG_JSON> \
  --deny-dataset --deny-checkpoint --deny-cuda --deny-metrics \
  --deny-network --deny-raw-predictions \
  --out <OUTPUT_ROOT>/00-preflight.json

# C1: deterministic property/parity suite; synthetic tensors only
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -B \
  <P1_HARNESS> verify \
  --preflight <OUTPUT_ROOT>/00-preflight.json \
  --canonical-fixture <CANONICAL_FIXTURE_JSON> \
  --clean-config <CLEAN_UNIFORM_RESOLVED_CONFIG_JSON> \
  --duca-config <DUCA_DENSITY_RESOLVED_CONFIG_JSON> \
  --fixture-valid-lengths 16,17,31,32,383,384,385,767,768 \
  --fixture-invalid-lengths 0,1,15 \
  --boundary-parity-lengths 16,17,31,32,385 \
  --density-families constant,near_constant,monotone,alternating,impulse,boundary_heavy,seeded_random,finite_extreme \
  --seed 20260811 \
  --coordinate-roundtrip-atol-dense 1e-5 \
  --coordinate-roundtrip-atol-selected 1e-5 \
  --boundary-parity-atol-dense 1e-6 \
  --out <OUTPUT_ROOT>/01-verification.json

# C2: schema/attestation seal; no re-execution of checks
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -B \
  <P1_RECEIPT_VALIDATOR> \
  --schema <P1_RECEIPT_SCHEMA_JSON> \
  --preflight <OUTPUT_ROOT>/00-preflight.json \
  --verification <OUTPUT_ROOT>/01-verification.json \
  --command-manifest <SEALED_COMMAND_MANIFEST_JSON> \
  --out <OUTPUT_ROOT>/DUCA_P1_REMOTE_CPU_RECEIPT-v001.json
```

`C0`, `C1`, and `C2` run serially and stop on the first non-zero exit. No
command may be added, removed, reordered, or retried with changed inputs. A
future harness may be admitted only if its exact path and snapshot identity are
sealed by the fresh P1 decision; this amendment does not require or authorize
its implementation.

## 5. Required deterministic witnesses

### 5.1 Canonical endpoint and constant-density identity

For valid `T_v={16,17,31,32,383,384,385,767,768}`, compare the canonical
fixture, clean path, DUCA uniform path, and constant-density decoder. Each must
have the exact expected `K_eff={16,16,16,32,368,384,384,384,384}`. Each vector
must match bit-for-bit; normal `T_v=768` must start at 0 and end at 767.

`T_v={0,1,15}` must fail before allocation with
`DUCA_VALID_LENGTH_LT_16`. No clipping, deduplication, tolerance repair,
floating linspace, banker rounding, or fallback generator is admissible.

### 5.2 Strict monotonicity and uniqueness

For all declared density families and valid lengths, assert integer positions,
exact length `K_eff`, endpoints, range, strict increase, and uniqueness. Record
the first violating pair and full case identity. Geometry checks also record
`max_unselected_run` and `max_uniform_displacement`; any frozen-bound violation
is a failure and may not be repaired in the harness.

### 5.3 Five boundary parity cases and five pipeline boundaries

The exact length cases are:

1. `T_v=16,K_eff=16`: minimum valid identity case.
2. `T_v=17,K_eff=16`: first non-multiple tail.
3. `T_v=31,K_eff=16`: last length before the next 16-frame budget step.
4. `T_v=32,K_eff=32`: first length after that budget step.
5. `T_v=385,K_eff=384`: first length above the requested-budget cap.

For each case, the clean-uniform and DUCA constant-density paths use identical
synthetic RGB sentinels, masks, physical training segments, fixed synthetic raw
proposals, scores, and labels. Capture these five boundaries:

1. selected integer indices;
2. gathered RGB after identical preprocessing;
3. compact valid mask and mapped training segments;
4. raw selected-axis proposals, scores, and labels before inverse mapping;
5. physical proposals presented to unchanged NMS and final serialized output.

Indices, RGB, masks, scores, labels, and serialized outputs are bit-identical.
Segment/proposal coordinate error is at most `1e-6` dense units. The receipt
records the first divergent boundary and tensor element.

### 5.4 Coordinate tag, round-trip, and double-map failures

Accepted state transition is exactly `selected_q -> physical_dense`. Test all
knots, interval midpoints, values within `1e-6` of knots, and seeded valid
segments. Maximum round-trip errors are `1e-5` dense and selected units; every
selected-frame knot is exact; segment order, scores, and labels are preserved.

- Missing/unknown input tag: `DUCA_COORD_TAG_UNKNOWN` before filtering/top-k.
- `physical_dense` passed to the adapter: `DUCA_COORD_DOUBLE_MAP`.
- Output not tagged `physical_dense`: `DUCA_COORD_OUTPUT_TAG_INVALID`.
- Reversed/out-of-domain/non-monotone knots: `DUCA_COORD_KNOTS_INVALID`.

All failures must occur before any NMS call; the NMS call count must remain
zero in these negative cases.

### 5.5 Pre-NMS order for every non-sliding branch

Instrument only the adapter boundary and the unchanged NMS callable. Exercise
the cross-product of single-class versus multiclass score handling, NMS present
versus `nms=None`, and `fps==-1` versus stride/fps conversion, always with
`sliding_window=false`. For NMS-present cases, the first NMS segment argument
must be bit-identical to tagged `physical_dense`, and the event order must be:

```text
raw_selected_q -> inverse_map_once -> physical_dense -> filter/topk_if_any -> nms -> seconds
```

For `nms=None`, mapping must still occur exactly once before later coordinate
conversion. The order-sensitive sensitivity fixture is fixed to positions
`[0,4,5,9]`, same-class segments `[[0,2],[1,3]]`, scores `[0.9,0.8]`, and a
harness-only hard-NMS threshold `0.30`; selected and physical survivors must
differ. Production pass/fail then uses the unchanged clean soft-NMS callable
and config.

### 5.6 Detector/config invariance

Canonical resolved serialization may differ only at acquisition/scout-density,
the canonical sampling-source field, the external coordinate adapter metadata,
and nonsemantic output/work-directory fields. Backbone interface, projection,
neck, prior generator, assignment, head, classification/regression losses,
optimizer/scheduler exposure, NMS callable and all NMS values, evaluator,
split, class map, block list, and augmentation policy must be identical.
Physical-grid head geometry control is forbidden. Record the exact first
unapproved key path; unresolved inheritance is a failure.

## 6. Failure signatures and stop rule

The receipt uses only these stable primary signatures:

```text
P1_AUTHORITY_OR_INPUT_BLOCKED
P1_SNAPSHOT_MISMATCH
P1_FORBIDDEN_ACCESS
P1_CUDA_VISIBLE_OR_INITIALIZED
P1_UNIFORM_ENDPOINT_MISMATCH
P1_CONSTANT_DENSITY_NOT_BIT_IDENTICAL
P1_KEFF_MISMATCH
P1_POSITION_NOT_STRICTLY_INCREASING
P1_POSITION_NOT_UNIQUE
P1_BOUNDARY_PARITY_MISMATCH
P1_COORD_TAG_UNKNOWN
P1_COORD_DOUBLE_MAP
P1_COORD_OUTPUT_TAG_INVALID
P1_COORD_KNOTS_INVALID
P1_COORD_ROUNDTRIP_TOLERANCE
P1_PRENMS_ORDER_VIOLATION
P1_NMS_IDENTITY_OR_CONFIG_CHANGED
P1_CONFIG_DRIFT
P1_COMMAND_OR_SCHEMA_DEVIATION
P1_NONZERO_EXIT
```

Any signature, non-zero exit, missing receipt field, forbidden access event,
unsealed argv/environment change, or unexpected output stops P1 immediately as
`P1_BLOCKED`. Preserve the first-failure output and do not tune thresholds,
repair positions, edit fixtures, or relabel the evidence. A retry is allowed
only for an objective infrastructure interruption, with the same sealed
snapshot, inputs, environment, and argv, and a new receipt linked to the failed
attempt. No scientific retry is self-authorized.

Even a complete future P1 pass remains `preparatory_synthetic`; it supplies no
model-quality, cost, metric, held-out, or paper-claim evidence and does not
automatically unlock P2.

## 7. `DUCA_P1_REMOTE_CPU_RECEIPT-v001` schema

```json
{
  "schema_version": "duca-p1-remote-cpu-receipt-v001",
  "receipt_id": null,
  "created_at": null,
  "project_id": "g-p-6a796fef9a00819194024cf1de3bd697",
  "queue_message_id": null,
  "fresh_pro_p1_decision_id": null,
  "base_revision": "63a726a4aaf48ecbf6780bb196de43a890c6b4df",
  "snapshot_revision": null,
  "accepted_builder_patch_receipt_id": null,
  "critic_p0_static_pass_receipt_id": null,
  "run_id": null,
  "run_lock_id": null,
  "duplicate_job_check": {"passed": false, "observed_job_ids": []},
  "remote": {
    "host": null,
    "working_directory": null,
    "python_executable": "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python",
    "python_version": null,
    "environment_identity": null,
    "cpu_identity": null,
    "cuda_visible_devices": "",
    "cuda_initialized": false
  },
  "inputs": {
    "p1_harness_path": null,
    "receipt_validator_path": null,
    "receipt_schema_path": null,
    "canonical_fixture_path": null,
    "clean_uniform_resolved_config_path": null,
    "duca_density_resolved_config_path": null,
    "sealed_command_manifest_path": null,
    "fixture_valid_lengths": [16,17,31,32,383,384,385,767,768],
    "fixture_invalid_lengths": [0,1,15],
    "boundary_parity_lengths": [16,17,31,32,385],
    "seed": 20260811
  },
  "commands": [
    {"id": "C0", "argv": null, "started_at": null, "ended_at": null, "exit_code": null, "stdout_path": null, "stderr_path": null},
    {"id": "C1", "argv": null, "started_at": null, "ended_at": null, "exit_code": null, "stdout_path": null, "stderr_path": null},
    {"id": "C2", "argv": null, "started_at": null, "ended_at": null, "exit_code": null, "stdout_path": null, "stderr_path": null}
  ],
  "access_attestation": {
    "dataset_media_annotation_accessed": false,
    "validation_or_test_accessed": false,
    "checkpoint_or_learned_weight_loaded": false,
    "gpu_or_cuda_initialized": false,
    "metric_or_evaluator_invoked": false,
    "network_accessed": false,
    "raw_prediction_cache_accessed": false,
    "forbidden_access_events": []
  },
  "checks": {
    "canonical_endpoint_identity": {"status": "not_run", "first_failure": null},
    "constant_density_bit_identity": {"status": "not_run", "first_failure": null},
    "valid_length_keff_contract": {"status": "not_run", "first_failure": null},
    "strict_monotonicity": {"status": "not_run", "first_failure": null},
    "uniqueness": {"status": "not_run", "first_failure": null},
    "five_case_five_boundary_parity": {"status": "not_run", "first_failure": null},
    "coordinate_roundtrip": {"status": "not_run", "max_error_dense": null, "max_error_selected": null, "first_failure": null},
    "coordinate_tag_fail_closed": {"status": "not_run", "first_failure": null},
    "double_map_fail_closed": {"status": "not_run", "first_failure": null},
    "pre_nms_all_non_sliding_paths": {"status": "not_run", "event_traces": [], "first_failure": null},
    "nms_identity_and_config": {"status": "not_run", "first_failure": null},
    "detector_config_invariance": {"status": "not_run", "first_unapproved_diff": null}
  },
  "failure": {
    "primary_signature": null,
    "case_id": null,
    "expected": null,
    "observed": null,
    "first_divergent_boundary": null,
    "artifact_path": null
  },
  "deviation_receipt": "none",
  "evidence_class": "preparatory_synthetic",
  "p1_status": "NOT_RUN",
  "scientific_evidence_status": "BLOCKED_PRE_RESULT",
  "pre_run_status": "BLOCKED"
}
```

For a future `P1_PASS`, every command exit code must be zero, every check must
be `pass`, all access attestations must remain false, `deviation_receipt` must
be the literal `"none"`, and the sealed argv must exactly match Section 4.
Otherwise the only admissible terminal status is `P1_BLOCKED`.

## 8. Evaluator conclusion

`EVALUATOR_DECISION: PRE_RUN_BLOCKED`.

The bounded P0/P1 amendment is complete as protocol evidence only. P1 remains
unauthorized and unexecuted; P2 `PRE_RUN` remains `BLOCKED`; scientific evidence
remains `BLOCKED_PRE_RESULT`.
