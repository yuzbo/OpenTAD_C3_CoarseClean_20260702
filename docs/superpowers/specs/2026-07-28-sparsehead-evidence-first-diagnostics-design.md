# SparseHead Evidence-First Diagnostics Design

**Date:** 2026-07-28  
**Status:** implemented and Linux CPU tested; four-condition CUDA gate pending
**Canonical repository:** `OpenTAD_C3_CoarseClean_20260702`  
**Historical source:** `codex/phystime-performance-diagnosis-20260712@e05f623133128c9a4cd56be4656c8fb5099426ac`

## 1. Decision

Implement the approved **Approach A: minimal evidence-first chain**.

The implementation recovers the strict-source-dtype frozen decode replay and
the existing PhysTime geometry/prediction diagnostics. It does not change
training, the NMS algorithm, evaluation, SDPQ, or checkpoint semantics. It
restores the already established P0 full-precision pre/post-NMS controls and
the explicit physical-second grid contract required for native replay parity.
It does not restore the historical owner/scheduler/submission-state framework.

This phase answers two questions before any SDPQ v2 design:

1. How much of the selected-axis versus physical-metric gap comes from the
   inference decode axis under frozen weights?
2. Where do physical-metric and SDPQ differ by duration, high-IoU behavior,
   assignment, support observability, and classification ranking?

## 2. Scientific contract

### Primary claim under test

Physical-time geometry changes detector behavior beyond presentation-level
coordinate conversion.

### Supporting diagnostic claim

SDPQ's representability and assignment coverage are not sufficient to explain
its accuracy; support evidence and prediction ranking must be measured
separately.

### Anti-claims

This phase must not claim:

- that decode replay establishes a causal training effect;
- that physical-metric is paper-ready;
- that SDPQ is improved or refuted by new training;
- that assignment coverage alone explains mAP;
- that the archived irregular bridge is a main method.

## 3. Frozen inputs

The replay consumes the already frozen native-J192 full60 selected-axis and
physical-metric epoch-59 checkpoints, with online and EMA weights. The four
conditions are:

- selected-online;
- selected-EMA;
- physical-online;
- physical-EMA.

Each condition captures one source forward and replays the same tensors with:

- `uniform_rank_seconds` decode;
- `physical_time_seconds` decode.

Prediction diagnostics consume existing OpenTAD prediction JSON and THUMOS
annotation JSON. Geometry diagnostics consume real dataset rows and model
assignment/support metadata. No validation/test GT enters model inference or
selection.

## 4. Architecture and data flow

### 4.1 Opt-in capture

`AnchorFreeHead` exposes an opt-in capture state. The normal path remains
disabled and byte-for-byte behavior-compatible when capture is off.

Captured ranking scores retain their original Torch dtype through NumPy
serialization. The artifact records:

- source Torch dtype;
- stored NumPy dtype;
- ordering-sensitive semantic role;
- shape and canonical SHA-256;
- point, mask, regression, class-score, axis, and window contracts.

Any float16-to-float32 widening of ranking scores is a schema violation.

### 4.2 Frozen dual-axis replay

The replay engine:

1. validates schema-v2 dtype provenance;
2. reconstructs proposals from captured point/regression tensors;
3. preserves the source score dtype for CPU sort and pre-NMS top-k;
4. executes both decode axes from the same artifact;
5. applies unchanged production post-processing and evaluator semantics;
6. emits direct-versus-native parity and cross-axis metrics.

Native-axis replay must exactly reproduce capture-enabled direct inference for
each condition. A mismatch stops the condition and writes a failure artifact;
captured native proposals may be used only as an audit reference, never as a
replacement for reconstructed proposals.

### 4.3 Geometry and assignment diagnostics

Recover the historical performance-drop analyzer to report:

- physical and selected candidate counts;
- support/evidence coverage and observation mass;
- eligible and positive assignments;
- GT without eligible/assigned locations;
- multi-GT conflicts;
- duration bins `[0,1)`, `[1,2)`, `[2,4)`, `[4,8)`, `[8,16)`,
  `[16,32)`, and `[32,inf)` seconds.

The existing `tools/bata/audit_sparse_head_assignment.py` remains the detailed
per-GT/native-axis audit. The restored analyzer adds aggregate summaries; it
does not replace the existing audit.

### 4.4 Prediction diagnostics

Recover the prediction analyzer to report per duration bin:

- class-agnostic recall;
- class-aware recall;
- best temporal IoU distribution;
- mAP-oriented recall at IoU 0.3/0.5/0.7;
- label accuracy after localization;
- start/end boundary MAE at fixed minimum IoU;
- top-K sensitivity.

This separates localization failure from classification/ranking failure.

## 5. Implementation surface

### Restore from the historical source

- `opentad/cores/phystime_decode_replay_capture.py`
- `tools/bata/replay_phystime_decode_cross.py`
- `tools/bata/validate_phystime_decode_cross_replay.py`
- `tools/bata/run_phystime_decode_cross_gate.py`
- `tools/bata/preflight_phystime_decode_cross.py`
- the replay/content validation logic from
  `tools/bata/validate_phystime_decode_cross_suite.py`
- `tools/bata/replay_phystime_p0_fullprecision_nms.py`
- `tools/bata/run_phystime_p0_fullprecision_gate.py`
- `tools/bata/validate_phystime_p0_fullprecision_replay.py`
- `tools/bata/validate_phystime_p0_fullprecision_suite.py`
- `tools/bata/analyze_phystime_performance_drop.py`
- `tools/bata/analyze_phystime_prediction_diagnostics.py`
- `configs/adatad/thumos/phystime_g1a_selected_axis_native_j192_decode_replay.py`
- `configs/adatad/thumos/phystime_g1a_physical_metric_native_j192_decode_replay.py`
- `tests/test_phystime_decode_cross_replay.py`
- `tests/test_phystime_performance_diagnostics.py`
- `tests/test_phystime_prediction_diagnostics.py`
- the three direct gate/replay/suite Slurm launchers, without the historical
  automatic submission framework.

### Selectively integrate

- `opentad/cores/test_engine.py`: create, collect, and finalize the opt-in
  capture only when configured.
- `opentad/models/dense_heads/anchor_free_head.py`: expose capture state and
  retain source score dtype.
- `tools/bata/validate_phystime_decode_cross_suite.py`: provide an
  evidence-only mode that validates the gate and four replay completion
  artifacts without requiring owner manifests, `jobs.tsv`, scheduler
  snapshots, or submission-attempt state. Its numeric, hash, config,
  checkpoint, shared-observation, completion, and fatal-log checks remain
  fail-closed.

Changes to `single_stage.py`, NMS, evaluator, or production post-processing are
forbidden unless an exact source comparison proves they are pre-existing P0
full-precision semantics required by the frozen evidence. Any such difference
must be isolated and parity-tested rather than copied wholesale.

### Explicitly excluded

- `capture_phystime_decode_cross_scheduler.py`
- `claim_phystime_decode_cross_owner.py`
- `manage_phystime_decode_cross_submission_state.py`
- `submit_phystime_decode_cross_replay.sh`
- historical automatic DAG ownership/retry machinery
- model, loss, optimizer, NMS, evaluator, checkpoint, or training changes
- SDPQ v2 implementation

## 6. Failure handling

The chain fails closed on:

- missing or inconsistent source dtype provenance;
- ordering-sensitive score widening;
- config/checkpoint/axis hash mismatch;
- direct/native replay mismatch;
- selected/physical shared-observation mismatch;
- non-finite tensors or predictions;
- incomplete four-condition gate;
- absent annotation or prediction identities.

The evidence-only suite accepts explicit gate/replay artifact paths. It does
not infer jobs from a directory and does not authorize, submit, retry, cancel,
or query Slurm jobs.

Failures produce structured JSON with the first mismatch, tensor dtype/shape,
candidate boundary, hashes, runtime fingerprint, and condition identity.

## 7. Verification

### Local, no-GPU

- Python compilation for every restored/integrated file;
- resolved-config loading for both replay arms;
- strict source-dtype capture/roundtrip tests;
- widening-rejection and top-k tie-boundary tests;
- direct/native reconstruction unit tests;
- duration-bin geometry aggregation tests;
- class-aware/class-agnostic high-IoU prediction tests;
- no-capture normal-path parity test;
- Bash syntax for the three direct launchers.
- evidence-only suite tests proving that missing, duplicate, or mismatched
  condition artifacts fail without requiring scheduler/owner files.

### N16R4

- full focused replay/diagnostic suite;
- one real selected/physical × online/EMA CUDA gate;
- no formal replay or metric claim unless all four native parity conditions
  pass.

## 8. Acceptance and stop conditions

Implementation is complete only when:

1. all restored configs resolve;
2. local static/non-Torch tests pass;
3. Linux Torch focused tests pass;
4. four-condition real gate passes exact native replay parity;
5. geometry and prediction analyzers emit complete duration/high-IoU reports.

If any native replay differs from direct inference, stop before formal replay.
If diagnostics show that current SDPQ failure is dominated by classification
or support evidence rather than assignment, the subsequent Pro design must
target that mechanism. No full60 or SDPQ v2 training is authorized by this
design.

## 9. Implementation status on 2026-07-28

The evidence-first surface is now `tested` at the Linux CPU level:

- source `cls_scores` retain their ordering-sensitive dtype through capture,
  NPZ storage, sort, and top-k;
- capture is opt-in and refuses unconsumed-state overwrite;
- the evidence suite accepts only explicit preflight, gate, P0, four-condition
  completion, and log paths; it does not inspect owner, `jobs.tsv`, scheduler,
  or submission-attempt state;
- production and replay share the exact configurable physical-second grid
  mapping and domain clamp used by the historical P0 evidence;
- legacy pre-cross rounding remains the default; P0/full-precision configs
  disable it explicitly;
- geometry, prediction, selected-axis post-processing, TrueTime dependency,
  full-precision NMS, and decode-replay focused tests are present.

The final isolated N16R4 CPU package had SHA-256
`e4814e3544784b3608c007a11946464b4f597e0fbf9a23a5910e3b0171bef388`
and ran from
`/data/run01/sczc063/yuzibo/sparsehead_remote_cpu_a6bdc084_20260728_07`.
Three configs resolved, the import closure passed, Python compilation and
three launcher syntax checks passed, and the focused suite reported
`59 passed in 64.69s`.

This does not satisfy acceptance items 4--5. No CUDA forward, four-condition
gate, formal replay, new mAP, or empirical SparseHead/SDPQ claim exists yet.
