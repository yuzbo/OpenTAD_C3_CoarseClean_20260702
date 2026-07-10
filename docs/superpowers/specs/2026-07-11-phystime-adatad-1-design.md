# PhysTime-AdaTAD 1.0 Design Specification

## 1. Status and Scope

This specification defines the first raw-video, pre-backbone sparse integration
of the implemented PhysTime-TAD 2.0 geometry, projection, and head with the
official OpenTAD AdaTAD VideoMAE-S pipeline.

PhysTime-AdaTAD 1.0 is not the cancelled I3D feature-token experiment. It must
start from THUMOS RGB videos, select raw frames before decoding and VideoMAE,
and train the official AdaTAD temporal adapters and the detection stack from
the downstream TAD loss.

The first experiment isolates the detector head. It does not learn a selector
and does not introduce actionness, a teacher, an offline ledger, or a dynamic
budget controller.

## 2. Research Question

Under identical raw-frame observations and the same official AdaTAD backbone,
does explicit physical-time detection outperform treating irregular
observations as an ordinary contiguous token sequence?

The primary comparison is made at K=384 observations from a logical 768-frame
window. Every compared head must receive the same selected frame indices for
each sample.

## 3. Non-Goals

- No learned frame selector in the 1.0 head-isolation experiment.
- No GT-dependent sampling decision.
- No actionness, coarse classifier, DUCA, C3 ledger, teacher, or prediction
  cache.
- No selected-axis remapping of PhysTime ground truth.
- No claim that the frozen VideoMAE trunk itself is physical-time aware.
- No feature archive download or pre-extracted I3D input.
- No paired-view consistency loss in the primary head comparison.

## 4. Compared Systems

### 4.1 Selected-Axis AdaTAD

The official VideoMAE-S adapter consumes K irregularly selected RGB frames.
The original ActionFormer projection and head treat the outputs as positions
0 through K-1. Ground truth is remapped to that selected-rank axis only for
this baseline.

This baseline measures the failure mode of ignoring the original timeline.

### 4.2 Physical-Grid ActionFormer AdaTAD

The same K frames and the same VideoMAE-S adapter are used. The existing
physical-grid ActionFormer assignment receives the original dense positions
and keeps ground truth on the original temporal axis.

This is the strongest low-change baseline and should reuse the existing
physical-grid ActionFormer implementation instead of duplicating it.

### 4.3 PhysTime-AdaTAD

The same K frames and VideoMAE-S adapter produce a tensor shaped [B, 384, K].
`PhysTimeMeasureProjection` consumes these observations together with
timestamps and support intervals in seconds. `PhysTimeHead` performs
classification, endpoint prediction, and boundary regression directly in
seconds.

Predictions remain in seconds through window merging and NMS. They must not be
converted through a selected-rank inverse map.

### 4.4 Dense AdaTAD Reference

The unmodified 768-frame official AdaTAD configuration is a reference point,
not part of the head-isolation comparison. It establishes the upper compute
and accuracy anchor.

## 5. Raw-Frame Sampling Contract

1. A logical window contains up to 768 original timeline positions.
2. The sampler chooses K sorted positions without reading GT, actionness,
   predictions, or learned scores.
3. `DecordDecode` receives only the selected frame indices, so unselected RGB
   frames are not decoded and never enter VideoMAE.
4. The sampler records both absolute video frame indices and window-relative
   dense positions.
5. Padding repeats the last valid frame only after the valid selected prefix;
   the mask remains prefix-valid.
6. Train, validation, and test use deterministic sample keys so that all head
   variants receive bit-identical frame indices.
7. The primary K=384 comparison uses the existing deterministic
   `random_fixed_subsample` policy. Bursty and contiguous-gap policies are
   robustness experiments after the primary gate.
8. Standard AdaTAD training crop selection may use annotations, but the sparse
   selection inside the accepted crop must remain GT-independent. The paper
   must state this distinction explicitly.

The selected-index checksum must be emitted during the real-data gate and must
match across all three sparse-head configurations.

## 6. Physical-Time Metadata

A new raw-frame geometry transform must run after `LoadFrames` has selected the
indices and before `Collect` removes metadata. It produces:

- `phystime_timestamps_sec`;
- `phystime_support_intervals_sec`;
- `phystime_duration_sec`;
- `phystime_domain_start_sec` and `phystime_domain_end_sec`;
- `phystime_support_provenance=original_raw_frame_cells`;
- `prediction_time_unit=seconds`;
- `gt_time_unit=seconds` for training and validation;
- `irregular_native_axis=True`;
- `remap_gt_to_selected_axis=False`;
- selected-index and no-GT audit flags.

Each valid observation timestamp is its original frame index divided by the
video FPS. Its support is the original one-frame cell clipped to video bounds.
Sparse gaps are not assigned to neighboring frames and support intervals must
not be inflated into Voronoi cells.

Ground-truth segments are converted once from the current crop/window axis to
video-absolute seconds. Double conversion and selected-axis conversion are
fail-closed errors.

## 7. AdaTAD Backbone Contract

PhysTime-AdaTAD must reuse the official configuration:

- `mmaction.Recognizer3D`;
- `VisionTransformerAdapter` with VideoMAE-S initialization;
- 16-frame chunks;
- the existing preprocessing and spatial augmentation pipeline;
- official temporal adapters and checkpoint behavior.

K must be divisible by 16. For K=384, the preprocessing uses 24 chunks. The
postprocessing output length is K, not 768; no dense temporal reconstruction
is allowed in the PhysTime branch.

The official optimization policy is preserved: the VideoMAE trunk has learning
rate zero, temporal adapters are trainable, and projection/head parameters are
trainable. Detector loss must backpropagate through the backbone computation
and produce nonzero gradients for adapters, PhysTime projection, and all head
branches.

## 8. Loss and Training Contract

The primary comparison uses only the native detection supervision of each
head. PhysTime-AdaTAD includes classification, boundary regression, and
endpoint losses. No discretization-consistency term is enabled in the primary
comparison because it would change the supervision budget.

All three K=384 sparse variants share:

- the same pretrained VideoMAE-S checkpoint;
- the same dataset split and raw videos;
- the same selected indices;
- the same spatial augmentation policy;
- the same 60-epoch workflow and validation cadence;
- the same optimizer schedule;
- the same seed;
- the same NMS and evaluation implementation.

The official optimizer policy remains adapter learning rate 2e-4 and detector
learning rate 1e-4 unless the base AdaTAD configuration changes both for every
head variant.

## 9. Required Configurations

- `phystime_adatad_sparse_k384.py`;
- `physical_grid_adatad_sparse_k384.py`;
- `selected_axis_adatad_sparse_k384.py`;
- a one-step execution config for each variant;
- the existing dense 768 AdaTAD configuration as reference.

The physical-grid configuration must inherit or reuse the existing registered
head behavior. The selected-axis configuration must use the unmodified
ActionFormerHead. The PhysTime configuration must use `PhysTimeTAD`,
`PhysTimeMeasureProjection`, and `PhysTimeHead` with the official AdaTAD
backbone block.

## 10. Mandatory Tests and Gates

### 10.1 Data Contract Tests

- Exactly K raw frame indices are produced for a full 768-frame window.
- Frame indices are sorted, in range, and identical across head variants.
- `DecordDecode` receives only K indices.
- Timestamps and supports match the selected raw frame indices and FPS.
- GT conversion to absolute seconds is correct for training crops and sliding
  windows.
- Validation and test sampling do not inspect GT.

### 10.2 Model Contract Tests

- All three formal configurations build from the registry.
- The PhysTime backbone output is [B, 384, K].
- The PhysTime branch never interpolates back to 768.
- PhysTime proposals and postprocessing remain in seconds.
- Adapter, projection, classification, regression, and endpoint gradients are
  finite and nonzero after `losses["cost"].backward()`.
- Every trainable detector parameter and every configured adapter parameter is
  covered exactly once by the optimizer.

### 10.3 Real THUMOS Gate

One real THUMOS training sample must complete decode, forward, backward, and
inference on CUDA. The gate records selected-index checksum, decoded frame
count, feature length, losses, gradient proofs, peak memory, and runtime.

Failure of this gate prevents full training jobs from starting.

## 11. Experiment Sequence

Phase 0 runs unit, registry, CUDA synthetic, and real-video one-step gates.

Phase 1 runs the matched K=384 head comparison:

1. selected-axis AdaTAD;
2. physical-grid ActionFormer AdaTAD;
3. PhysTime-AdaTAD.

Phase 2 is released only if PhysTime-AdaTAD is stable and competitive. It adds
K=192/384/768 curves and random, bursty, and contiguous-gap robustness.

Multi-seed runs are released only after the primary head comparison produces a
result-to-claim decision.

## 12. Success Criteria and Claim Boundary

The main claim is supported only if PhysTime-AdaTAD improves localization,
especially mAP@0.6 and mAP@0.7 or boundary error, over both sparse-head
baselines under identical selected frames, without erasing the compute savings
of processing K rather than 768 raw frames.

If PhysTime only beats selected-axis but not physical-grid ActionFormer, the
claim must be reduced to evidence that original-time geometry matters; it does
not establish the necessity of the full PhysTime head.

If sparse PhysTime does not remain competitive with the dense reference at
K=384, it cannot be the paper main method without a stronger accuracy-cost
trade-off result.

## 13. Cancellation of the Feature Track

The I3D feature-token jobs are cancelled and are not paper evidence. Their code
remains as focused unit and feature-input diagnostics, but no future launcher
may describe them as PhysTime-AdaTAD or as a raw-video end-to-end experiment.
