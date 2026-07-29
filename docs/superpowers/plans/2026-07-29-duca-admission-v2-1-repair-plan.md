# DUCA Admission v2.1 adjudication and repair plan

Date: `2026-07-29`

Status: `partially_decided / formal_experiment_no_go`

External review: `U-PRO-ADMISSION-V21-1`

## Decision

The pure selected-axis pre-backbone model remains the paper mainline. The
current Admission v2 formal numeric calibration and admission paths are
superseded because they use synthetic head-only fixtures and cannot establish
real-video, full-model, process-grouped equivalence. They may not authorize
Phase 1.

Admission v2.1 is the only candidate replacement, but the external proposal is
not executable verbatim. Four scientific or physical contracts must be
corrected and frozen before the full implementation or Slurm DAG is released.

## Accepted without change

1. Keep the selected-axis plugin and unchanged standard detector head.
2. Treat the physical-time head only as a separately named integration
   diagnostic.
3. Use real training-only videos, disjoint scale-fit/calibration/holdout roles,
   plugin and independently materialized reference executors, full train/eval
   taps, production AMP, independent processes, immutable raw distributions,
   and success/failure receipts.
4. Keep all windows from one video in one statistical cluster and account for
   process dependence.
5. Never let FP32 diagnostic replay rescue a production-AMP failure.
6. Keep Phase 4 and official-final sealed.

## Independently verified blocker: the proposed window contract is impossible

The frozen split manifest
`41349cd39a6a550b6e1613de968577b1605c93902edd52a88309121b9e90c057`
contains 100 `detector_selector_train` videos, so `3 x 32` disjoint roles are
numerically possible.

However, the current `SlidingWindowDataset.split_video_to_windows` contract
back-shifts the terminal window to length 768 whenever a video has at least
768 snippets. Therefore a source video has either:

- one or more natural full windows and no natural short window; or
- one natural short window and no natural full window.

The immutable training metadata contains:

- 70 videos with natural full windows only;
- 30 videos with natural short windows only;
- 0 videos with both;
- short strata counts `1-256: 7`, `257-512: 13`, `513-767: 10`.

Thus both “every video has one natural full and one natural short window” and
“at least eight videos in the 1-256 stratum” fail before any candidate output
or decoded-frame experiment. Synthetic cropping, replicated frames, or padding
must not be used to manufacture compliance.

## Decisions still required

### Window coverage

Recommended replacement for discussion:

- retain three disjoint roles of 32 videos;
- select all 30 naturally short videos, ten per role;
- select 66 of 70 naturally long videos, 22 per role;
- stratify short videos by training-metadata quantiles rather than the
  infeasible fixed bins, with deterministic hash tie-breaking;
- require full and short coverage at role level, not within every video.

This replacement uses only pre-candidate training metadata, but it changes the
external proposal and therefore needs an explicit scientific decision.

### Crossed uncertainty

The phrase “resample videos and processes separately” is insufficient for the
sparse balanced incidence table. The protocol must specify an exact
pigeonhole/multiway bootstrap weighting algorithm, the estimand under duplicate
row/column draws, empty-cell behavior, quantile convention, Monte Carlo error,
and deterministic family registry. The proposed catastrophic bound `C_c` is
undefined and cannot be implemented as written.

### Scientific noninferiority margin

`delta_practical = 2 * reporting_quantum` is a serialization-resolution rule,
not a scientific justification for acceptable localization degradation.
Reporting precision and scientific relevance are different quantities. A
margin must be justified independently of candidate/development results by an
explicit effect-preservation or domain-materiality policy. Until such a source
exists, the Phase-1 scientific NI protocol remains blocked.

Calibration variability may be used only as an assay-sensitivity gate; it may
not enlarge the margin.

### Runtime isolation

Slurm `afterok`, clean checkouts, environment allowlists, artifact hashing and
access probes are enforceable from the repository. Hard claims that network is
disabled, forbidden mounts are absent, file access is completely audited, or
an object is storage-locked require cluster/container support outside the
repository. Receipts must distinguish enforced controls from observed
fingerprints and must not claim OS isolation that the job cannot enforce.

## Official-comparability correction

The 100-video `detector_selector_train` pool is a development/calibration role,
not the official training split. The current Phase-4 trainer inherits that
block list and therefore cannot support a comparison with published
OpenTAD/AdaTAD results, even though its evaluation path correctly uses the
complete registered OpenTAD THUMOS evaluation set.

The paper-facing refit must be a new, separately sealed protocol:

1. freeze the method, thresholds, budgets, post-processing and stopping rule
   using training-only development roles;
2. generate leakage-safe out-of-fold utility/risk targets for every one of the
   200 registered training videos;
3. refit the candidate and every trainable control on all 200 videos;
4. preserve the upstream effective global batch of two, 60 epochs and 100
   optimizer updates per epoch (6000 total), using two one-video DDP ranks or a
   separately verified accumulation-equivalent implementation;
5. evaluate exactly once on the complete registered OpenTAD evaluation set
   with the upstream evaluator, temporal-IoU grid, NMS and video-key contract.

OpenTAD intentionally removes two malformed/empty THUMOS test videos and uses
211 evaluation videos. That 211-video registered set is the complete
OpenTAD-comparable evaluation set; it must not be confused with an arbitrary
subset. Until the full-train refit and its exact receipts exist, all 100-video
results remain development evidence and are forbidden from the paper main
table.

The current Phase-4 cell pipeline is hard-disabled at entry so that the
role-scoped trainer cannot accidentally consume the official-final split or
emit a falsely paper-facing result.

## Implemented safe slice

1. The old Admission v2 launcher accepts only `engineering-fixture`.
2. Direct old v2 calibration/admission calls fail closed.
3. Synthetic rows are explicitly labeled `engineering_fixture`,
   `fixture=true`, `admission_effect=false`.
4. Old v2 receipts are parseable only through an explicit historical-read-only
   option and cannot authorize any production entrypoint.
5. A metadata-only v2.1 feasibility auditor mirrors the production sliding
   enumerator, emits a content-bound typed failed receipt, and never authorizes
   Phase 1.

## Correct execution order

1. Merge and code-gate the fail-closed Stage-0 slice.
2. Obtain an explicit decision on the four unresolved contracts above.
3. Freeze a corrected v2.1 design before implementing role assignment,
   windows, targets, independent geometry, full-model workers or statistics.
4. Build and hash the real dataset inventory and data-access evidence.
5. Run scale-fit, calibration and admission holdout in fresh Slurm processes.
6. Finalize Admission v2.1 only if every structural, numeric, identity and
   isolation contract passes.
7. Release Phase-1 v2 only through `afterok` on that exact receipt.

No current artifact is a model-performance result. No paper-admissible
empirical conclusion is available.
