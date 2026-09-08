# User-Stopped D2S and PA-TAD Diagnostic Evaluation

The user requested stopping both remaining training jobs and proceeding directly
to testing on 2026-09-08. Slurm confirms D2S `1276621` and PA-TAD `1276622` were
cancelled at 17:15:04 Asia/Shanghai. Do not automatically resume or replace them.

## Frozen Scope

| Method | Matched seeds | Evaluated arms | Evaluation cells |
| --- | --- | --- | --- |
| Feature-change-driven dual-resolution temporal refresh (D2S) | 4407, 4408 | D160, G96, D2S-U128-B128 | 6 |
| Global-only coarse pyramid with local residuals at fine scales (PA-TAD) | 4407 | D160, G96, PATAD-U128-B128 | 3 |

Selection is by full-training terminal artifact availability at the user stop,
not by accuracy. D2S seed 4409 and PA-TAD seeds 4408/4409 are excluded from matched
comparison. D2S has 8/9 completed training cells; PA-TAD has 7/9. Unpaired completed
baseline seeds remain in the original roots. No checkpoint or recovery file is
deleted. No shortened-training weights are evaluated as final weights.

Each selected terminal receipt states 200 training video identities, 60 epochs,
6000 successful updates and final epoch-59 EMA. PRECHECK additionally requires
every stored AdamW parameter step to be 6000, matching checkpoint/receipt/sample
order identities, strict EMA loading, the full 792-window loader and one real
label-free forward per cell. Successful PRECHECK is recorded before testing.

Full population means every one of the 200 training videos participates in each
epoch under the frozen random-truncation pipeline. It does not mean every frame
of each entire video is processed each epoch: training samples 768-frame temporal
clips. Evaluation uses all 211 frozen held-out videos and 792 ordered windows.

## Source And Artifact Locations

- Original training commit: `21aa2945b934a0dba469a517c224efe9b30d3967`.
- Original immutable source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_d2s_patad_21aa2945_src`.
- D2S training root: `/data/run01/sczc063/yuzibo/projects/d2s_tad_full200_compute_21aa2945_formal_r5`.
- PA-TAD training root: `/data/run01/sczc063/yuzibo/projects/patad_full200_compute_21aa2945_formal_r4`.
- Final checkpoint: `<training-root>/work_dirs/<arm>_seed<seed>/checkpoint/epoch_59.pth`.
- Training receipt: `<training-root>/work_dirs/<arm>_seed<seed>/training_terminal_receipt.json`.
- Local evaluation source: `E:/DeskTop/TAD/zoomtoken_stopped_matrix_eval_20260908`.
- GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-stopped-matrix-eval-20260908
- Entry: `tools/bata/zoomtoken_stopped_matrix_eval.py`.
- Launcher: `scripts/submit_zoomtoken_stopped_matrix_eval_n16r4.sbatch`.

Evaluation writes only to new diagnostic roots. It preserves the training
identity and separately records the execution commit. Model, config and dataset
implementation remain identical to the training commit. The shared inference
helper binds the original absolute pretrained path before strict final EMA load.

## Interpretation Limits

These are full-training weights but incomplete original three-seed matrices.
The diagnostic freezes all selected cells before metrics, completes all their
label-free predictions, then opens the full metric annotation in a separately
named diagnostic barrier. It reuses unchanged NMS, official mAP with point-metric
parity, short-action recall and boundary-error definitions. It does not run the
three-seed bootstrap or issue a formal matrix/paper admission verdict. Available
point results cannot substitute for the cancelled complete experiment plan.

Original nine-cell checkpoint and prediction seals remain enforced for formal
runs. A six- or three-cell diagnostic must never be passed off as a nine-cell
completion. Existing negative evidence in other routes remains unchanged.

## Deployment Receipt, 2026-09-08

- Evaluation runtime commit: `1f600f8fa9c696c0d476970b353a05f7e004a05f`.
- Evaluation immutable source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_stopped_matrix_eval_1f600f8f_src`.
- GitHub ref, exact remote HEAD and clean status were verified through the
  prescribed proxy. Model and config trees are unchanged from training.
- D2S PRECHECK `1280173` is pending for Priority; evaluation `1280175` is pending
  on `afterok:1280173` with invalid dependencies cancelling the test job.
- PA-TAD PRECHECK `1280174` is pending for Priority; evaluation `1280176` is
  pending on `afterok:1280174` with the same cancellation policy.
- D2S diagnostic root: `/data/run01/sczc063/yuzibo/projects/d2s_user_stop_diagnostic_20260908_1f600f8f_r1`.
- PA-TAD diagnostic root: `/data/run01/sczc063/yuzibo/projects/patad_user_stop_diagnostic_20260908_1f600f8f_r1`.
- All jobs request one Slurm GPU, eight CPU cores. PRECHECK has a 30-minute
  limit; evaluation eight hours. No fixed physical GPU index is assigned.
- Local tests: 26 shared/diagnostic checks, 25 D2S-context checks, 21 PA-TAD-context
  checks, and 20 prescribed C3 regression checks passed (overlapping suites).
  Python compilation, both Bash syntax checks and git diff checks passed.
- Actual checkpoint-state and GPU-forward PRECHECK is still pending. Existing
  full-training terminal receipts are verified; do not claim GPU PRECHECK PASS.
- Automation `zoomtoken-experiment-monitor` is ACTIVE every 30 minutes and bound
  to the four jobs above. It must not restart the cancelled training jobs.

The deployment statuses above are a dated snapshot, not final accuracy results.
`control/diagnostic_plan.json` records successful runtime admission;
`diagnostic_results.json` records completion of the selected diagnostic set.

## Correctness Audit And Replacement, 2026-09-08 18:23 Asia/Shanghai

The first PRECHECK jobs 1280173/1280174 failed with `module: command not found`
before model execution; dependent jobs 1280175/1280176 were cancelled without
starting. Those failed namespaces and original training artifacts are preserved.

The replacement diagnostic source is `d34d51d29900532a399c5a1edc0f2563c5b3641c`,
clean and pushed on the same GitHub branch. It initializes Environment Modules,
matches official AP tie ordering and corrects future trainable-parameter metadata.
Model, configuration and NMS code are unchanged. All nine selected real checkpoint
AdamW states equal 6000, EMA tensors are finite, and CPU strict loading passed.

| Route | GPU PRECHECK | Diagnostic evaluation |
| --- | --- | --- |
| D2S, six cells | 1280197 COMPLETED 0:0 | 1280198 RUNNING on g0006 |
| PA-TAD, three cells | 1280199 COMPLETED 0:0 | 1280200 RUNNING on g0050 |

Both PRECHECK jobs performed one real label-free GPU forward for each selected
cell and wrote their admission plans. Full 792-window evaluation is not yet
complete. No final metric or original three-seed completion is claimed.

- Immutable source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_stopped_matrix_eval_d34d51d2_src`.
- D2S output: `/data/run01/sczc063/yuzibo/projects/d2s_user_stop_diagnostic_20260908_d34d51d2_r2`.
- PA-TAD output: `/data/run01/sczc063/yuzibo/projects/patad_user_stop_diagnostic_20260908_d34d51d2_r2`.
- The existing 30-minute monitor is ACTIVE and bound to these replacement jobs.

An additional audit fix corrects the original formal nine-cell CLI's file-digest
handoff. That separate entry is not called by these running diagnostics; the
immutable d34d51d2 runtime need not be replaced for this fix. Full findings,
reproductions, model/data-flow analysis, corrected parameter counts and remaining
limits are in `audits/2026-09-08-stopped-matrix/EXPERIMENT_AUDIT.md`.

## Monitor Snapshot, 2026-09-08 19:00 Asia/Shanghai

| Route | PRECHECK | Evaluation | Observed elapsed |
| --- | --- | --- | --- |
| Feature-change-driven dual-resolution temporal refresh (D2S) | 1280197 COMPLETED 0:0 | 1280198 RUNNING, g0006 | 41m12s |
| Global-only coarse pyramid with local fine-scale residuals (PA-TAD) | 1280199 COMPLETED 0:0 | 1280200 RUNNING, g0050 | 41m12s |

Both diagnostic roots contain their admission plans but no published final
`diagnostic_results.json`. Evaluation stderr contains only successful environment
module loads. The stdout tail contains the initial Kinetics-pretraining
classification-head/adapter-key warning and dependency deprecation warnings,
not a failure of strict final EMA loading. The inference loop publishes results
after a complete cell and does not print per-window progress. No exact current
window count can be inferred from these logs. Slurm batch accounting reports
nonzero CPU work (1h56m42s / 1h55m18s) and about 21.6 GB peak RSS per job; these
are operational observations, not benchmark claims or evidence of completion.
No new failure or actionable state change was found. No repair, resubmission,
training restart, metric-GT opening or source modification was performed.

## Prediction Serialization Failure, 2026-09-08

The preceding healthy-running snapshot is superseded by terminal evidence:
D2S evaluation 1280198 FAILED 1:0 at 19:05:31 after 46m51s; PA-TAD evaluation
1280200 FAILED 1:0 at 19:04:21 after 45m41s (Asia/Shanghai). Both reached
`build_prediction_bundle_payload` after the first D160 seed-4407 inference/NMS
pass, then raised `prediction must have finite score and positive duration`.
Neither published a prediction bundle, metric result, or diagnostic GT-open
marker. The candidate methods were not reached; no accuracy conclusion follows.

The old exception did not include the offending values, so its exact runtime
row cannot be recovered from these logs. A bounded CPU reproduction with the
unchanged real SingleStageDetector clipping/rounding and official Soft-NMS
produces finite zero-duration detections and the same serializer exception.
Official clipping can map both endpoints to 0 or video duration; rounding to
0.01 s can also collapse short intervals. The official evaluator retains these
as false positives. The task-local positive-duration check was incompatible.

Correction: retain finite zero-duration predictions unchanged, while continuing
to reject reversed intervals and nonfinite values, now with the prediction UID
and numeric values in the error. No filtering, score changes, model/config/NMS
changes or official-evaluator changes are made. PRECHECK now executes the same
post-NMS and in-memory bundle validation for its one real input window before
returning success; it does not publish that partial-window bundle as a complete
prediction artifact. The previous forward-only PRECHECK missed this stage.

Local evaluator/diagnostic/C3 tests: 53 passed. Regression cases show an early
zero-duration false positive lowers AP to 50%, matching the official evaluator;
discarding it would incorrectly yield 100%. A production CPU postprocessing
test preserves all four synthetic predictions, including three collapsed ones.
The monitor is temporarily paused during repair to prevent duplicate submission.
Original failed outputs and full-training weights remain untouched. Replacement
GPU PRECHECK and diagnostic jobs must use a new immutable source/output root.

## Serialization Repair Deployment, 2026-09-08

- Pushed runtime commit: `2a8639b6650cf3b04d246fe1efc9380e913ec341`.
- Immutable source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_stopped_matrix_eval_2a8639b6_src`.
- GitHub ref, remote HEAD and clean status verified through the required proxy;
  `opentad/` and THUMOS model configurations are unchanged from training.
- Remote production postprocessing/diagnostic/statistics regressions: 34 passed.
- D2S PRECHECK 1280308 and evaluation 1280309 accepted by Slurm, with
  `afterok:1280308` and cancellation if that dependency fails.
- PA-TAD PRECHECK 1280310 and evaluation 1280311 accepted by Slurm, with
  `afterok:1280310` and the same failure policy.
- D2S output: `/data/run01/sczc063/yuzibo/projects/d2s_user_stop_diagnostic_20260908_2a8639b6_r3`.
- PA-TAD output: `/data/run01/sczc063/yuzibo/projects/patad_user_stop_diagnostic_20260908_2a8639b6_r3`.
- Requests remain one Slurm GPU/eight CPU cores per job, 30 minutes per PRECHECK
  and eight hours per evaluation. No training restart or GPU-index override.
- The same 30-minute monitor is ACTIVE and bound to these replacement jobs.

Submission is not PRECHECK success or a final metric. New runtime admission must
confirm `post_nms_serialization_checked: true` for each of the nine selected
one-window PRECHECK witnesses. Full evaluation still requires every selected cell
to finish all 211 videos/792 windows and publish the diagnostic result file.

Post-submission Slurm snapshot: PRECHECK 1280308/1280310 are PENDING (Priority);
evaluations 1280309/1280311 are PENDING (Dependency). No replacement GPU PRECHECK
or evaluation has started at this snapshot. Queueing is not submission failure.
