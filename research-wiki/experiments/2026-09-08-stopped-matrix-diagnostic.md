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
