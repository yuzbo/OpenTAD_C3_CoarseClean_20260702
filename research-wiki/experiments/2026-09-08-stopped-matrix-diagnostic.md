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
