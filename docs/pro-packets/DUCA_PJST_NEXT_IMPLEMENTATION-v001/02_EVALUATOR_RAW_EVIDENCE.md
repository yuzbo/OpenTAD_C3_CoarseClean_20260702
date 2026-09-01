# EVALUATOR_PJST_D1_CANONICAL_PRETRAIN_TERMINAL-v006

status: TERMINATOR_BLOCKED / EVIDENCE_INADMISSIBLE
evidence_class: OFF_ON_PREDICTIONS_SEALED / POINTS_EXACTLY_REPRODUCED / PAIRED_BOOTSTRAP_NOT_EXECUTED
evaluator_task: 01a042c6-becf-7540-b324-5f1a82208715
local_cwd: C:/Users/skywalker/.codex/worktrees/36b0/OpenTAD_C3_CoarseClean_20260702
local_revision: 7bd120f0d342bf175c97c365fba7cbd359df055e
local_clean: true
remote_checkout: /data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_7bd120f_20260827
remote_revision: 7bd120f0d342bf175c97c365fba7cbd359df055e
remote_clean: true

## Admitted predecessor boundary

- Pro decision: `REVISE_ADMITTED`.
- Independent Critic: `TERMINATOR_STATIC_PASS` on exact `7bd120f0d342bf175c97c365fba7cbd359df055e`.
- Jobs `1257283` and `1257284` both failed during model construction, before terminal EMA loading or official inference, with the first causal error `FileNotFoundError: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth can not be found.`
- Job `1257285` remains dependency-blocked by those failed jobs. None of `1257283/1257284/1257285` is scientific evidence.

## Submission-time admission checks

- Fresh remote checkout and all four v3 output roots were absent before deployment/submission.
- Exact remote revision/clean gate: PASS.
- Canonical pretrain, Stage-1 checkpoint, OFF/ON epoch-59 checkpoints, annotation, class map and video root: readable.
- Stage-1 validator: epoch `29`, SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`, matched OFF/ON config contract: PASS.
- Finalizer shell syntax: PASS.
- `python -m pytest tests/test_duca_pjst_d1_terminal_finalizer.py -q`: `4 passed`.
- Existing read-only `PRECHECK_ONLY=1` validator path: `31 passed`; no training, inference or GPU work ran on the login node.
- Remote checkout remained clean after checks.

## Frozen bindings

- OFF config: `configs/adatad/thumos/duca_pjst_d1_stage2_off.py`
- ON config: `configs/adatad/thumos/duca_pjst_d1_stage2_on.py`
- OFF checkpoint: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/gpu1_id0/checkpoint/epoch_59.pth`
- ON checkpoint: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/gpu1_id0/checkpoint/epoch_59.pth`
- terminal checkpoint state: `state_dict_ema`, expected epoch `59`
- Stage-1 checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- Stage-1 identity: epoch `29`, SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- canonical VideoMAE-S pretrain: `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`
- video root: `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`
- annotation / official ground truth: `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`
- class map: `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`
- evaluator: existing `tools/test.py` official THUMOS14 validation mAP evaluator, tIoU `[0.3,0.4,0.5,0.6,0.7]`
- unchanged NMS: soft-NMS `sigma=0.7`, `max_seg_num=2000`, `multiclass=True`, `voting_thresh=0.7`
- inference overrides: `post_processing.save_dict=True`, `inference.load_from_raw_predictions=False`
- seed: `3407`; selector budget: `K384`
- bootstrap: existing `tools/bata/bootstrap_duca_h65_official_map.py` plus `merge_duca_h65_bootstrap_shards.py`; OFF baseline, ON candidate; nonce `DUCA-PJST-D1-OFF-ON-v001`; namespace `PAIRED_VIDEO_BOOTSTRAP_V1`; 16 contiguous shards covering `[0,10000)`; nearest-rank 250/9750 merge.

## Exact Slurm argv and accepted jobs

OFF job `1257897`:

```text
sbatch --parsable --gpus=1 --account=sczc063 --partition=gpu --qos=normal --export=ALL,DUCA_REPO_ROOT=/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_7bd120f_20260827,DUCA_EVAL_COMMIT=7bd120f0d342bf175c97c365fba7cbd359df055e,DUCA_PJST_ARM=OFF,DUCA_EVAL_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval_v3,DUCA_STAGE1_CHECKPOINT=/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth,DUCA_STAGE1_CHECKPOINT_SHA256=bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3,DUCA_STAGE1_CHECKPOINT_EPOCH=29 scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch
```

ON job `1257898`:

```text
sbatch --parsable --gpus=1 --account=sczc063 --partition=gpu --qos=normal --export=ALL,DUCA_REPO_ROOT=/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_7bd120f_20260827,DUCA_EVAL_COMMIT=7bd120f0d342bf175c97c365fba7cbd359df055e,DUCA_PJST_ARM=ON,DUCA_EVAL_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/terminal_eval_v3,DUCA_STAGE1_CHECKPOINT=/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth,DUCA_STAGE1_CHECKPOINT_SHA256=bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3,DUCA_STAGE1_CHECKPOINT_EPOCH=29 scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch
```

Bundled finalizer job `1257899`, dependency `afterok:1257897:1257898`:

```text
sbatch --parsable --gpus=1 --cpus-per-task=4 --account=sczc063 --partition=gpu --qos=normal --dependency=afterok:1257897:1257898 --export=ALL,DUCA_REPO_ROOT=/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_7bd120f_20260827,DUCA_BOOTSTRAP_COMMIT=7bd120f0d342bf175c97c365fba7cbd359df055e,DUCA_BOOTSTRAP_SHARD_ROOT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/bootstrap_shards_v3,DUCA_BOOTSTRAP_OUTPUT_ROOT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/bootstrap_v3,DUCA_BOOTSTRAP_WORKERS=4,DUCA_PJST_OFF_PREDICTION=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval_v3/work/result_detection.json,DUCA_PJST_ON_PREDICTION=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/terminal_eval_v3/work/result_detection.json scripts/run_duca_pjst_d1_terminal_bootstrap_bundle_n16r4.sbatch
```

## Output roots

- OFF: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval_v3`
- ON: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/terminal_eval_v3`
- bootstrap shards: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/bootstrap_shards_v3`
- bootstrap final: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/bootstrap_v3`

## Resources, estimate and handoff

- OFF/ON: one GPU each, scheduler-default one CPU and 15,550 MiB requested memory, eligible to run concurrently; expected roughly 2-4 hours after allocation for each full 211-video pass.
- bundled finalizer: one GPU allocation, four CPUs, 62,200 MiB requested memory, four bounded local workers; expected roughly 6-12 hours after dependency release. Estimates are operational only and do not affect admission.
- current_scientific_question: Does PJST-D1 have a statistically admitted negative or uncertain whole-video population effect under the fixed H65 K384 matched Stage-2 estimand?
- next_owner: same evaluation-only Evaluator
- next_action: monitor exactly jobs `1257897`, `1257898`, `1257899` to terminal; admit only sealed 211/211 identity, per-metric point reproduction within `1e-6 pp`, and complete 10,000-draw paired output.
- dependency: `afterok:1257897:1257898`, then bundled finalizer `1257899`
- expected_return_at: `2026-08-28T10:30:00+08:00`
- single_recovery: none

## Terminal update — 2026-08-27T19:03:12+08:00

### Scheduler terminal state

- OFF job `1257897`: `COMPLETED 0:0`, `2026-08-27T18:42:09+08:00` to `2026-08-27T19:02:03+08:00`, elapsed `00:19:54`, node `g0006`.
- ON job `1257898`: `COMPLETED 0:0`, `2026-08-27T18:42:09+08:00` to `2026-08-27T19:03:03+08:00`, elapsed `00:20:54`, node `g0006`.
- bundled finalizer job `1257899`: `FAILED 1:0`, `2026-08-27T19:03:09+08:00` to `2026-08-27T19:03:12+08:00`, elapsed `00:00:03`, node `g0006`.
- No resubmission, path mutation, symlink, code change, retraining, resume, second bundle or correction loop occurred.

### OFF/ON prediction and evaluator seals

- OFF: `video_count=211`, `result_count=422000`, prediction SHA256 `351a81758baf274f98ce27e5524536b5a524d43487f016bd6c2c7c13f8c0b76a`, checkpoint SHA256 `c3954b2aca2030d2696ee146826b387d2db7b8b8dfee27dd494e520679113a29`.
- ON: `video_count=211`, `result_count=422000`, prediction SHA256 `f00a3a425d04536b5ceebb5119d1c531bd6dd2fe256b4db67acc88251e013cb9`, checkpoint SHA256 `bde2943bcaca23b81dbe6a4a08630e672ccdc17bbaec171d83ed559e42182e08`.
- Video identity: OFF `211/211`, ON `211/211`, exact video-ID sets identical; OFF-only `[]`, ON-only `[]`.
- Annotation SHA, class-map SHA, evaluation-config SHA and official evaluator source SHA are identical across arms. Evaluator source SHA256 is `e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`.
- Both sidecars bind exact commit `7bd120f0d342bf175c97c365fba7cbd359df055e`, epoch `59`, state key `state_dict_ema`, official validation, and their arm-specific frozen checkpoint/config.

### Per-metric point reproduction

The v3 values are bit-for-bit equal to the original `gpu1_id0/intermediate_validation/epoch_060_ema.json` values for every metric; reproduction error is exactly `0 pp` for each arm/metric, within the frozen `1e-6 pp` tolerance.

| Metric | OFF v3 (%) | ON v3 (%) | ON-OFF (pp) | Reproduction |
|---|---:|---:|---:|---|
| mAP@0.3 | 80.04698811399541 | 79.25176713703161 | -0.79522097696380 | exact, 0 pp error |
| mAP@0.4 | 75.56871468951523 | 74.31627020458057 | -1.25244448493466 | exact, 0 pp error |
| mAP@0.5 | 68.02175107729447 | 67.87476663595304 | -0.14698444134143 | exact, 0 pp error |
| mAP@0.6 | 58.03293530737000 | 57.74244005311084 | -0.29049525425916 | exact, 0 pp error |
| mAP@0.7 | 43.64602698293250 | 43.76876582551604 | +0.12273884258354 | exact, 0 pp error |
| Average mAP | 65.06328323422153 | 64.59080197123842 | -0.47248126298311 | exact, 0 pp error |

### First real failure and statistical boundary

- First causal failure line: `[PJST_D1_BUNDLE][FAIL] missing prediction: /data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval_v3/work/result_detection.json`.
- First failed binding field: `DUCA_PJST_OFF_PREDICTION`. The sole accepted bundle argv inherited the v005 flat path `.../work/result_detection.json`, while the sealed evaluator output is `.../work/gpu1_id0/result_detection.json`. The ON sealed output has the same `gpu1_id0/` layout; the launcher failed on OFF first.
- This is a deterministic bundled-finalizer transport failure after both valid inference jobs, not a model result and not a paired-statistics result.
- Bootstrap completeness: shard root absent; `0/16` shards; `0/10000` paired whole-video replicates; final `paired_bootstrap.json` absent.
- Frozen CI: unavailable. Frozen PASS/KILL/gate: unavailable and must not be inferred or changed post hoc.
- Statistical conclusion boundary: the exactly reproduced point estimate remains ON-OFF Average mAP `-0.47248126298311 pp`, with no positive support at the average. Without the preregistered paired interval it is not a formally admitted negative whole-video population result. The isolated mAP@0.7 `+0.12273884258354 pp` cannot be interpreted as a benefit.

### Terminal handoff

- current_scientific_question: Does PJST-D1 have a statistically admitted negative or uncertain whole-video population effect under the fixed H65 K384 matched Stage-2 estimand?
- next_owner: DUCA Coordinator terminal hold
- next_action: ingest the sealed 211/211 OFF/ON predictions and exact point reproduction, but quarantine the missing paired interval; do not resubmit, repair, issue a gate, or form a second engineering loop under this task.
- dependency: none; this authorized terminal DAG is finished and blocked.
- expected_return_at: terminal now
- single_recovery: none
