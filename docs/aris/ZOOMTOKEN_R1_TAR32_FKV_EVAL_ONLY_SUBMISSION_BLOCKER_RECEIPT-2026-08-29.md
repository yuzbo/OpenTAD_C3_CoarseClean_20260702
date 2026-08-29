# ZoomToken R1-TAR32-FKV evaluation-only submission blocker receipt

## Scope

- task: `ZT-CPTC-TAR32-TERMINAL-001`
- frozen candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- checkpoint: epoch-59 `state_dict_ema`
- checkpoint SHA-256: `fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`
- formal evaluation-only job: `1261121`
- JobName: `zt-r1-tar32-eval-b0a1`
- submission count under this evaluation-only action: `1`

## Authoritative terminal state

- Slurm state / exit: `FAILED` / `2:0`
- submit / start / end: `2026-08-29 20:26:04` / `20:26:08` / `20:26:12 +08:00`
- elapsed: `00:00:04`
- node / allocation: `g0067`, two GPUs, eight CPUs
- stdout: empty
- stderr: `[ZOOMTOKEN_R1_TAR32_EVAL][FAIL] canonical video inventory is not 411 MP4 files`
- evaluation result root: not created
- prediction/evaluator output: absent
- model/checkpoint load: not reached
- canonical validation loader: not reached
- training/resume/parameter update: not reached and absent

## Root cause

The Builder's result-blind inventory check used a non-recursive, regular-file-only
query at the video-root top level. The canonical inventory is instead 411 valid MP4
symbolic links below the `training` and `validation` subdirectories. Read-only
reconstruction gives:

- top-level regular MP4 files: `0`
- recursive MP4 symbolic links: `411`
- recursive MP4 targets with links followed: `411`

The corrected check is `find -L "$VIDEO_ROOT" -type f -name '*.mp4'`. This changes
only the external launcher admission check; it does not modify the frozen candidate,
config, checkpoint, dataset, evaluator, Soft-NMS, or scientific decision rule.

## Evidence classification and disposition

This is a pre-model engineering/protocol blocker, not TAR32 accuracy, route, boundary,
or efficiency evidence. No partial metric exists and no scientific direction is
inferred. `CPTC-vFinal-20260829` explicitly permits one evaluation-only submission;
because job `1261121` consumed that submission even though scientific execution count
remained zero, Codex does not silently submit a replacement. The exact blocker and
minimal correction must be adjudicated by one fresh Project Pro turn before any new
Slurm action. `ZT-CPTC-RP-K100-v001` remains frozen and unstarted.

## Immutable remote evidence

- launcher commit before submission: `3ade5500`
- deployed launcher SHA-256: `36707ce2f098f92e7762792bb480b68b4b3f66e16addcaaa8460cda5ba7a4548`
- Slurm stdout:
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_eval_slurm_logs_b0a1ca11_20260829/zt-r1-tar32-eval-b0a1-1261121.out`
- Slurm stderr:
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_eval_slurm_logs_b0a1ca11_20260829/zt-r1-tar32-eval-b0a1-1261121.err`
- proposed corrected launcher:
  `scripts/run_zoomtoken_r1_tar32_fkv_eval_only_n16r4.sh`
