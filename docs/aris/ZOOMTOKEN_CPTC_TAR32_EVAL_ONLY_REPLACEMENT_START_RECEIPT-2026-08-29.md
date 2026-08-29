# ZoomToken CPTC TAR32 evaluation-only replacement start receipt

## Authority and counters

- task: `ZT-CPTC-TAR32-TERMINAL-001`
- action: `RPL1_EVALUATION_ONLY_COMPLETION`
- authority request: `PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001`
- replacement for: Slurm job `1261121`
- scheduler submissions before replacement: `1`
- scientific evaluation attempts before replacement: `0`
- authorized scheduler ordinal: `2`
- authorized scientific attempt ordinal: `1`
- third submission: forbidden

## Frozen identities

- candidate/source: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`, clean and equal to
  `origin/codex/zoomtoken-r1-tar32-fkv-v001`
- source root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_src_b0a1ca11`
- checkpoint: epoch-59 `state_dict_ema`
- checkpoint SHA: `fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`
- config SHA: `b372d759c402bd82dbc758faa4b69e89351d757e57c8f76d1369f5fee7edc8ec`
- annotation SHA: `ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad`
- class-map SHA: `a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31`
- pretrained SHA: `4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`
- population: canonical validation, 211 videos / 792 ordered items
- evaluator/postprocess: official evaluator / configured Soft-NMS
- resources: two GPUs / eight CPUs
- training/resume/parameter update: false / false / false

## Builder checks

- local launcher SHA: `5b157901598782aeb62a95803ff4f8955c8402bfdcfcd2d1d6f9acf89b46e34e`
- `bash -n`: PASS
- launcher diff `3ade5500..e311804f`: exactly one scientific execution line,
  replacing top-level regular-file counting with recursive `find -L`
- remote source HEAD / tracking / clean tree: exact / exact / `0` dirty entries
- remote five SHA checks: all exact
- canonical `find -L` MP4 count: `411`
- frozen result root: absent
- command surface: `tools/test.py`; no train/resume/optimizer/parameter update
- `sbatch --test-only`: accepted; projected ID `1261139` has no `sacct` record and
  is not a submission

## Independent change-surface Critic

Verdict: `PASS`. The Critic verified the only launcher difference at
`scripts/run_zoomtoken_r1_tar32_fkv_eval_only_n16r4.sh:69`; candidate,
checkpoint, config/data identities, evaluator/Soft-NMS, two-GPU/eight-CPU
resources, and no-training semantics remain unchanged. Documentation changes at
`e311804f` do not alter the deployed scientific surface.

## Formal replacement submission

Result-blind Evaluator returned `PRE_RUN_READY_REPLACEMENT` with no blocker. It
independently verified the frozen candidate/checkpoint and five SHA identities,
canonical 411 inventory, fresh root, official evaluator/Soft-NMS, two-GPU/eight-CPU
resources, no-training semantics, replacement counters, and frozen cost/successor
boundaries.

- replacement Job ID / name: `1261142` / `zt-r1-tar32-eval-b0a1`
- replacement for: `1261121`
- submitted / started: `2026-08-29 21:11:36+08:00` / `21:11:38+08:00`
- initial allocation: `g0067`, two GPUs, eight CPUs, `124400M`, eight-hour limit
- result root:
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_eval_only_b0a1ca11_seed42_20260829`
- log root:
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_eval_slurm_logs_b0a1ca11_20260829`
- scheduler submission ordinal / scientific attempt ordinal: `2` / `1`
- terminal watcher: FastCtx background job `j-sdfhnd`, 300-second terminal-only check

No live or partial accuracy, prediction, cost, power, route or boundary value was
read at submission. The watcher owns waiting; Codex does not foreground-poll and
does not create a third submission, cost job, successor, retry or resume.
