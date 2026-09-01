# EVALUATOR_PJST_D1_CYCLE4_N16R4_PRERUN_AND_LAUNCH-v001

- role: independent Evaluator
- process_id: codex-evaluator-pjst-d1-cycle4-n16r4-v001
- evaluation_workspace: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-evaluator-20260826`
- frozen_revision: `c195b97c46acae166e0721fcb412b70221ae7d49`
- frozen_branch: `codex/duca-pjst-cycle4-builder-20260826`
- remote_checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c195b97c_20260826`
- stage1_checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- expected_stage1_sha256: `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- seed: `3407`
- formal_work_root: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c195b97c_20260826`
- accepted_science: frozen/replayed H65 selector, K=384, derivative-only PJST-D1 at first VideoMAE temporal mixing, matched OFF/ON representation attribution

## Objective

Use the shortest target-environment path. Do not add another implementation, contract, smoke harness, or theory round.

1. Establish a new remote checkout at the exact frozen revision without modifying or reusing the terminal Cycle3 checkout.
2. In the canonical N16R4 environment (`/etc/profile`, CUDA 11.8, miniforge3/24.11, `/data/run01/sczc063/yuzibo/conda_envs/opentad`, `PYTHONNOUSERSITE=1`), verify exact HEAD/clean state, Stage-1 checkpoint path/SHA/epoch, run the complete focused PJST pytest, validator, py_compile, and launcher syntax check.
3. `PRE_RUN_READY` requires zero failed and zero skipped focused tests, validator PASS, exact clean revision, exact checkpoint SHA, readable canonical THUMOS/annotation/category/pretrain paths, and no already-active formal job for this exact revision/work root.
4. If and only if `PRE_RUN_READY`, immediately submit exactly two formal Slurm jobs from the same revision and Stage-1 checkpoint:
   - OFF: `MODE=STAGE2_OFF`, `DUCA_STAGE2_WORK_DIR=/data/run01/sczc063/yuzibo/duca_pjst_d1_c195b97c_20260826/off`
   - ON: `MODE=STAGE2_ON`, `DUCA_STAGE2_WORK_DIR=/data/run01/sczc063/yuzibo/duca_pjst_d1_c195b97c_20260826/on`
   - launcher: `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`
   - pass exact `DUCA_REPO_ROOT`, checkpoint path/SHA/epoch, `PYTHONNOUSERSITE=1`; use distinct job names/comments and preserve Slurm-provided `CUDA_VISIBLE_DEVICES`.

The launcher must retain 60 epochs / 6000 successful updates, seed 3407, checkpoint every 5 epochs with resumable state, terminal epoch-59 `state_dict_ema`, official THUMOS paths/evaluator/NMS, and the same selector/RGB/K exposure. Do not access official validation results before terminal evaluation; do not tune or select an intermediate checkpoint.

## Stop rule

On any failed/skipped focused test, checkpoint/config identity mismatch, dirty checkout, missing canonical resource, duplicate exact formal job, or Slurm rejection, stop and return `NEEDS_ATTENTION` with the single objective blocker. Do not patch code, change config, retry a rejected scientific job, or infer efficacy.

## Return

Return exact SSH/runtime identity, remote HEAD/clean state, commands/results, PRE_RUN verdict, Stage-1 binding, OFF/ON job IDs and scheduler states if submitted, exact configs/work roots/seed/resource request/result roots, evidence class, `next_owner`, `next_action`, `dependency`, `expected_return_at`, and one bounded recovery.

