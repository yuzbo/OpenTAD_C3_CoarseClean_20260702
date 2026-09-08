#!/usr/bin/env bash
#SBATCH --job-name=ctdp-terminal-eval
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source /etc/profile
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
cd "${CTDP_EVAL_REPO:?a frozen evaluator checkout is required}"
[[ "$(git rev-parse HEAD)" == "${CTDP_EVAL_COMMIT:?exact evaluator SHA is required}" ]]
[[ -z "$(git status --porcelain)" ]]
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export THUMOS14_ROOT="$BASE/thumos14"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_ROOT="${CTDP_TRAIN_ROOT:?training artifact root is required}"
EVAL_ROOT="${CTDP_EVAL_ROOT:?new evaluation output root is required}"
ARMS=(g0 g1 g2 g3)
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python -m pytest -q tests/test_ctdp_terminal_receipt.py tests/test_ctdp_successful_updates.py tests/test_ctdp_coordinate_contract.py
else
  TASK="${SLURM_ARRAY_TASK_ID:?formal evaluation requires an array index 0-3}"
  [[ "$TASK" =~ ^[0-3]$ ]]
  ARMS=("g$TASK")
fi
for ARM in "${ARMS[@]}"; do
  CFG="configs/adatad/thumos/duca_ctdp_geometry_${ARM}.py"
  WORK="$EVAL_ROOT/$ARM"
  [[ ! -e "$WORK" ]]
  if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
    CKPT="$TRAIN_ROOT/duca_ctdp_geometry_${ARM}_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth"
    MODE=(--ctdp-eval-precheck --not_eval --max-batches 1)
  else
    CKPT="$TRAIN_ROOT/duca_ctdp_geometry_${ARM}_seed3407/gpu1_id0/checkpoint/epoch_59.pth"
    MODE=(--ctdp-metrics-json "$WORK/gpu1_id0/official_eval/metrics.json")
  fi
  [[ -r "$CKPT" ]]
  torchrun --nproc_per_node=1 --master_port="$((31500 + SLURM_JOB_ID % 1000))" \
    tools/test.py "$CFG" --checkpoint "$CKPT" --seed 3407 "${MODE[@]}" \
    --cfg-options work_dir="$WORK" \
    model.backbone.custom.pretrain="$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
done
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo CTDP_ALL_EVAL_PRECHECKS_OK
fi
