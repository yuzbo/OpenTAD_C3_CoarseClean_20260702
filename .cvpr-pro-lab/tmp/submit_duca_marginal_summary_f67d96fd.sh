#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-marg-summary-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/run01/sczc063/yuzibo/duca_marginal_f67d96fd_20260831
#SBATCH --output=/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/recovery-slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/recovery-slurm-%j.err

source /etc/profile
set -euo pipefail

DUCA_BASE=/data/run01/sczc063/yuzibo
DUCA_RECOVERY_SNAPSHOT="$DUCA_BASE/duca_marginal_f67d96fd_20260831"
DUCA_RECOVERY_OUTPUT="$DUCA_BASE/duca_marginal_summary_f67d96fd_20260831"

module load cuda/11.8
module load miniforge3/24.11
source "$DUCA_BASE/conda_envs/opentad/bin/activate"
export XDG_CACHE_HOME="$DUCA_BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$DUCA_BASE/tmp/xdg_config"
export HF_HOME="$DUCA_BASE/hf_cache"
export PYTHONPATH="$DUCA_RECOVERY_SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$DUCA_RECOVERY_SNAPSHOT"

COMMON_ARGS=(
  --output-dir "$DUCA_RECOVERY_OUTPUT"
  --device cuda:0
  --num-workers 8
  --annotation "$DUCA_BASE/thumos14/annotations/thumos_14_anno.json"
  --class-map "$DUCA_BASE/thumos14/annotations/category_idx.txt"
  --train-data "$DUCA_BASE/thumos14/raw_data/video"
  --pretrain "$DUCA_BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
)

python tools/bata/run_duca_marginal_frozen_h65_probe.py \
  --stage pre-run \
  "${COMMON_ARGS[@]}"

python tools/bata/run_duca_marginal_frozen_h65_probe.py \
  --stage summarize \
  "${COMMON_ARGS[@]}"
