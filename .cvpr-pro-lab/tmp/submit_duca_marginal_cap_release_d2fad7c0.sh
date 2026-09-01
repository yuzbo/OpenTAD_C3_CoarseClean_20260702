#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-cap-release
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/run01/sczc063/yuzibo/duca_marginal_cap_release_d2fad7c0_20260831
#SBATCH --output=/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/cap-release-slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/cap-release-slurm-%j.err

source /etc/profile
set -eo pipefail

DUCA_BASE=/data/run01/sczc063/yuzibo
DUCA_SNAPSHOT="$DUCA_BASE/duca_marginal_cap_release_d2fad7c0_20260831"
DUCA_OUTPUT="$DUCA_BASE/duca_marginal_summary_f67d96fd_20260831"

module load miniforge3/24.11
source "$DUCA_BASE/conda_envs/opentad/bin/activate"
export XDG_CACHE_HOME="$DUCA_BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$DUCA_BASE/tmp/xdg_config"
export HF_HOME="$DUCA_BASE/hf_cache"
export PYTHONPATH="$DUCA_SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$DUCA_SNAPSHOT"

python tools/bata/run_duca_marginal_frozen_h65_probe.py \
  --stage oracle-cap-release \
  --output-dir "$DUCA_OUTPUT" \
  --device cpu \
  --num-workers 0 \
  --evaluator-threads 8 \
  --annotation "$DUCA_BASE/thumos14/annotations/thumos_14_anno.json" \
  --class-map "$DUCA_BASE/thumos14/annotations/category_idx.txt" \
  --train-data "$DUCA_BASE/thumos14/raw_data/video" \
  --pretrain "$DUCA_BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
