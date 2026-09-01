#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-whole-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831
#SBATCH --output=/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/evaluate-slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/evaluate-slurm-%j.err

source /etc/profile
set -euo pipefail

DUCA_BASE=/data/run01/sczc063/yuzibo
DUCA_SNAPSHOT="$DUCA_BASE/duca_whole_video_33e4ed13_20260831"
DUCA_INPUT="$DUCA_BASE/duca_marginal_summary_f67d96fd_20260831"
DUCA_OUTPUT="$DUCA_BASE/duca_whole_video_result_33e4ed13_20260831"
DUCA_HEAD=33e4ed137c33eef07f0452b44506a6993bdf7535

module load miniforge3/24.11
source "$DUCA_BASE/conda_envs/opentad/bin/activate"
export HOME="$DUCA_BASE/tmp/home"
export XDG_CACHE_HOME="$DUCA_BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$DUCA_BASE/tmp/xdg_config"
export HF_HOME="$DUCA_BASE/hf_cache"
export PYTHONPATH="$DUCA_SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=

cd "$DUCA_SNAPSHOT"
test "$(git rev-parse HEAD)" = "$DUCA_HEAD"
test -z "$(git status --porcelain)"

python tools/bata/run_duca_whole_video_consistent_budget_falsifier.py \
  --stage evaluate \
  --input-dir "$DUCA_INPUT" \
  --output-dir "$DUCA_OUTPUT" \
  --expected-head "$DUCA_HEAD" \
  --checkpoint "$DUCA_BASE/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth" \
  --expected-checkpoint-sha256 dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c \
  --annotation "$DUCA_BASE/thumos14/annotations/thumos_14_anno.json" \
  --class-map "$DUCA_BASE/thumos14/annotations/category_idx.txt" \
  --train-data "$DUCA_BASE/thumos14/raw_data/video" \
  --pretrain "$DUCA_BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth" \
  --evaluator-threads 8
