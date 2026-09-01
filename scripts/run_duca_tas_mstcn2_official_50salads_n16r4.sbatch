#!/usr/bin/env bash
#SBATCH -J mstcn2_50s_official
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH -o /data/run01/sczc063/yuzibo/duca_tas_mstcn2_9d31fb3/slurm/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/duca_tas_mstcn2_9d31fb3/slurm/%x_%A_%a.err

set -euo pipefail

repo=/data/run01/sczc063/yuzibo/external_official_action_segmentation_repos/DUCA_MS-TCN2_9d31fb3
envp=/data/run01/sczc063/yuzibo/venvs/duca_mstcn2_official_9d31fb3
expected_revision=9d31fb3c23467b9ce3030d43b6d33a96869b6422
dataset=50salads
split="${SLURM_ARRAY_TASK_ID}"

test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_revision}"
test -z "$(git -C "${repo}" status --short)"
test -x "${envp}/bin/python"
test -d "${repo}/data/${dataset}/features"
test -d "${repo}/data/${dataset}/groundTruth"
test -f "${repo}/data/${dataset}/mapping.txt"
test -f "${repo}/data/${dataset}/splits/train.split${split}.bundle"
test -f "${repo}/data/${dataset}/splits/test.split${split}.bundle"

cd "${repo}"

"${envp}/bin/python" main.py \
  --action=train \
  --dataset="${dataset}" \
  --split="${split}" \
  --num_epochs=100 \
  --num_layers_PG=11 \
  --num_layers_R=10 \
  --num_R=3

"${envp}/bin/python" main.py \
  --action=predict \
  --dataset="${dataset}" \
  --split="${split}" \
  --num_epochs=100 \
  --num_layers_PG=11 \
  --num_layers_R=10 \
  --num_R=3

"${envp}/bin/python" eval.py --dataset="${dataset}" --split="${split}"
