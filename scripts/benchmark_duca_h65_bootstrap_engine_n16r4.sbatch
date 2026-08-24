#!/usr/bin/env bash
#SBATCH --job-name=duca-h65-bs-bench
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
: "${DUCA_ANNOTATION_PATH:?DUCA_ANNOTATION_PATH is required}"
: "${DUCA_RANKPACK_PREDICTION:?DUCA_RANKPACK_PREDICTION is required}"
: "${DUCA_TRUETIME_PREDICTION:?DUCA_TRUETIME_PREDICTION is required}"
: "${DUCA_BENCHMARK_OUTPUT:?DUCA_BENCHMARK_OUTPUT is required}"

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
cd "$DUCA_REPO_ROOT"

python tools/bata/benchmark_duca_h65_bootstrap_engine.py \
  --annotation "$DUCA_ANNOTATION_PATH" \
  --prediction "rankpack_k384=$DUCA_RANKPACK_PREDICTION" \
  --prediction "truetime_k384=$DUCA_TRUETIME_PREDICTION" \
  --nonce "DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823" \
  --namespace "PAIRED_VIDEO_BOOTSTRAP_ENGINE_BENCHMARK_V1" \
  --draws "${DUCA_BENCHMARK_DRAWS:-4}" \
  --output "$DUCA_BENCHMARK_OUTPUT"
