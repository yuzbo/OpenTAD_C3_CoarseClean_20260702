#!/bin/bash
#SBATCH --job-name=zt-d2s-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/d2s_tad_full200_compute/slurm_%j.log
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/d2s_tad_full200_compute/slurm_%j.err

set -euo pipefail
mkdir -p /data/run01/sczc063/yuzibo/projects/d2s_tad_full200_compute

SOURCE_ROOT="${ZOOMTOKEN_SOURCE_ROOT:?submit with ZOOMTOKEN_SOURCE_ROOT bound to the clean candidate checkout}"
cd "${SOURCE_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export ZOOMTOKEN_SEEDS="4407,4408,4409"
bash scripts/run_zoomtoken_d2s_tad_full200_compute_n16r4.sh
