#!/usr/bin/env bash
#SBATCH --job-name=duca_env_probe
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
python -c 'import sys; print(sys.executable); print("DUCA_ENV_PROBE_PASS")'
