#!/bin/bash
#SBATCH --job-name=zt-stop-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail
source /usr/share/modules/init/bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
bash "${ZOOMTOKEN_SOURCE_ROOT}/scripts/run_zoomtoken_stopped_matrix_eval_n16r4.sh"
