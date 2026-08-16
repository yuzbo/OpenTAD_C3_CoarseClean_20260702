#!/usr/bin/env bash
#SBATCH --job-name=sparsehead-route-t
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

source /etc/profile
set -euo pipefail
module load cuda/11.8 miniforge3/24.11

: "${SPARSEHEAD_SOURCE:?set SPARSEHEAD_SOURCE}"
: "${SPARSEHEAD_EXPECTED_COMMIT:?set SPARSEHEAD_EXPECTED_COMMIT}"
: "${SPARSEHEAD_RUN_ROOT:?set SPARSEHEAD_RUN_ROOT}"
: "${SPARSEHEAD_ROUTE_T_SEED:?set SPARSEHEAD_ROUTE_T_SEED}"

arms=(dense risk generic shuffled_risk similarity random uniform bypass)
task_id="${SLURM_ARRAY_TASK_ID:?array task id is required}"
if (( task_id < 0 || task_id >= ${#arms[@]} )); then
  echo "invalid array task id: ${task_id}" >&2
  exit 2
fi
arm="${arms[$task_id]}"

cd "$SPARSEHEAD_SOURCE"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$SPARSEHEAD_EXPECTED_COMMIT" ]]; then
  echo "source revision mismatch: $actual_commit != $SPARSEHEAD_EXPECTED_COMMIT" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "source checkout is dirty" >&2
  exit 4
fi

export SPARSEHEAD_ROUTE_T_ARM="$arm"
export PYTHONPATH="$SPARSEHEAD_SOURCE:${PYTHONPATH:-}"
python_bin="/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python"
config="configs/adatad/thumos/route_t_full_official.py"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/train.py "$config" --seed "$SPARSEHEAD_ROUTE_T_SEED"

