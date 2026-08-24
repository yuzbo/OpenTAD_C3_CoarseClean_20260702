#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-bs-fasttest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
: "${DUCA_EXPECTED_COMMIT:?DUCA_EXPECTED_COMMIT is required}"

cd "$DUCA_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$DUCA_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
python -m py_compile \
  tools/bata/bootstrap_duca_h65_official_map.py \
  tools/bata/benchmark_duca_h65_bootstrap_engine.py \
  tools/bata/analyze_duca_h65_singleclock_strata.py
python -m pytest \
  tests/test_duca_h65_bootstrap_contract.py \
  tests/test_duca_h65_singleclock_strata.py \
  tests/test_duca_h65_singleclock_finalizer.py \
  tests/test_duca_h65_cycle4_singleclock_contract.py \
  -q
