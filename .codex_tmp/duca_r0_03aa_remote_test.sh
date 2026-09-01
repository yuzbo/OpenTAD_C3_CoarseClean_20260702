#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_r0_03aa4ce_20260722"
COMMIT=03aa4ce
BRANCH=codex/duca-boundary-burst-20260722
REPO=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
if [[ ! -e "$SNAP" ]]; then
  export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
  export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
  git clone --filter=blob:none --single-branch --branch "$BRANCH" "https://ghfast.top/$REPO" "$SNAP"
fi
cd "$SNAP"
git checkout --detach "$COMMIT"
test "$(git rev-parse --short HEAD)" = "$COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
python -m py_compile \
  opentad/datasets/base/sliding_dataset.py \
  tools/bata/duca_exact_physical_solver.py \
  tools/bata/build_duca_r0_boundary_burst_oracles.py
python -m pytest \
  tests/test_duca_r0_boundary_burst_oracle.py \
  tests/test_duca_r0_holdout_replay.py \
  tests/test_duca_temporal_sampling_contract.py -q
bash -n scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh
printf 'REMOTE_R0_PASS commit=%s\n' "$COMMIT"
