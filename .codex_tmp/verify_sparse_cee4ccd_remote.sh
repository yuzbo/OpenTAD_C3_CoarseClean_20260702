#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

SNAPSHOT=/data/run01/sczc063/yuzibo/projects/opentad_duca_sparse_cee4ccd_20260723
EXPECTED=cee4ccd33fb20e11978e4a2a6eaa3f5845b51489
BRANCH=codex/duca-sparse-probe-gatefix-20260723
REPO=https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

if [ ! -d "${SNAPSHOT}/.git" ]; then
  git clone --depth 1 --branch "${BRANCH}" "${REPO}" "${SNAPSHOT}"
fi
cd "${SNAPSHOT}"
test "$(git rev-parse HEAD)" = "${EXPECTED}"
test -z "$(git status --porcelain)"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
python -m py_compile tools/bata/duca_frontend_initialization.py
bash -n scripts/run_duca_independent_official60_gpu1.sh
python -m pytest \
  tests/test_duca_sparse_probe_interpolation.py \
  tests/test_duca_two_stage_curriculum.py -q
test -z "$(git status --porcelain)"
printf 'SNAPSHOT=%s\nHEAD=%s\n' "${SNAPSHOT}" "${EXPECTED}"
