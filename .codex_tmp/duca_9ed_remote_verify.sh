#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=9ed10139317c4196072d471ced883eb1dfc31703
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_9ed1013_20260722

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

if [[ ! -d ${SNAPSHOT}/.git ]]; then
  git clone --branch codex/duca-boundary-burst-20260722 --single-branch \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git \
    ${SNAPSHOT}
fi
git -C ${SNAPSHOT} fetch origin codex/duca-boundary-burst-20260722
git -C ${SNAPSHOT} checkout --detach ${COMMIT}
cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

python -m py_compile \
  tools/bata/duca_p0_evaluation.py \
  tools/bata/finalize_duca_r0_boundary_burst.py
bash -n scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh
python -m pytest \
  tests/test_duca_r0_evidence_contract.py \
  tests/test_duca_r0_holdout_replay.py \
  tests/test_duca_boundary_burst_configs.py -q
python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

echo "VERIFY_OK commit=${COMMIT} snapshot=${SNAPSHOT}"
