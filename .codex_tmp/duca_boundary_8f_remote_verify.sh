#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_8f591ab_20260722"
COMMIT=8f591ab
BRANCH=codex/duca-boundary-burst-20260722
REPO=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

if [[ ! -e "$SNAP" ]]; then
  export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
  export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
  git clone --filter=blob:none --single-branch --branch "$BRANCH" \
    "https://ghfast.top/$REPO" "$SNAP"
fi

cd "$SNAP"
git checkout --detach "$COMMIT"
COMMIT=$(git rev-parse HEAD)
test "$COMMIT" = "8f591ab39a3afa5274d70f6f296548e1a2aa934a"
test -z "$(git status --porcelain --untracked-files=normal)"

module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

python -m py_compile tools/bata/finalize_duca_r0_boundary_burst.py
python -m pytest tests/test_duca_r0_evidence_contract.py -q
python -m pytest \
  tests/test_duca_boundary_burst_artifact_contract.py \
  tests/test_duca_boundary_burst_runtime_binding.py \
  tests/test_duca_boundary_burst_hard_swap_alignment.py \
  tests/test_duca_r5_paper_matrix.py -q
python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

printf 'REMOTE_8F_VERIFY_PASS commit=%s snapshot=%s\n' "$COMMIT" "$SNAP"
