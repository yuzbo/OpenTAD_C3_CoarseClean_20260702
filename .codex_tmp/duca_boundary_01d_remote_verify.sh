#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_44c7227_20260722"
COMMIT=44c7227b575b22c666b2f309c69b1dcfdc4102c8
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
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"

module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$SNAP"

python -m py_compile \
  opentad/models/detectors/temporalmaxer.py \
  tools/bata/run_duca_temporalmaxer_one_step.py \
  tools/bata/duca_r5_paper_matrix.py \
  tools/bata/duca_boundary_burst_hard_swap_alignment.py \
  tools/bata/run_duca_protected_physical_p3_shard.py \
  tools/bata/duca_selected_axis_training.py

python -m pytest \
  tests/test_duca_boundary_burst_submission_journal.py \
  tests/test_duca_boundary_burst_artifact_contract.py \
  tests/test_duca_boundary_burst_runtime_binding.py \
  tests/test_duca_boundary_burst_hard_swap_alignment.py \
  tests/test_duca_r5_paper_matrix.py -q

python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

for script in \
  scripts/submit_duca_boundary_burst_official60_suite.sh \
  scripts/run_duca_boundary_burst_hard_swap_alignment_gpu1.sh \
  scripts/run_duca_boundary_burst_r4_gpu1.sbatch \
  scripts/launch_duca_r5_paper_matrix.sh; do
  bash -n "$script"
done

if rg -n -i 'not implemented|placeholder|sentinel|TODO|mock|deploy_ledger' \
  opentad/models/detectors/temporalmaxer.py \
  tools/bata/run_duca_temporalmaxer_one_step.py \
  tools/bata/duca_r5_paper_matrix.py \
  tools/bata/duca_boundary_burst_hard_swap_alignment.py; then
  exit 1
fi

printf 'REMOTE_44C_VERIFY_PASS commit=%s snapshot=%s\n' "$COMMIT" "$SNAP"
