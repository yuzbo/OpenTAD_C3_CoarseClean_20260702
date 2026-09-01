#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_4a07a2a_20260722"
COMMIT=4a07a2af72e68f1330467161cbcac2ffba53d367
BRANCH=codex/duca-boundary-burst-20260722
REPO=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
if [[ ! -e "$SNAP" ]]; then
  export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
  export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
  git clone --filter=blob:none --single-branch --branch "$BRANCH" "https://ghfast.top/$REPO" "$SNAP"
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
  opentad/datasets/base/sliding_dataset.py \
  opentad/models/duca/acquisition.py \
  opentad/models/duca/transition_only.py \
  opentad/models/selectors/duca_allocation_artifact_replay.py \
  opentad/models/selectors/duca_online_frame_selector.py \
  tools/bata/aggregate_duca_boundary_burst_results.py \
  tools/bata/select_duca_boundary_burst_candidates.py
python -m pytest \
  tests/test_duca_transition_only.py \
  tests/test_duca_boundary_burst_selection.py \
  tests/test_duca_boundary_burst_configs.py \
  tests/test_duca_selection_quality_analysis.py \
  tests/test_duca_frontend_p0_contract.py \
  tests/test_duca_r0_holdout_replay.py \
  tests/test_duca_temporal_sampling_contract.py -q
python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q
for script in \
  scripts/run_duca_boundary_burst_p0_gpu1.sh \
  scripts/run_duca_boundary_burst_gate_gpu1.sh \
  scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh \
  scripts/submit_duca_boundary_burst_official60_suite.sh \
  scripts/run_duca_frontend_pretrain_variant_gpu1.sh \
  scripts/run_duca_two_stage_curriculum_variant_gpu1.sh; do
  bash -n "$script"
done
printf 'REMOTE_STATIC_PASS commit=%s\n' "$COMMIT"
