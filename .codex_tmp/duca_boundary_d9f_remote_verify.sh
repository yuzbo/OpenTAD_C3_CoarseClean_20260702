#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_d9fb398_20260722"
COMMIT=d9fb398578716d278e818745677a92976bcedf2c
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
  tools/bata/duca_exact_physical_solver.py \
  tools/bata/select_duca_boundary_burst_candidates.py \
  tools/bata/aggregate_duca_boundary_burst_results.py \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  tools/bata/run_duca_protected_e2e_exact_full_model_gate.py

python -m pytest \
  tests/test_duca_r0_boundary_burst_oracle.py \
  tests/test_duca_r0_holdout_replay.py \
  tests/test_duca_boundary_burst_selection.py \
  tests/test_duca_boundary_burst_artifact_contract.py \
  tests/test_duca_boundary_burst_runtime_binding.py \
  tests/test_duca_boundary_burst_full_model_gate.py \
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

OLD_RUN="$BASE/projects/c3_lowres_action_probe/duca_boundary_f90595d_r0_formal_20260722_0753"
test -f "$OLD_RUN/holdout_inputs.jsonl"
test -f "$OLD_RUN/runtime_bindings.json"
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="$(python - "$OLD_RUN/runtime_bindings.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["split"]["holdout_block_list"])
PY
)"
test -f "$DUCA_FRONTEND_HOLDOUT_BLOCK_LIST"

REPLAY_ROOT="$BASE/tmp/duca_r0_determinism_d9fb398_20260722"
test ! -e "$REPLAY_ROOT"
mkdir -p "$REPLAY_ROOT/run1" "$REPLAY_ROOT/run2"
for run in run1 run2; do
  python -m tools.bata.build_duca_r0_boundary_burst_oracles \
    --input-jsonl "$OLD_RUN/holdout_inputs.jsonl" \
    --config configs/adatad/thumos/duca_boundary_burst_r0_holdout_export.py \
    --output-jsonl "$REPLAY_ROOT/$run/holdout_families.jsonl" \
    --summary-json "$REPLAY_ROOT/$run/holdout_families.summary.json" \
    --max-unselected-hole 2
done

cmp "$REPLAY_ROOT/run1/holdout_families.jsonl" \
  "$REPLAY_ROOT/run2/holdout_families.jsonl"
sha256sum \
  "$REPLAY_ROOT/run1/holdout_families.jsonl" \
  "$REPLAY_ROOT/run1/holdout_families.summary.json" \
  "$REPLAY_ROOT/run2/holdout_families.jsonl" \
  "$REPLAY_ROOT/run2/holdout_families.summary.json"

printf 'REMOTE_D9F_VERIFY_PASS commit=%s replay_root=%s\n' "$COMMIT" "$REPLAY_ROOT"
