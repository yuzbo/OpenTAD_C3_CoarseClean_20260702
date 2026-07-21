#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_SERIAL][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${SPLIT_MANIFEST}" ]] || fail "frontend split manifest is missing"
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == "${SPLIT_SHA256}" ]] \
  || fail "frontend split manifest hash drift"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

export DUCA_FRONTEND_TRAIN_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_train_block_list.txt"
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_holdout_block_list.txt"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --checkpoint "${ADATAD_PRETRAIN_PATH}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split-sha256 "${SPLIT_SHA256}" \
  --output-json "${RUN_ROOT}/p0_real_gate.json"
frontend_variants=(
  lr_control_c25_a50_s100
  lr_coarse50_action100_scorer25
  lr_coarse100_action200_scorer50
)
for variant in "${frontend_variants[@]}"; do
  export DUCA_FRONTEND_VARIANT="${variant}"
  export RUN_DIR="${RUN_ROOT}/p0/${variant}/run"
  export WORK_DIR="${RUN_ROOT}/p0/${variant}/work"
  bash scripts/run_duca_frontend_pretrain_variant_gpu1.sh
done

DECISION="${RUN_ROOT}/frontend_decision.json"
CANDIDATE_MANIFEST="${RUN_ROOT}/frontend_candidate_manifest.json"
"${PYTHON}" -m tools.bata.aggregate_duca_frontend_candidates \
  --expected-commit "${EXPECTED_COMMIT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${SPLIT_SHA256}" \
  --receipt "${RUN_ROOT}/p0/lr_control_c25_a50_s100/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/lr_coarse50_action100_scorer25/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/lr_coarse100_action200_scorer50/run/completion.json" \
  --candidate-manifest "${CANDIDATE_MANIFEST}" \
  --decision-json "${DECISION}"

if [[ "${DUCA_FRONTEND_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_TWO_STAGE_SERIAL] frontend completed ${DECISION}"
  exit 0
fi

DECISION_SHA256="$(sha256sum "${DECISION}" | awk '{print $1}')"

export DUCA_FRONTEND_DECISION_JSON="${DECISION}"
export DUCA_FRONTEND_DECISION_SHA256="${DECISION_SHA256}"
export DUCA_TWO_STAGE_GATE_ROOT="${RUN_ROOT}/two_stage_gate"
bash scripts/run_duca_two_stage_curriculum_gate_gpu1.sh
GATE_SUITE="${DUCA_TWO_STAGE_GATE_ROOT}/gate_suite.json"
GATE_SUITE_SHA256="$(sha256sum "${GATE_SUITE}" | awk '{print $1}')"

export DUCA_SELECTED_OPT_GATE_SUITE="${GATE_SUITE}"
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="${GATE_SUITE_SHA256}"
official_variants=(
  two_stage_exact_uniform
  global_curriculum_g0
  global_curriculum_g1
  global_curriculum_g2
)
for variant in "${official_variants[@]}"; do
  export DUCA_SELECTED_OPT_VARIANT="${variant}"
  export RUN_DIR="${RUN_ROOT}/official60/${variant}/run"
  export WORK_DIR="${RUN_ROOT}/official60/${variant}/work"
  bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
done

"${PYTHON}" -m tools.bata.aggregate_duca_two_stage_results \
  --expected-commit "${EXPECTED_COMMIT}" \
  --frontend-decision "${DECISION}" \
  --frontend-decision-sha256 "${DECISION_SHA256}" \
  --gate-suite "${GATE_SUITE}" \
  --gate-suite-sha256 "${GATE_SUITE_SHA256}" \
  --completion "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.json" \
  --completion "${RUN_ROOT}/official60/global_curriculum_g0/run/completion.json" \
  --completion "${RUN_ROOT}/official60/global_curriculum_g1/run/completion.json" \
  --completion "${RUN_ROOT}/official60/global_curriculum_g2/run/completion.json" \
  --output-json "${RUN_ROOT}/final_suite_results.json"

echo "[DUCA_TWO_STAGE_SERIAL] completed ${RUN_ROOT}/final_suite_results.json"
