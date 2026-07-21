#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_BURST_P0][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${SPLIT_MANIFEST}" ]] || fail "split manifest is missing"
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == "${SPLIT_SHA256}" ]] || fail "split drift"

export DUCA_FRONTEND_TRAIN_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_train_block_list.txt"
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_holdout_block_list.txt"
variant_configs=(
  configs/adatad/thumos/duca_gaussian_frontend_pretrain_matched_fixed384.py
  configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py
  configs/adatad/thumos/duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py
)
gate_args=()
for config in "${variant_configs[@]}"; do gate_args+=(--variant-config "${config}"); done
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  --config configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py \
  "${gate_args[@]}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --checkpoint "${ADATAD_PRETRAIN_PATH}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split-sha256 "${SPLIT_SHA256}" \
  --output-json "${RUN_ROOT}/p0_real_gate.json"

variants=(gaussian_matched burst_r2q3 burst_r4q5)
for variant in "${variants[@]}"; do
  export DUCA_FRONTEND_VARIANT="${variant}"
  export RUN_DIR="${RUN_ROOT}/p0/${variant}/run"
  export WORK_DIR="${RUN_ROOT}/p0/${variant}/work"
  bash scripts/run_duca_frontend_pretrain_variant_gpu1.sh
done

"${PYTHON}" -m tools.bata.select_duca_boundary_burst_candidates \
  --expected-commit "${EXPECTED_COMMIT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${SPLIT_SHA256}" \
  --receipt "${RUN_ROOT}/p0/gaussian_matched/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/burst_r2q3/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/burst_r4q5/run/completion.json" \
  --output-json "${RUN_ROOT}/frontend_decision.json"

echo "[DUCA_BURST_P0] completed ${RUN_ROOT}/frontend_decision.json"
