#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_DENSE_FULL_STACK_COST][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

DENSE_CONFIG="${DUCA_DENSE_ADATAD_CONFIG:-}"
DENSE_CHECKPOINT="${DUCA_DENSE_ADATAD_CHECKPOINT:-}"
DENSE_BINDING="${DUCA_DENSE_ADATAD_BINDING_JSON:-}"
DENSE_BINDING_SHA256="${DUCA_DENSE_ADATAD_BINDING_SHA256:-}"
DENSE_TRAINED_COMMIT="${DUCA_DENSE_ADATAD_TRAINED_COMMIT:-}"
CELLCF_CHECKPOINT="${DUCA_CELLCF_CHECKPOINT:-}"
CELLCF_POST_RUN="${DUCA_CELLCF_POST_RUN_EVIDENCE_JSON:-}"
CELLCF_POST_RUN_SHA256="${DUCA_CELLCF_POST_RUN_EVIDENCE_SHA256:-}"
CELLCF_TRAINED_COMMIT="${DUCA_CELLCF_TRAINED_COMMIT:-}"
CELLCF_CONFIG="${DUCA_CELLCF_TRAINED_CONFIG:-}"
OUTPUT_ROOT="${DUCA_DENSE_FULL_STACK_COST_ROOT:-}"
SAMPLES="${DUCA_DENSE_FULL_STACK_COST_SAMPLES:-500}"
WARMUP="${DUCA_DENSE_FULL_STACK_COST_WARMUP:-20}"
REPEATS="${DUCA_DENSE_FULL_STACK_COST_REPEATS:-3}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "cost profiling must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ "${SAMPLES}" =~ ^[1-9][0-9]*$ ]] || fail "sample count must be a positive integer"
[[ "${WARMUP}" =~ ^[1-9][0-9]*$ ]] || fail "warmup count must be a positive integer"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || fail "repeat count must be a positive integer"
[[ "${SAMPLES}" -ge 500 ]] || fail "at least 500 measured samples are required"
[[ "${WARMUP}" -ge 20 ]] || fail "at least 20 warmup samples are required"
[[ "${REPEATS}" -ge 3 ]] || fail "at least three fresh repeats are required"
for path in "${DENSE_CONFIG}" "${DENSE_CHECKPOINT}" "${DENSE_BINDING}" \
  "${CELLCF_CONFIG}" "${CELLCF_CHECKPOINT}" "${CELLCF_POST_RUN}"; do
  [[ -f "${path}" ]] || fail "required evidence is missing: ${path}"
done
[[ "${DENSE_BINDING_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "dense binding SHA256 is invalid"
[[ "${CELLCF_POST_RUN_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "CellCF post-run SHA256 is invalid"
[[ "${DENSE_TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "dense trained commit is invalid"
[[ "${CELLCF_TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "CellCF trained commit is invalid"
[[ -n "${OUTPUT_ROOT}" ]] || fail "DUCA_DENSE_FULL_STACK_COST_ROOT is required"
OUTPUT_ROOT="$(
  duca_cellcf_require_external_path \
    "OUTPUT_ROOT" "${REPO_ROOT}" "${BASE}" "${OUTPUT_ROOT}"
)" || fail "OUTPUT_ROOT violates the formal path contract"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite an existing cost root"
mkdir -p "${OUTPUT_ROOT}"
PROFILE_SESSION_ID="slurm-${SLURM_JOB_ID}"

dense_args=()
cellcf_args=()
profile_dense() {
  local repeat="$1"
  local order_position="$2"
  local prefix="${OUTPUT_ROOT}/dense_repeat${repeat}"
  "${PYTHON}" -m tools.bata.profile_duca_full_stack_cost \
    "${DENSE_CONFIG}" --checkpoint "${DENSE_CHECKPOINT}" --use-ema --amp \
    --backbone-pretrain "${ADATAD_PRETRAIN_PATH}" \
    --method-name dense-adatad --output-prefix "${prefix}" \
    --trained-commit "${DENSE_TRAINED_COMMIT}" \
    --profile-session-id "${PROFILE_SESSION_ID}" \
    --profile-pair-id "repeat-${repeat}" \
    --profile-repeat-index "${repeat}" \
    --profile-order-position "${order_position}" \
    --samples "${SAMPLES}" --warmup-samples "${WARMUP}" --loader-workers 0 \
    --checkpoint-evidence "${DENSE_BINDING}" \
    --checkpoint-evidence-sha256 "${DENSE_BINDING_SHA256}" \
    --sample-power
  dense_args+=(--dense "${prefix}.summary.json")
}

profile_cellcf() {
  local repeat="$1"
  local order_position="$2"
  local prefix="${OUTPUT_ROOT}/cellcf_repeat${repeat}"
  "${PYTHON}" -m tools.bata.profile_duca_full_stack_cost \
    "${CELLCF_CONFIG}" --checkpoint "${CELLCF_CHECKPOINT}" --use-ema --amp \
    --backbone-pretrain "${ADATAD_PRETRAIN_PATH}" \
    --method-name cellcf-fixed384 --output-prefix "${prefix}" \
    --trained-commit "${CELLCF_TRAINED_COMMIT}" \
    --profile-session-id "${PROFILE_SESSION_ID}" \
    --profile-pair-id "repeat-${repeat}" \
    --profile-repeat-index "${repeat}" \
    --profile-order-position "${order_position}" \
    --samples "${SAMPLES}" --warmup-samples "${WARMUP}" --loader-workers 0 \
    --post-run-evidence "${CELLCF_POST_RUN}" \
    --post-run-evidence-sha256 "${CELLCF_POST_RUN_SHA256}" \
    --sample-power
  cellcf_args+=(--cellcf "${prefix}.summary.json")
}

for repeat in $(seq 1 "${REPEATS}"); do
  if ((repeat % 2 == 1)); then
    profile_dense "${repeat}" 1
    profile_cellcf "${repeat}" 2
  else
    profile_cellcf "${repeat}" 1
    profile_dense "${repeat}" 2
  fi
done

"${PYTHON}" -m tools.bata.summarize_duca_dense_full_stack_cost \
  "${dense_args[@]}" "${cellcf_args[@]}" \
  --output-json "${OUTPUT_ROOT}/dense_vs_cellcf.json" \
  --output-tsv "${OUTPUT_ROOT}/dense_vs_cellcf.tsv"
