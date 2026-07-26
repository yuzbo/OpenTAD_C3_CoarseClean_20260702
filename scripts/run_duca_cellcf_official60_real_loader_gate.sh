#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_OFFICIAL60_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
SYNTHETIC_GATE="${DUCA_CELLCF_SYNTHETIC_GATE_JSON:-}"
OUTPUT_JSON="${DUCA_CELLCF_GATE_JSON:-}"
PRETRAIN_SHA256="${DUCA_CELLCF_VIDEOMAE_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "real-loader CUDA gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "DUCA_EXPECTED_COMMIT is invalid"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "gate requires a clean tree"
[[ -f "${SYNTHETIC_GATE}" ]] || fail "DUCA_CELLCF_SYNTHETIC_GATE_JSON is missing"
[[ -n "${OUTPUT_JSON}" ]] || fail "DUCA_CELLCF_GATE_JSON is required"
OUTPUT_JSON="$(
  duca_cellcf_require_external_path \
    "OUTPUT_JSON" "${REPO_ROOT}" "${BASE}" "${OUTPUT_JSON}"
)" || fail "OUTPUT_JSON violates the formal path contract"
[[ ! -e "${OUTPUT_JSON}" ]] || fail "refusing to overwrite real-loader evidence"
mkdir -p "$(dirname "${OUTPUT_JSON}")"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "VideoMAE pretrain is missing"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] || fail "official ASFormer source is missing"
if [[ -z "${PRETRAIN_SHA256}" ]]; then
  PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"
fi
[[ "${PRETRAIN_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid VideoMAE SHA256"

"${PYTHON}" -m tools.bata.run_duca_cellcf_real_loader_cuda_gate \
  --expected-commit "${EXPECTED_COMMIT}" \
  --synthetic-gate-json "${SYNTHETIC_GATE}" \
  --videomae-checkpoint "${ADATAD_PRETRAIN_PATH}" \
  --expected-videomae-sha256 "${PRETRAIN_SHA256}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --device cuda:0 \
  --output-json "${OUTPUT_JSON}"
