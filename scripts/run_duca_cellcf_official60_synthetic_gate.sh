#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_OFFICIAL60_SYNTHETIC][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
OUTPUT_JSON="${DUCA_CELLCF_SYNTHETIC_GATE_JSON:-}"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "DUCA_EXPECTED_COMMIT is invalid"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "gate requires a clean tree"
[[ -n "${OUTPUT_JSON}" ]] || fail "DUCA_CELLCF_SYNTHETIC_GATE_JSON is required"
OUTPUT_JSON="$(
  duca_cellcf_require_external_path \
    "OUTPUT_JSON" "${REPO_ROOT}" "${BASE}" "${OUTPUT_JSON}"
)" || fail "OUTPUT_JSON violates the formal path contract"
[[ ! -e "${OUTPUT_JSON}" ]] || fail "refusing to overwrite existing synthetic evidence"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

"${PYTHON}" -m tools.bata.run_duca_cellcf_synthetic_gate \
  --device cpu \
  --output-json "${OUTPUT_JSON}"
