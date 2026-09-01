#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_P0][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_protected_physical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
OUTPUT_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -x "${PYTHON}" ]] || fail "Python environment is missing"
[[ -f "${DUCA_PROTECTED_ADATAD_PRETRAIN}" ]] || fail "VideoMAE-S pretrain is missing"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] || fail "official ASFormer is missing"
[[ -n "${OUTPUT_JSON}" && ! -e "${OUTPUT_JSON}" ]] || fail "fresh external output is required"

PRETRAIN_SHA256="$(sha256sum "${DUCA_PROTECTED_ADATAD_PRETRAIN}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.freeze_duca_protected_physical_protocol \
  --expected-commit "${EXPECTED_COMMIT}" \
  --adatad-pretrain "${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
  --adatad-pretrain-sha256 "${PRETRAIN_SHA256}" \
  --output-json "${OUTPUT_JSON}"
sha256sum "${OUTPUT_JSON}" > "${OUTPUT_JSON}.sha256"
