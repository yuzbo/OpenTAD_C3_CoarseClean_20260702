#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_X3D_OFFICIAL_BACKEND][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CONFIG="${CONFIG:-configs/adatad/thumos/duca_online_x3d_official_adatad_backend_full_train.py}"
export VALIDATOR="${VALIDATOR:-tools/bata/validate_duca_x3d_official_adatad_backend.py}"
export RUN_TAG="${RUN_TAG:-duca_x3d_official_adatad_backend_$(date +%Y%m%d_%H%M%S_%z)}"
export DUCA_X3D_REQUIRE_JSONL_EXISTS=1

if [[ -z "${DUCA_X3D_ACTIONNESS_JSONL:-}" ]]; then
  fail "DUCA_X3D_ACTIONNESS_JSONL must point to exported train-free X3D actionness JSONL"
fi
if [[ ! -f "${DUCA_X3D_ACTIONNESS_JSONL}" ]]; then
  fail "DUCA_X3D_ACTIONNESS_JSONL file missing: ${DUCA_X3D_ACTIONNESS_JSONL}"
fi

bash scripts/run_duca_online_official_adatad_backend_gpu1.sh
