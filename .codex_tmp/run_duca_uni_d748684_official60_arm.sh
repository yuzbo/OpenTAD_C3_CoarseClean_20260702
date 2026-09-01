#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
REPO_ROOT=/data/run01/sczc063/yuzibo/projects/opentad_duca_uni_d748684_20260721
P0_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_p0_20260721_0320
GATE_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_gate_20260721_0325
RUN_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_official60_20260721_0330
EXPECTED_COMMIT=d748684bc6a3da5b5cbbb0b78a64b71ef1cdd1dc
PROTOCOL_JSON="${P0_ROOT}/protocol_manifest.json"
AUTHORIZATION_JSON="${GATE_ROOT}/authorization.json"
VARIANT="${DUCA_VARIANT:?DUCA_VARIANT is required}"

case "${VARIANT}" in
  exact_uniform|protected_e2e|protected_e2e_bridge025|protected_e2e_uni_companion)
    ;;
  *)
    echo "[DUCA_UNI_OFFICIAL60][FAIL] unsupported variant ${VARIANT}" >&2
    exit 2
    ;;
esac

module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f "${PROTOCOL_JSON}" && -f "${AUTHORIZATION_JSON}" ]]

export BASE
export DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON="${PROTOCOL_JSON}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256
DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256="$(
  sha256sum "${PROTOCOL_JSON}" | awk '{print $1}'
)"
export DUCA_PROTECTED_AUTHORIZATION_JSON="${AUTHORIZATION_JSON}"
export DUCA_PROTECTED_AUTHORIZATION_SHA256
DUCA_PROTECTED_AUTHORIZATION_SHA256="$(
  sha256sum "${AUTHORIZATION_JSON}" | awk '{print $1}'
)"
export DUCA_PROTECTED_VARIANT="${VARIANT}"
export RUN_DIR="${RUN_ROOT}/arms/${VARIANT}/run"
export WORK_DIR="${RUN_ROOT}/arms/${VARIANT}/work"

bash scripts/run_duca_protected_physical_official60_variant_gpu1.sh
