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

module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f "${PROTOCOL_JSON}" && -f "${AUTHORIZATION_JSON}" ]]

PROTOCOL_SHA256="$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')"
AUTHORIZATION_SHA256="$(sha256sum "${AUTHORIZATION_JSON}" | awk '{print $1}')"

python -m tools.bata.aggregate_duca_protected_physical_official60 \
  --expected-commit "${EXPECTED_COMMIT}" \
  --protocol-manifest-sha256 "${PROTOCOL_SHA256}" \
  --authorization-sha256 "${AUTHORIZATION_SHA256}" \
  --evidence "${RUN_ROOT}/arms/exact_uniform/run/post_run_evidence.json" \
  --evidence "${RUN_ROOT}/arms/protected_e2e/run/post_run_evidence.json" \
  --evidence "${RUN_ROOT}/arms/protected_e2e_bridge025/run/post_run_evidence.json" \
  --evidence "${RUN_ROOT}/arms/protected_e2e_uni_companion/run/post_run_evidence.json" \
  --output-json "${RUN_ROOT}/official60_suite.json"
