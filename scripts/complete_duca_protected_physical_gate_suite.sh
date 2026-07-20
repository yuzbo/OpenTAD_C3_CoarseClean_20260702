#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_COMPLETE_GATES][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_protected_physical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
PROTOCOL_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
PROTOCOL_SHA256="${DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256:-}"
MAIN_GATE="${DUCA_PROTECTED_MAIN_GATE_JSON:-}"
BRIDGE025_GATE="${DUCA_PROTECTED_BRIDGE025_GATE_JSON:-}"
UNI_COMPANION_GATE="${DUCA_PROTECTED_UNI_COMPANION_GATE_JSON:-}"
RHO_GATE="${DUCA_PROTECTED_RHO_GATE_JSON:-}"
SHORT_SHARD="${DUCA_PROTECTED_P3_SHORT_JSON:-}"
MEDIUM_SHARD="${DUCA_PROTECTED_P3_MEDIUM_JSON:-}"
LONG_SHARD="${DUCA_PROTECTED_P3_LONG_JSON:-}"
AGGREGATE_JSON="${DUCA_PROTECTED_P3_AGGREGATE_JSON:-}"
AUTHORIZATION_JSON="${DUCA_PROTECTED_AUTHORIZATION_JSON:-}"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
for path in \
  "${PROTOCOL_JSON}" "${MAIN_GATE}" "${BRIDGE025_GATE}" \
  "${UNI_COMPANION_GATE}" "${RHO_GATE}" \
  "${SHORT_SHARD}" "${MEDIUM_SHARD}" "${LONG_SHARD}"; do
  [[ -f "${path}" ]] || fail "required evidence is missing: ${path}"
done
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] || fail "P0 hash drift"
[[ -n "${AGGREGATE_JSON}" && ! -e "${AGGREGATE_JSON}" ]] || fail "fresh aggregate output is required"
[[ -n "${AUTHORIZATION_JSON}" && ! -e "${AUTHORIZATION_JSON}" ]] || fail "fresh authorization output is required"

MAIN_GATE_SHA256="$(sha256sum "${MAIN_GATE}" | awk '{print $1}')"
BRIDGE025_GATE_SHA256="$(sha256sum "${BRIDGE025_GATE}" | awk '{print $1}')"
UNI_COMPANION_GATE_SHA256="$(sha256sum "${UNI_COMPANION_GATE}" | awk '{print $1}')"
RHO_GATE_SHA256="$(sha256sum "${RHO_GATE}" | awk '{print $1}')"
SHORT_SHA256="$(sha256sum "${SHORT_SHARD}" | awk '{print $1}')"
MEDIUM_SHA256="$(sha256sum "${MEDIUM_SHARD}" | awk '{print $1}')"
LONG_SHA256="$(sha256sum "${LONG_SHARD}" | awk '{print $1}')"

"${PYTHON}" -m tools.bata.aggregate_duca_protected_physical_p3 \
  --short-json "${SHORT_SHARD}" \
  --short-sha256 "${SHORT_SHA256}" \
  --medium-json "${MEDIUM_SHARD}" \
  --medium-sha256 "${MEDIUM_SHA256}" \
  --long-json "${LONG_SHARD}" \
  --long-sha256 "${LONG_SHA256}" \
  --output-json "${AGGREGATE_JSON}"
AGGREGATE_SHA256="$(sha256sum "${AGGREGATE_JSON}" | awk '{print $1}')"
printf '%s  %s\n' "${AGGREGATE_SHA256}" "${AGGREGATE_JSON}" \
  > "${AGGREGATE_JSON}.sha256"

"${PYTHON}" -m tools.bata.authorize_duca_protected_physical_suite \
  --protocol-manifest "${PROTOCOL_JSON}" \
  --protocol-manifest-sha256 "${PROTOCOL_SHA256}" \
  --main-gate "${MAIN_GATE}" \
  --main-gate-sha256 "${MAIN_GATE_SHA256}" \
  --bridge025-gate "${BRIDGE025_GATE}" \
  --bridge025-gate-sha256 "${BRIDGE025_GATE_SHA256}" \
  --uni-companion-gate "${UNI_COMPANION_GATE}" \
  --uni-companion-gate-sha256 "${UNI_COMPANION_GATE_SHA256}" \
  --rho-gate "${RHO_GATE}" \
  --rho-gate-sha256 "${RHO_GATE_SHA256}" \
  --p3-aggregate "${AGGREGATE_JSON}" \
  --p3-aggregate-sha256 "${AGGREGATE_SHA256}" \
  --output-json "${AUTHORIZATION_JSON}"
sha256sum "${AUTHORIZATION_JSON}" > "${AUTHORIZATION_JSON}.sha256"
