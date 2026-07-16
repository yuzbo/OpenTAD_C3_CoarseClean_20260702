#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_PROFILE_RECOVERY_MATRIX][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
PROFILE_RECOVERY="${SPATIAL_ZOOM_S1_PROFILE_RECOVERY:?set SPATIAL_ZOOM_S1_PROFILE_RECOVERY}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal recovery matrix requires one Slurm allocation"
[[ -f "${PROFILE_RECOVERY}" ]] || fail "profile recovery certificate does not exist"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

EXPECTED_ORDER="$(
  cd "${ROOT}"
  python -c 'from tools.bata.spatial_zoom_s1_contract import build_s1_profile_order; print(" ".join("{}:{}".format(row["resolution"], row["seed"]) for row in build_s1_profile_order()))'
)"
FROZEN_ORDER="256:3408 224:3409 256:3409 224:3407 160:3407 224:3408 160:3408 160:3409 256:3407"
[[ "${EXPECTED_ORDER}" == "${FROZEN_ORDER}" ]] || fail "profile order differs from the frozen contract"

for cell in ${FROZEN_ORDER}; do
  export SPATIAL_ZOOM_S1_RESOLUTION="${cell%%:*}"
  export SPATIAL_ZOOM_S1_SEED="${cell##*:}"
  bash "${ROOT}/scripts/run_spatial_zoom_s1_test_profile_slurm.sh"
done

CAMPAIGN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_root"])' "${PROFILE_RECOVERY}")"
DESCRIPTOR_COUNT="$(find "${CAMPAIGN_ROOT}/descriptors" -maxdepth 1 -type f -name '*.run.json' | wc -l)"
[[ "${DESCRIPTOR_COUNT}" == "9" ]] || fail "recovery campaign did not publish nine descriptors"
printf '[SPATIAL_ZOOM_S1_PROFILE_RECOVERY_MATRIX] PASS campaign_root=%s\n' "${CAMPAIGN_ROOT}"
