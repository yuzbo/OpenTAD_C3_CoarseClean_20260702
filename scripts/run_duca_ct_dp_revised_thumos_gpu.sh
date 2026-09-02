#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_CT_DP_REVISED][FAIL] $*" >&2; exit 1; }

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
REPO_ROOT="${REPO_ROOT:-${BASE}/projects/opentad_duca_ct_dp_revised_20260902}"
CONFIG="${1:?config path required}"
SEED="${SEED:-3407}"
EXP_ID="${EXP_ID:-0}"

[[ -d "${REPO_ROOT}" ]] || fail "repo not found: ${REPO_ROOT}"
cd "${REPO_ROOT}"
EXPECTED_COMMIT="${CTDP_EXPECTED_COMMIT:?CTDP_EXPECTED_COMMIT must be the full 40-character target SHA}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "CTDP_EXPECTED_COMMIT must be a full SHA"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "checkout HEAD mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
[[ -z "$(git status --porcelain)" ]] || fail "checkout is not clean"
[[ -f "${CONFIG}" ]] || fail "config not found: ${CONFIG}"

if [[ "${CTDP_STAGE:-}" == "mechanism" ]]; then
  GEOMETRY_RECEIPT="${CTDP_GEOMETRY_RECEIPT:?mechanism jobs require CTDP_GEOMETRY_RECEIPT}"
  [[ -f "${GEOMETRY_RECEIPT}" ]] || fail "geometry gate receipt not found: ${GEOMETRY_RECEIPT}"
  python - "${GEOMETRY_RECEIPT}" "${EXPECTED_COMMIT}" <<'PY'
import json, sys
path, expected = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    receipt = json.load(handle)
observed = receipt.get("commit") or receipt.get("commit_sha") or receipt.get("git_commit")
if observed != expected:
    raise SystemExit(f"geometry gate commit mismatch: expected {expected}, got {observed}")
status = receipt.get("status") or receipt.get("gate_status")
if status not in ("PASS", "passed", "complete"):
    raise SystemExit(f"geometry gate is not passing: {status!r}")
arm = str(receipt.get("arm") or receipt.get("geometry_arm") or "").lower()
if arm and arm not in ("g2", "geometry_g2", "m00"):
    raise SystemExit(f"mechanism stage requires the G2/M00 geometry receipt, got {arm!r}")
PY
fi

export HOME="${BASE}/tmp/home"
export XDG_CACHE_HOME="${BASE}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${BASE}/tmp/xdg_config"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${REPO_ROOT}/logs" "${BASE}/slurm_logs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

module load cuda/11.8
module load miniforge3/24.11
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python executable not found: ${PYTHON}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" -m py_compile \
    opentad/models/bricks/scale_adaptive_conv1d.py \
    opentad/models/selectors/dual_phase_frame_selector.py \
    opentad/models/detectors/actionformer.py
  echo "[DUCA_CT_DP_REVISED] precheck passed"
  exit 0
fi

echo "[DUCA_CT_DP_REVISED] repo=${REPO_ROOT} commit=$(git rev-parse --short HEAD) config=${CONFIG} seed=${SEED}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 2000))}"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" tools/train.py "${CONFIG}" --seed "${SEED}" --id "${EXP_ID}"
