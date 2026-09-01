#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[TRUETIME_JOINT_SELECTOR][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN="${ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke_exec.py}"
VALIDATOR="${VALIDATOR:-tools/bata/validate_truetime_joint_selector_precheck.py}"
SMOKE_TOOL="${SMOKE_TOOL:-tools/bata/run_truetime_joint_selector_smoke.py}"
RUN_TAG="${RUN_TAG:-c3_truetime_joint_selector_adatad_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_LOW="${MASTER_PORT_LOW:-30000}"
MASTER_PORT_HIGH="${MASTER_PORT_HIGH:-60999}"
MASTER_PORT_MAX_ATTEMPTS="${MASTER_PORT_MAX_ATTEMPTS:-256}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
OUTPUT_ROOT="${C3_TRUETIME_OUTPUT_ROOT:-${BASE}/projects/c3_lowres_action_probe/truetime_joint_selector}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  fail "true-time joint selector route must use GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

pick_master_port() {
  if [[ -n "${MASTER_PORT}" ]]; then
    echo "${MASTER_PORT}"
    return 0
  fi
  "${PYTHON}" - "${RUN_TAG}" "truetime_joint_selector" "${MASTER_PORT_LOW}" "${MASTER_PORT_HIGH}" "${MASTER_PORT_MAX_ATTEMPTS}" <<'PY'
import hashlib
import os
import socket
import sys

run_tag, variant = sys.argv[1], sys.argv[2]
low, high, max_attempts = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
if not (1024 <= low <= high <= 65535):
    raise SystemExit(f"invalid MASTER_PORT range: {low}-{high}")
span = high - low + 1
seed = "|".join(
    [
        run_tag,
        variant,
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("SLURM_STEP_ID", ""),
        str(os.getpid()),
    ]
)
start = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % span
for offset in range(min(max_attempts, span)):
    port = low + ((start + offset) % span)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit(f"no free MASTER_PORT found in {low}-{high} after {max_attempts} attempts")
PY
}

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${VALIDATOR}"
require_file "${SMOKE_TOOL}"

PROOF_DIR="${PROOF_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
mkdir -p "${PROOF_DIR}"
export TRUETIME_SELECTOR_GRAD_PROOF_JSON="${TRUETIME_SELECTOR_GRAD_PROOF_JSON:-${PROOF_DIR}/selector_grad_geometry_proof.json}"

echo "[TRUETIME_JOINT_SELECTOR] repo=${REPO_ROOT}"
echo "[TRUETIME_JOINT_SELECTOR] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[TRUETIME_JOINT_SELECTOR] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[TRUETIME_JOINT_SELECTOR] precheck_only=${PRECHECK_ONLY} unlock=${ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN}"
echo "[TRUETIME_JOINT_SELECTOR] proof=${TRUETIME_SELECTOR_GRAD_PROOF_JSON}"

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  "${VALIDATOR}" \
  "${SMOKE_TOOL}" \
  opentad/models/utils/truetime_geometry.py \
  opentad/models/detectors/truetime_joint_selector_smoke.py \
  opentad/models/selectors/truetime_joint_selector.py
"${PYTHON}" "${SMOKE_TOOL}" --config "${CONFIG}" --output-json "${TRUETIME_SELECTOR_GRAD_PROOF_JSON}"
"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}"
"${PYTHON}" "${VALIDATOR}" \
  --config "${EXEC_CONFIG}" \
  --allow-launch-unlocked \
  --require-grad-proof \
  --proof-json "${TRUETIME_SELECTOR_GRAD_PROOF_JSON}"
"${PYTHON}" -m pytest \
  tests/test_truetime_geometry.py \
  tests/test_truetime_joint_selector.py \
  tests/test_truetime_joint_selector_config.py \
  -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[TRUETIME_JOINT_SELECTOR] PRECHECK_ONLY complete; full train/paper/deploy claims remain locked"
  exit 0
fi

if [[ "${ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN}" != "1" ]]; then
  fail "ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN=1 is required beyond PRECHECK_ONLY"
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
  fail "formal full train must run inside a Slurm allocation/step"
fi

RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}/run}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/c3_truetime_joint_selector_c3_adatad_smoke/${RUN_TAG}}"
MASTER_PORT="$(pick_master_port)"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"
echo "[TRUETIME_JOINT_SELECTOR] train work_dir=${WORK_DIR} master_port=${MASTER_PORT}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${EXEC_CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" \
  2>&1 | tee "${RUN_DIR}/srun.out"
