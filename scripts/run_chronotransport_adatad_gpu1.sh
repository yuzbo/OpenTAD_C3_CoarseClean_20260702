#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CHRONOTRANSPORT_CONFIG:-configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_KIND="${CHRONOTRANSPORT_RUN_KIND:-stage_a_smoke}"
SEED="${CHRONOTRANSPORT_SEED:-42}"
RUN_ID="${CHRONOTRANSPORT_RUN_ID:-0}"
CHECKPOINT="${CHRONOTRANSPORT_CHECKPOINT:-}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
OUT_ROOT="${CHRONOTRANSPORT_OUT_ROOT:-${BASE}/chronotransport_runs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
export YUZIBO_ROOT="${BASE}"

fail() {
  printf 'ChronoTransport launcher: %s\n' "$*" >&2
  exit 2
}

case "${BASE}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "YUZIBO_ROOT must stay under /data/run01/sczc063/yuzibo" ;;
esac
case "${OUT_ROOT}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "CHRONOTRANSPORT_OUT_ROOT must stay under /data/run01/sczc063/yuzibo" ;;
esac

# This launcher is intentionally tied to physical GPU1. With one visible GPU,
# torchrun local_rank=0 maps to physical device 1.
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "CUDA_VISIBLE_DEVICES must be exactly 1 (physical GPU1)"

# A formal process must run in Slurm or in an explicitly documented protected
# allocation. The opt-in is deliberately verbose to prevent accidental login-
# node execution.
if [[ -z "${SLURM_JOB_ID:-}" && "${CHRONOTRANSPORT_PROTECTED_ALLOCATION:-0}" != "1" ]]; then
  fail "requires a Slurm allocation/step or CHRONOTRANSPORT_PROTECTED_ALLOCATION=1"
fi

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

if [[ "${CHRONOTRANSPORT_SKIP_ENV_SETUP:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load cuda/11.8
    module load miniforge3/24.11
  fi
  # shellcheck disable=SC1091
  source "${BASE}/conda_envs/opentad/bin/activate"
fi

export HOME="${CHRONOTRANSPORT_HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export HF_HOME="${HF_HOME:-${BASE}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${BASE}/tmp/torch_cache}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${HF_HOME}" "${TORCH_HOME}"

VALIDATOR_ARGS=(--config "${CONFIG}" --output "${OUT_ROOT}/validator_${CHRONOTRANSPORT_MODE:-dense}.json")
if [[ "${CHRONOTRANSPORT_MODE:-dense}" == "learned" && "${PRECHECK_ONLY}" != "1" ]]; then
  VALIDATOR_ARGS+=(--require-measured-cost --require-risk-ready)
fi

python tools/bata/validate_chronotransport_adatad.py "${VALIDATOR_ARGS[@]}"
python -m pytest -p no:cacheprovider \
  tests/test_chronotransport_core.py \
  tests/test_chronotransport_repository_contract.py \
  tests/test_chronotransport_vit_adapter_integration.py \
  tests/test_chronotransport_stage_a_smoke.py -q
python - <<'PY'
from pathlib import Path
paths = [
    Path("opentad/models/chronotransport/actions.py"),
    Path("opentad/models/chronotransport/cache.py"),
    Path("opentad/models/chronotransport/transport.py"),
    Path("opentad/models/chronotransport/risk.py"),
    Path("opentad/models/chronotransport/scheduler.py"),
    Path("opentad/models/chronotransport/losses.py"),
    Path("opentad/models/chronotransport/profiler.py"),
    Path("opentad/models/chronotransport/runtime.py"),
    Path("tools/bata/validate_chronotransport_adatad.py"),
    Path("tools/bata/check_chronotransport_checkpoint.py"),
]
for path in paths:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf 'ChronoTransport PRECHECK_ONLY=1 PASS; no GPU model job was launched.\n'
  exit 0
fi

[[ "${RUN_KIND}" == "stage_a_smoke" || "${RUN_KIND}" == "stage_a_eval" ]] || \
  fail "only stage_a_smoke and stage_a_eval are unlocked; Stage B/C remain gated"
[[ -n "${CHECKPOINT}" ]] || fail "CHRONOTRANSPORT_CHECKPOINT is required for Stage-A model execution"
[[ -f "${CHECKPOINT}" ]] || fail "checkpoint does not exist: ${CHECKPOINT}"

python tools/bata/check_chronotransport_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"

TEST_ARGS=("${CONFIG}" --checkpoint "${CHECKPOINT}" --seed "${SEED}" --id "${RUN_ID}")
if [[ "${RUN_KIND}" == "stage_a_smoke" ]]; then
  TEST_ARGS+=(--not_eval --max-batches "1")
fi

LAUNCH_LOG="${OUT_ROOT}/${RUN_KIND}_${CHRONOTRANSPORT_MODE:-dense}_seed${SEED}_id${RUN_ID}.launch.log"
printf 'Launching %s with mode=%s on physical GPU1\n' "${RUN_KIND}" "${CHRONOTRANSPORT_MODE:-dense}" | tee "${LAUNCH_LOG}"

# This executes one observable inference/evaluation job. It does not auto-start
# Stage B or Stage C and does not claim that a counterfactual regret ledger is
# complete; that runner remains a separate kill-gated implementation phase.
torchrun --standalone --nproc_per_node=1 tools/test.py "${TEST_ARGS[@]}" 2>&1 | tee -a "${LAUNCH_LOG}"
