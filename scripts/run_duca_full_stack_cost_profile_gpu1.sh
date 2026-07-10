#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_FULL_STACK_COST][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_RANDOM_INIT="${ALLOW_RANDOM_INIT:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py}"
PROFILE_CHECKPOINT="${PROFILE_CHECKPOINT:-}"
PROFILE_METHOD="${PROFILE_METHOD:-duca-fixed384}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-30}"
PROFILE_WARMUP_SAMPLES="${PROFILE_WARMUP_SAMPLES:-5}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-1}"
PROFILE_SAMPLE_POWER="${PROFILE_SAMPLE_POWER:-1}"
PROFILE_POWER_INTERVAL_MS="${PROFILE_POWER_INTERVAL_MS:-20}"
PROFILE_POWER_GPU_ID="${PROFILE_POWER_GPU_ID:-}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${BASE}/projects/c3_lowres_action_probe/duca_cost_profiles/${PROFILE_METHOD}_$(date +%Y%m%d_%H%M%S_%z)}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export DUCA_PROFILE_RUNTIME=0
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "$(dirname "${OUTPUT_PREFIX}")"

[[ -f "${CONFIG}" ]] || fail "config missing: ${CONFIG}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"
[[ -x "${PYTHON}" ]] || fail "Python missing: ${PYTHON}"

if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "0" || "${CUDA_VISIBLE_DEVICES}" == "1" ]] \
    || fail "Slurm profile must see one logical GPU; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] \
    || fail "outside Slurm this launcher is restricted to physical GPU1"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/bata/duca_full_stack_cost.py \
  tools/bata/compare_duca_full_stack_cost.py \
  tools/bata/profile_duca_full_stack_cost.py
"${PYTHON}" -m pytest \
  tests/test_duca_full_stack_cost.py \
  tests/test_profile_duca_full_stack_cost_cli.py -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[DUCA_FULL_STACK_COST] PRECHECK_ONLY complete"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "real GPU profiling must run inside a Slurm allocation"
if [[ "${ALLOW_RANDOM_INIT}" != "1" ]]; then
  [[ -n "${PROFILE_CHECKPOINT}" ]] || fail "PROFILE_CHECKPOINT is required for a paper profile"
  [[ -f "${PROFILE_CHECKPOINT}" ]] || fail "checkpoint missing: ${PROFILE_CHECKPOINT}"
fi

ARGS=(
  "${CONFIG}"
  --output-prefix "${OUTPUT_PREFIX}"
  --method-name "${PROFILE_METHOD}"
  --config-commit "$(git rev-parse HEAD)"
  --backbone-pretrain "${ADATAD_PRETRAIN_PATH}"
  --device cuda:0
  --samples "${PROFILE_SAMPLES}"
  --warmup-samples "${PROFILE_WARMUP_SAMPLES}"
  --batch-size "${PROFILE_BATCH_SIZE}"
  --loader-workers 0
  --amp
)

if [[ "${ALLOW_RANDOM_INIT}" == "1" ]]; then
  ARGS+=(--allow-random-init)
else
  ARGS+=(--checkpoint "${PROFILE_CHECKPOINT}" --use-ema)
fi
if [[ "${PROFILE_SAMPLE_POWER}" == "1" ]]; then
  ARGS+=(--sample-power --power-interval-ms "${PROFILE_POWER_INTERVAL_MS}")
  if [[ -n "${PROFILE_POWER_GPU_ID}" ]]; then
    ARGS+=(--power-gpu-id "${PROFILE_POWER_GPU_ID}")
  fi
fi

echo "[DUCA_FULL_STACK_COST] method=${PROFILE_METHOD} samples=${PROFILE_SAMPLES} warmup=${PROFILE_WARMUP_SAMPLES}"
echo "[DUCA_FULL_STACK_COST] checkpoint=${PROFILE_CHECKPOINT:-random-init-smoke} output=${OUTPUT_PREFIX}"
"${PYTHON}" tools/bata/profile_duca_full_stack_cost.py "${ARGS[@]}"
