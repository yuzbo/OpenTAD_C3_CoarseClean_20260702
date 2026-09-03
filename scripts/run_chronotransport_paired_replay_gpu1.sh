#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
CONFIG="${CHRONOTRANSPORT_CONFIG:-configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py}"
CHECKPOINT="${CHRONOTRANSPORT_CHECKPOINT:-}"
SCHEDULE="${CHRONOTRANSPORT_REPLAY_SCHEDULE:-periodic2_transport}"
LIMIT="${CHRONOTRANSPORT_REPLAY_LIMIT:-1}"
SPLIT="${CHRONOTRANSPORT_REPLAY_SPLIT:-diagnostic}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
OUT_ROOT="${CHRONOTRANSPORT_OUT_ROOT:-${BASE}/chronotransport_runs/paired_replay}"
OUTPUT="${CHRONOTRANSPORT_REPLAY_OUTPUT:-${OUT_ROOT}/${SCHEDULE}_${SPLIT}.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export YUZIBO_ROOT="${BASE}"
export CHRONOTRANSPORT_CONFIG="${CONFIG}"
export CHRONOTRANSPORT_CHECKPOINT="${CHECKPOINT}"
export CHRONOTRANSPORT_REPLAY_SPLIT="${SPLIT}"

fail() {
  printf 'ChronoTransport paired replay launcher: %s\n' "$*" >&2
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
case "${OUTPUT}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "CHRONOTRANSPORT_REPLAY_OUTPUT must stay under /data/run01/sczc063/yuzibo" ;;
esac
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "CUDA_VISIBLE_DEVICES must be exactly 1 (physical GPU1)"
[[ -n "${SLURM_JOB_ID:-}" || "${CHRONOTRANSPORT_PROTECTED_ALLOCATION:-0}" == "1" ]] || \
  fail "requires a Slurm allocation/step or protected allocation"
[[ "${SPLIT}" == "train" || "${SPLIT}" == "diagnostic" ]] || \
  fail "replay split must be train or diagnostic"
[[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || fail "CHRONOTRANSPORT_REPLAY_LIMIT must be positive"

cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
export HOME="${BASE}/tmp/home"
export XDG_CACHE_HOME="${BASE}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${BASE}/tmp/xdg_config"
export HF_HOME="${BASE}/hf_cache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${OUT_ROOT}" "$(dirname "${OUTPUT}")"

python tools/bata/validate_chronotransport_adatad.py \
  --config "${CONFIG}" \
  --output "${OUT_ROOT}/validator_replay.json"
python -m pytest -p no:cacheprovider tests/test_chronotransport_opentad_replay.py -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf 'ChronoTransport paired replay PRECHECK_ONLY=1 PASS; no GPU replay was launched.\n'
  exit 0
fi

[[ -n "${CHECKPOINT}" ]] || fail "CHRONOTRANSPORT_CHECKPOINT is required"
[[ -f "${CHECKPOINT}" ]] || fail "checkpoint does not exist: ${CHECKPOINT}"
python tools/bata/check_chronotransport_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"

python tools/bata/run_chronotransport_paired_replay.py \
  --factory tools.bata.chronotransport_opentad_factory:paired_replay_factory \
  --output "${OUTPUT}" \
  --schedule "${SCHEDULE}" \
  --limit "${LIMIT}"
