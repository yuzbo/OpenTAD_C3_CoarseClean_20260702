#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
CONFIG="${CHRONOTRANSPORT_CONFIG:-configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py}"
CHECKPOINT="${CHRONOTRANSPORT_CHECKPOINT:-}"
GATE_A="${CHRONOTRANSPORT_DENSE_GATE_A:-}"
GATE_B="${CHRONOTRANSPORT_DENSE_GATE_B:-}"
SEED="${CHRONOTRANSPORT_STAGE_B_SEED:-3407}"
EPOCHS="${CHRONOTRANSPORT_STAGE_B_EPOCHS:-1}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
OUT_ROOT="${CHRONOTRANSPORT_OUT_ROOT:-${BASE}/chronotransport_runs/stage_b_formal_seed${SEED}}"
CHRONOTRANSPORT_SPLIT_MANIFEST="${CHRONOTRANSPORT_SPLIT_MANIFEST:-${OUT_ROOT}/split_manifest.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export YUZIBO_ROOT="${BASE}"
export CHRONOTRANSPORT_CONFIG="${CONFIG}"
export CHRONOTRANSPORT_CHECKPOINT="${CHECKPOINT}"
export CHRONOTRANSPORT_SPLIT_MANIFEST

fail() {
  printf 'ChronoTransport formal Stage-B launcher: %s\n' "$*" >&2
  exit 2
}

case "${BASE}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "YUZIBO_ROOT must stay under /data/run01/sczc063/yuzibo" ;;
esac
for path in "${OUT_ROOT}" "${CHRONOTRANSPORT_SPLIT_MANIFEST}"; do
  case "${path}" in
    /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
    *) fail "formal Stage-B outputs must stay under /data/run01/sczc063/yuzibo" ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "CUDA_VISIBLE_DEVICES must be exactly 1 (physical GPU1)"
[[ -n "${SLURM_JOB_ID:-}" || "${CHRONOTRANSPORT_PROTECTED_ALLOCATION:-0}" == "1" ]] || \
  fail "requires a Slurm allocation/step or protected allocation"
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "CHRONOTRANSPORT_STAGE_B_SEED must be non-negative"
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || fail "CHRONOTRANSPORT_STAGE_B_EPOCHS must be positive"

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
mkdir -p "${OUT_ROOT}"

python tools/bata/validate_chronotransport_adatad.py \
  --config "${CONFIG}" \
  --output "${OUT_ROOT}/validator_stage_b_formal.json"
python -m pytest -p no:cacheprovider \
  tests/test_chronotransport_core.py \
  tests/test_chronotransport_pipeline.py \
  tests/test_chronotransport_opentad_replay.py \
  tests/test_chronotransport_stage_b_formal.py -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf 'ChronoTransport formal Stage-B PRECHECK_ONLY=1 PASS; no experiment was launched.\n'
  exit 0
fi

[[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}" ]] || fail "compatible input checkpoint is required"
[[ -n "${GATE_A}" && -f "${GATE_A}" ]] || fail "CHRONOTRANSPORT_DENSE_GATE_A is required"
[[ -n "${GATE_B}" && -f "${GATE_B}" ]] || fail "CHRONOTRANSPORT_DENSE_GATE_B is required"
python tools/bata/check_chronotransport_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"
python tools/bata/validate_chronotransport_dense_gate.py \
  --first "${GATE_A}" \
  --second "${GATE_B}" \
  --tolerance 1e-6

python tools/bata/run_chronotransport_stage_b_formal.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${OUT_ROOT}" \
  --split-manifest "${CHRONOTRANSPORT_SPLIT_MANIFEST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}"
