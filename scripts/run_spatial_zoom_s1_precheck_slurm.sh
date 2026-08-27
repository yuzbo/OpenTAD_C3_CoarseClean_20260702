#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_PRECHECK][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
OUT_ROOT="${SPATIAL_ZOOM_S1_OUT_ROOT:-${BASE}/spatial_zoom_s1/precheck}"
MODE="${SPATIAL_ZOOM_S1_PRECHECK_MODE:-full}"
DEVICE="${SPATIAL_ZOOM_S1_DEVICE:-cuda:0}"
EXPECTED_PRETRAIN_SHA256="${SPATIAL_ZOOM_S1_PRETRAIN_SHA256:-}"
FROZEN_PRETRAIN_SHA256="4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
export PYTHONDONTWRITEBYTECODE=1

case "${OUT_ROOT}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "output must stay under /data/run01/sczc063/yuzibo" ;;
esac
[[ "${MODE}" == "static" || "${MODE}" == "clip" || "${MODE}" == "full" ]] || fail "unknown precheck mode ${MODE}"
if [[ "${MODE}" != "static" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || fail "clip/full CUDA precheck requires a Slurm allocation"
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
    fail "clip/full CUDA precheck requires exactly one Slurm-visible GPU"
  [[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "Slurm allocation must expose one GPU"
  [[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *,* ]] || \
    fail "SLURM_JOB_GPUS must identify exactly one allocated physical GPU"
fi
if [[ "${MODE}" == "full" && ! "${EXPECTED_PRETRAIN_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  fail "full precheck requires preregistered SPATIAL_ZOOM_S1_PRETRAIN_SHA256"
fi
if [[ "${MODE}" == "full" && "${EXPECTED_PRETRAIN_SHA256,,}" != "${FROZEN_PRETRAIN_SHA256}" ]]; then
  fail "SPATIAL_ZOOM_S1_PRETRAIN_SHA256 must equal the repository-frozen checkpoint identity"
fi

cd "${ROOT}"
mkdir -p "${OUT_ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

python -m py_compile \
  tools/bata/spatial_zoom_s1_contract.py \
  tools/bata/validate_spatial_zoom_s1.py \
  tools/bata/run_spatial_zoom_s1_precheck.py \
  tools/bata/spatial_zoom_s1_cost.py \
  tools/bata/profile_spatial_zoom_s1.py \
  tools/bata/analyze_spatial_zoom_s1_results.py
python -m pytest -p no:cacheprovider tests/test_spatial_zoom_s1_infrastructure.py -q
python tools/bata/validate_spatial_zoom_s1.py \
  --output "${OUT_ROOT}/config_matrix.json"

ARGS=(--mode "${MODE}" --output "${OUT_ROOT}/precheck_${MODE}.json")
if [[ "${MODE}" != "static" ]]; then
  ARGS+=(--device "${DEVICE}" --amp)
else
  ARGS+=(--device cpu)
fi
if [[ "${MODE}" == "full" ]]; then
  ARGS+=(--expected-pretrained-sha256 "${EXPECTED_PRETRAIN_SHA256,,}")
fi
python tools/bata/run_spatial_zoom_s1_precheck.py "${ARGS[@]}"

printf '[SPATIAL_ZOOM_S1_PRECHECK] PASS mode=%s output=%s\n' "${MODE}" "${OUT_ROOT}"
