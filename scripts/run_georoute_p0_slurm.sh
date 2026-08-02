#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_P0][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
CONFIG="${GEOROUTE_SOURCE_CONFIG:?set GEOROUTE_SOURCE_CONFIG}"
PRETRAINED="${GEOROUTE_PRETRAINED:?set GEOROUTE_PRETRAINED}"
OUTPUT="${GEOROUTE_P0_OUTPUT:?set GEOROUTE_P0_OUTPUT}"
RENDEZVOUS_OUTPUT="${OUTPUT%.json}.rendezvous.json"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
ROUTE_MODE="${GEOROUTE_P0_ROUTE_MODE:?set GEOROUTE_P0_ROUTE_MODE}"
ESTIMATOR="${GEOROUTE_P0_POLICY_ESTIMATOR:?set GEOROUTE_P0_POLICY_ESTIMATOR}"
TOKENS="${GEOROUTE_P0_TOKENS_PER_TUBELET:-32}"
CONTEXT="${GEOROUTE_P0_CONTEXT_TOKENS:-0}"
STRUCTURED_ROI="${GEOROUTE_P0_STRUCTURED_ROI_TOKENS:-0}"
STRUCTURED_RESIDUAL="${GEOROUTE_P0_STRUCTURED_RESIDUAL_TOKENS:-0}"
GEOMETRY_SHIFT="${GEOROUTE_P0_GEOMETRY_TEMPORAL_SHIFT_TUBELETS:-0}"
ROI_FRACTION="${GEOROUTE_P0_ROI_FRACTION:-0.5}"
POLICY_TEMPERATURE="${GEOROUTE_P0_POLICY_TEMPERATURE:-0.7}"
SCORE_FUNCTION_WEIGHT="${GEOROUTE_P0_SCORE_FUNCTION_WEIGHT:-1.0}"
SCORE_FUNCTION_BASELINE_MOMENTUM="${GEOROUTE_P0_SCORE_FUNCTION_BASELINE_MOMENTUM:-0.95}"
HEIGHT="${GEOROUTE_P0_HEIGHT:-160}"
WIDTH="${GEOROUTE_P0_WIDTH:-160}"
SEED="${GEOROUTE_P0_SEED:-3407}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'P0 requires Slurm'
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${SLURM_GPUS_ON_NODE:-1}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
    bash "${ROOT}/scripts/run_georoute_p0_slurm.sh"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'P0 requires one Slurm-visible GPU and must use logical cuda:0'
[[ -d "${ROOT}/.git" ]] || fail 'GeoRoute source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source snapshot is not clean'
[[ -f "${CONFIG}" && -f "${PRETRAINED}" ]] || fail 'P0 config or pretrained checkpoint is missing'
[[ ! -e "${OUTPUT}" ]] || fail 'P0 output namespace already exists'
[[ ! -e "${RENDEZVOUS_OUTPUT}" ]] || fail 'P0 rendezvous output namespace already exists'
[[ "${HEIGHT}" =~ ^[1-9][0-9]*$ && "${WIDTH}" =~ ^[1-9][0-9]*$ ]] || \
  fail 'P0 height and width must be positive decimal integers'
case "${OUTPUT}" in /data/run01/sczc063/yuzibo/*) ;; *) fail 'output must stay inside the remote write boundary' ;; esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__'

python -m tools.bata.georoute_rendezvous_gate \
  --output "${RENDEZVOUS_OUTPUT}" \
  --expected-commit "${EXPECTED_COMMIT}"
export GEOROUTE_P0_RENDEZVOUS_RECEIPT="${RENDEZVOUS_OUTPUT}"

args=(
  --config "${CONFIG}"
  --pretrained "${PRETRAINED}"
  --output "${OUTPUT}"
  --device cuda:0
  --route-mode "${ROUTE_MODE}"
  --policy-estimator "${ESTIMATOR}"
  --tokens-per-tubelet "${TOKENS}"
  --context-tokens "${CONTEXT}"
  --structured-roi-tokens "${STRUCTURED_ROI}"
  --structured-residual-tokens "${STRUCTURED_RESIDUAL}"
  --geometry-temporal-shift-tubelets "${GEOMETRY_SHIFT}"
  --roi-fraction "${ROI_FRACTION}"
  --policy-temperature "${POLICY_TEMPERATURE}"
  --score-function-weight "${SCORE_FUNCTION_WEIGHT}"
  --score-function-baseline-momentum "${SCORE_FUNCTION_BASELINE_MOMENTUM}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --seed "${SEED}"
)
if [[ -n "${GEOROUTE_P0_PILOT_ARM:-}" ]]; then
  args+=(--pilot-arm "${GEOROUTE_P0_PILOT_ARM}")
fi
if [[ -n "${GEOROUTE_P0_HYBRID_CAUSAL_ARM:-}" ]]; then
  args+=(--hybrid-causal-arm "${GEOROUTE_P0_HYBRID_CAUSAL_ARM}")
fi
for binding in \
  "geometry-side-channel:GEOROUTE_P0_GEOMETRY_SIDE_CHANNEL" \
  "absolute-position-enabled:GEOROUTE_P0_ABSOLUTE_POSITION_ENABLED" \
  "absolute-coordinates-enabled:GEOROUTE_P0_ABSOLUTE_COORDINATES_ENABLED" \
  "roi-relative-coordinates-enabled:GEOROUTE_P0_ROI_RELATIVE_COORDINATES_ENABLED" \
  "geometry-projection-enabled:GEOROUTE_P0_GEOMETRY_PROJECTION_ENABLED"; do
  flag="${binding%%:*}"
  variable="${binding#*:}"
  value="${!variable:-}"
  if [[ -n "${value}" ]]; then
    args+=("--${flag}" "${value}")
  fi
done
python -m tools.bata.run_georoute_p0_gate "${args[@]}"
