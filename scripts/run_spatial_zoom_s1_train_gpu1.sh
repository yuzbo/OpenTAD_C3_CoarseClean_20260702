#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_TRAIN][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${SPATIAL_ZOOM_S1_RUN_ROOT:?set SPATIAL_ZOOM_S1_RUN_ROOT}"
MANIFEST="${SPATIAL_ZOOM_S1_MANIFEST:?set SPATIAL_ZOOM_S1_MANIFEST}"
ANNOTATION="${SPATIAL_ZOOM_S1_ANNOTATION:?set SPATIAL_ZOOM_S1_ANNOTATION}"
PRECHECK="${SPATIAL_ZOOM_S1_PRECHECK:?set SPATIAL_ZOOM_S1_PRECHECK}"
RESOLUTION="${SPATIAL_ZOOM_S1_RESOLUTION:?set SPATIAL_ZOOM_S1_RESOLUTION}"
SEED="${SPATIAL_ZOOM_S1_SEED:?set SPATIAL_ZOOM_S1_SEED}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1

case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "run root must stay under /data/run01/sczc063/yuzibo" ;;
esac
case "${RESOLUTION}" in
  160|224|256) ;;
  *) fail "resolution must be one of 160/224/256" ;;
esac
case "${SEED}" in
  3407|3408|3409) ;;
  *) fail "seed must be one of 3407/3408/3409" ;;
esac
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "CUDA_VISIBLE_DEVICES must be physical GPU1"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal S1 training requires a Slurm allocation"
[[ -f "${MANIFEST}" ]] || fail "manifest does not exist: ${MANIFEST}"
[[ -f "${ANNOTATION}" ]] || fail "annotation does not exist: ${ANNOTATION}"
[[ -f "${PRECHECK}" ]] || fail "full precheck certificate does not exist: ${PRECHECK}"

cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

EXPECTED_RUN_ROOT="$(python tools/bata/resolve_spatial_zoom_s1_experiment.py \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --precheck "${PRECHECK}" \
  --path-only)"
[[ "${RUN_ROOT}" == "${EXPECTED_RUN_ROOT}" ]] || \
  fail "SPATIAL_ZOOM_S1_RUN_ROOT must equal ${EXPECTED_RUN_ROOT}"

SOURCE_CONFIG="${ROOT}/configs/adatad/thumos/s1_dense${RESOLUTION}_videomae_s_768x1_adapter.py"
CONTROL_DIR="${RUN_ROOT}/control"
WORK_DIR="${RUN_ROOT}/dense${RESOLUTION}/seed${SEED}"
BOUND_CONFIG="${CONTROL_DIR}/dense${RESOLUTION}_seed${SEED}.py"
SELECTION="${WORK_DIR}/checkpoint_selection.json"
[[ -f "${SOURCE_CONFIG}" ]] || fail "source config does not exist: ${SOURCE_CONFIG}"
[[ ! -e "${WORK_DIR}" ]] || fail "refusing to reuse work dir: ${WORK_DIR}"
[[ ! -e "${BOUND_CONFIG}" ]] || fail "bound config already exists: ${BOUND_CONFIG}"
mkdir -p "${CONTROL_DIR}"

python tools/bata/build_spatial_zoom_s1_training_config.py \
  "${SOURCE_CONFIG}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --precheck "${PRECHECK}" \
  --seed "${SEED}" \
  --work-dir "${WORK_DIR}" \
  --output "${BOUND_CONFIG}"

torchrun --standalone --nproc_per_node=1 \
  tools/train.py "${BOUND_CONFIG}" --seed "${SEED}" --id 0

shopt -s nullglob
EVIDENCE_PATHS=("${WORK_DIR}"/gate_evidence/epoch_*.evidence.json)
shopt -u nullglob
[[ "${#EVIDENCE_PATHS[@]}" -gt 0 ]] || fail "training produced no gate evidence"
SELECT_ARGS=()
for evidence in "${EVIDENCE_PATHS[@]}"; do
  SELECT_ARGS+=(--evidence "${evidence}")
done
python tools/bata/select_spatial_zoom_s1_checkpoint.py \
  --config "${BOUND_CONFIG}" \
  --seed "${SEED}" \
  "${SELECT_ARGS[@]}" \
  --output "${SELECTION}"

printf '[SPATIAL_ZOOM_S1_TRAIN] PASS resolution=%s seed=%s selection=%s\n' \
  "${RESOLUTION}" "${SEED}" "${SELECTION}"
