#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_TEST_PROFILE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_ROOT="${SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT:?set SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${SPATIAL_ZOOM_S1_RUN_ROOT:?set SPATIAL_ZOOM_S1_RUN_ROOT}"
MANIFEST="${SPATIAL_ZOOM_S1_MANIFEST:?set SPATIAL_ZOOM_S1_MANIFEST}"
ANNOTATION="${SPATIAL_ZOOM_S1_ANNOTATION:?set SPATIAL_ZOOM_S1_ANNOTATION}"
TEST_OPEN="${SPATIAL_ZOOM_S1_TEST_OPEN:?set SPATIAL_ZOOM_S1_TEST_OPEN}"
PROFILE_RECOVERY="${SPATIAL_ZOOM_S1_PROFILE_RECOVERY:?set SPATIAL_ZOOM_S1_PROFILE_RECOVERY}"
RESOLUTION="${SPATIAL_ZOOM_S1_RESOLUTION:?set SPATIAL_ZOOM_S1_RESOLUTION}"
SEED="${SPATIAL_ZOOM_S1_SEED:?set SPATIAL_ZOOM_S1_SEED}"
export PYTHONDONTWRITEBYTECODE=1

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
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal S1 test/profile requires a Slurm allocation"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "formal S1 test/profile requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "Slurm allocation must expose one GPU"
[[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *,* ]] || \
  fail "SLURM_JOB_GPUS must identify exactly one allocated physical GPU"

WORK_DIR="${RUN_ROOT}/dense${RESOLUTION}/seed${SEED}"
BOUND_CONFIG="${RUN_ROOT}/control/dense${RESOLUTION}_seed${SEED}.py"
SELECTION="${WORK_DIR}/checkpoint_selection.json"
for path in "${MANIFEST}" "${ANNOTATION}" "${TEST_OPEN}" "${PROFILE_RECOVERY}" "${BOUND_CONFIG}" "${SELECTION}"; do
  [[ -f "${path}" ]] || fail "required artifact does not exist: ${path}"
done
[[ -d "${TRAINING_ROOT}/.git" || -f "${TRAINING_ROOT}/.git" ]] || \
  fail "training source root is not a Git checkout: ${TRAINING_ROOT}"
TRAINING_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_code_commit"])' "${PROFILE_RECOVERY}")"
[[ "$(git -C "${TRAINING_ROOT}" rev-parse HEAD)" == "${TRAINING_COMMIT}" ]] || \
  fail "training source root differs from the certificate-bound commit"
[[ -z "$(git -C "${TRAINING_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "training source root must be clean"

cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint_path"])' "${SELECTION}")"
[[ -f "${CHECKPOINT}" ]] || fail "selected checkpoint does not exist: ${CHECKPOINT}"
TEST_EVIDENCE="${WORK_DIR}/gpu1_id0/test_evidence/test.evidence.json"

python tools/bata/preflight_spatial_zoom_s1_profile.py \
  --config "${BOUND_CONFIG}" \
  --seed "${SEED}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --checkpoint "${CHECKPOINT}" \
  --test-open-certificate "${TEST_OPEN}" \
  --profile-recovery-certificate "${PROFILE_RECOVERY}" \
  --test-evidence "${TEST_EVIDENCE}"

if [[ -f "${TEST_EVIDENCE}" ]]; then
  printf '[SPATIAL_ZOOM_S1_TEST_PROFILE] reuse validated test evidence: %s\n' "${TEST_EVIDENCE}"
else
  (
    cd "${TRAINING_ROOT}"
    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="s1-test-${SLURM_JOB_ID}-${RESOLUTION}-${SEED}" \
      tools/test.py "${BOUND_CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --seed "${SEED}" \
      --id 0 \
      --s1-test-open-certificate "${TEST_OPEN}"
  )
  cd "${ROOT}"
  python tools/bata/preflight_spatial_zoom_s1_profile.py \
    --config "${BOUND_CONFIG}" \
    --seed "${SEED}" \
    --manifest "${MANIFEST}" \
    --annotation "${ANNOTATION}" \
    --checkpoint "${CHECKPOINT}" \
    --test-open-certificate "${TEST_OPEN}" \
    --profile-recovery-certificate "${PROFILE_RECOVERY}" \
    --test-evidence "${TEST_EVIDENCE}"
fi

[[ -f "${TEST_EVIDENCE}" ]] || fail "sealed test evidence was not produced"
CAMPAIGN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_root"])' "${PROFILE_RECOVERY}")"
PROFILE_PREFIX="${CAMPAIGN_ROOT}/dense${RESOLUTION}/seed${SEED}/dense${RESOLUTION}_seed${SEED}"
torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="s1-profile-${SLURM_JOB_ID}-${RESOLUTION}-${SEED}" \
  tools/bata/profile_spatial_zoom_s1.py \
  "${BOUND_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --split test \
  --test-open-certificate "${TEST_OPEN}" \
  --test-evidence "${TEST_EVIDENCE}" \
  --profile-recovery-certificate "${PROFILE_RECOVERY}" \
  --output-prefix "${PROFILE_PREFIX}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --samples 0 \
  --warmup-samples 50 \
  --batch-size 1 \
  --loader-workers 0 \
  --amp \
  --use-ema \
  --sample-power \
  --power-gpu-id "${SLURM_JOB_GPUS}" \
  --power-interval-ms 20

PROFILE="${PROFILE_PREFIX}.summary.json"
DESCRIPTOR="${CAMPAIGN_ROOT}/descriptors/dense${RESOLUTION}_seed${SEED}.run.json"
mkdir -p "$(dirname "${DESCRIPTOR}")"
python tools/bata/build_spatial_zoom_s1_run_descriptor.py \
  --config "${BOUND_CONFIG}" \
  --seed "${SEED}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-selection "${SELECTION}" \
  --test-evidence "${TEST_EVIDENCE}" \
  --profile "${PROFILE}" \
  --profile-recovery-certificate "${PROFILE_RECOVERY}" \
  --output "${DESCRIPTOR}"

printf '[SPATIAL_ZOOM_S1_TEST_PROFILE] PASS resolution=%s seed=%s descriptor=%s\n' \
  "${RESOLUTION}" "${SEED}" "${DESCRIPTOR}"
