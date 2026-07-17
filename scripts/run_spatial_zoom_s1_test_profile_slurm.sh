#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_TEST_PROFILE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT:?set SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT}"
TRAINING_ROOT="${SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT:?set SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${SPATIAL_ZOOM_S1_RUN_ROOT:?set SPATIAL_ZOOM_S1_RUN_ROOT}"
MANIFEST="${SPATIAL_ZOOM_S1_MANIFEST:?set SPATIAL_ZOOM_S1_MANIFEST}"
ANNOTATION="${SPATIAL_ZOOM_S1_ANNOTATION:?set SPATIAL_ZOOM_S1_ANNOTATION}"
TEST_OPEN="${SPATIAL_ZOOM_S1_TEST_OPEN:?set SPATIAL_ZOOM_S1_TEST_OPEN}"
PROFILE_RECOVERY="${SPATIAL_ZOOM_S1_PROFILE_RECOVERY:?set SPATIAL_ZOOM_S1_PROFILE_RECOVERY}"
POWER_SCRATCH_ROOT="${SPATIAL_ZOOM_S1_POWER_SCRATCH_ROOT:?set SPATIAL_ZOOM_S1_POWER_SCRATCH_ROOT}"
RESOLUTION="${SPATIAL_ZOOM_S1_RESOLUTION:?set SPATIAL_ZOOM_S1_RESOLUTION}"
SEED="${SPATIAL_ZOOM_S1_SEED:?set SPATIAL_ZOOM_S1_SEED}"
PREFLIGHT_ONLY="${SPATIAL_ZOOM_S1_PREFLIGHT_ONLY:-0}"
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
case "${PREFLIGHT_ONLY}" in
  0|1) ;;
  *) fail "SPATIAL_ZOOM_S1_PREFLIGHT_ONLY must be 0 or 1" ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal S1 test/profile requires a Slurm allocation"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "formal S1 test/profile requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "Slurm allocation must expose one GPU"
[[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *,* ]] || \
  fail "SLURM_JOB_GPUS must identify exactly one allocated physical GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || \
  fail "formal S1 sidecar profile requires exactly five allocated CPUs"
[[ "${SLURM_MEM_PER_NODE:-0}" -ge 90000 ]] || \
  fail "formal S1 sidecar profile requires at least 90000 MiB node memory"
command -v taskset >/dev/null 2>&1 || fail "formal S1 sidecar profile requires taskset"

case "${POWER_SCRATCH_ROOT}" in
  /tmp/*|/var/tmp/*) ;;
  *) fail "power sidecar scratch must use node-local /tmp or /var/tmp" ;;
esac
mkdir -p "${POWER_SCRATCH_ROOT}"

WORK_DIR="${RUN_ROOT}/dense${RESOLUTION}/seed${SEED}"
BOUND_CONFIG="${RUN_ROOT}/control/dense${RESOLUTION}_seed${SEED}.py"
SELECTION="${WORK_DIR}/checkpoint_selection.json"
for path in "${MANIFEST}" "${ANNOTATION}" "${TEST_OPEN}" "${PROFILE_RECOVERY}" "${BOUND_CONFIG}" "${SELECTION}"; do
  [[ -f "${path}" ]] || fail "required artifact does not exist: ${path}"
done
[[ -d "${TRAINING_ROOT}/.git" || -f "${TRAINING_ROOT}/.git" ]] || \
  fail "training source root is not a Git checkout: ${TRAINING_ROOT}"
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || \
  fail "profile source root is not a Git checkout: ${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
IFS=',' read -r -a CPU_ARRAY <<< "${ALLOCATED_CPUS}"
[[ "${#CPU_ARRAY[@]}" == "5" ]] || fail "Slurm affinity does not expose five CPUs"
DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]},${CPU_ARRAY[2]},${CPU_ARRAY[3]}"
SIDECAR_CPU="${CPU_ARRAY[4]}"

TRAINING_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_code_commit"])' "${PROFILE_RECOVERY}")"
PROFILE_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["profile_code_commit"])' "${PROFILE_RECOVERY}")"
CAMPAIGN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_root"])' "${PROFILE_RECOVERY}")"
SIDECAR_GATE="${CAMPAIGN_ROOT}/sidecar_gate.json"
[[ -f "${SIDECAR_GATE}" ]] || fail "formal matrix requires the passed sidecar Gate"
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${PROFILE_COMMIT}" ]] || \
  fail "profile source root differs from the certificate-bound commit"
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "profile source root must be clean"
[[ "$(git -C "${TRAINING_ROOT}" rev-parse HEAD)" == "${TRAINING_COMMIT}" ]] || \
  fail "training source root differs from the certificate-bound commit"
[[ -z "$(git -C "${TRAINING_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "training source root must be clean"

cd "${ROOT}"

CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint_path"])' "${SELECTION}")"
[[ -f "${CHECKPOINT}" ]] || fail "selected checkpoint does not exist: ${CHECKPOINT}"
TEST_EVIDENCE="${WORK_DIR}/gpu1_id0/test_evidence/test.evidence.json"

(
  cd "${TRAINING_ROOT}"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    taskset -c "${DETECTOR_CPUS}" \
    python "${ROOT}/tools/bata/preflight_spatial_zoom_s1_profile.py" \
      --config "${BOUND_CONFIG}" \
      --seed "${SEED}" \
      --manifest "${MANIFEST}" \
      --annotation "${ANNOTATION}" \
      --checkpoint "${CHECKPOINT}" \
      --test-open-certificate "${TEST_OPEN}" \
      --profile-recovery-certificate "${PROFILE_RECOVERY}" \
      --sidecar-gate-evidence "${SIDECAR_GATE}" \
      --test-evidence "${TEST_EVIDENCE}" \
      --allocated-cpus "${ALLOCATED_CPUS}" \
      --detector-cpus "${DETECTOR_CPUS}" \
      --sidecar-cpu "${SIDECAR_CPU}"
)

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  [[ -f "${TEST_EVIDENCE}" ]] || \
    fail "preflight-only gate cannot open a previously unopened sealed test"
  printf '[SPATIAL_ZOOM_S1_TEST_PROFILE] PREFLIGHT PASS resolution=%s seed=%s\n' \
    "${RESOLUTION}" "${SEED}"
  exit 0
fi

if [[ -f "${TEST_EVIDENCE}" ]]; then
  printf '[SPATIAL_ZOOM_S1_TEST_PROFILE] reuse validated test evidence: %s\n' "${TEST_EVIDENCE}"
else
  (
    cd "${TRAINING_ROOT}"
    taskset -c "${DETECTOR_CPUS}" \
    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="s1-test-${SLURM_JOB_ID}-${RESOLUTION}-${SEED}" \
      tools/test.py "${BOUND_CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --seed "${SEED}" \
      --id 0 \
      --s1-test-open-certificate "${TEST_OPEN}"
  )
  (
    cd "${TRAINING_ROOT}"
    PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      taskset -c "${DETECTOR_CPUS}" \
      python "${ROOT}/tools/bata/preflight_spatial_zoom_s1_profile.py" \
        --config "${BOUND_CONFIG}" \
        --seed "${SEED}" \
        --manifest "${MANIFEST}" \
        --annotation "${ANNOTATION}" \
        --checkpoint "${CHECKPOINT}" \
        --test-open-certificate "${TEST_OPEN}" \
        --profile-recovery-certificate "${PROFILE_RECOVERY}" \
        --sidecar-gate-evidence "${SIDECAR_GATE}" \
        --test-evidence "${TEST_EVIDENCE}" \
        --allocated-cpus "${ALLOCATED_CPUS}" \
        --detector-cpus "${DETECTOR_CPUS}" \
        --sidecar-cpu "${SIDECAR_CPU}"
  )
fi

[[ -f "${TEST_EVIDENCE}" ]] || fail "sealed test evidence was not produced"
PROFILE_PREFIX="${CAMPAIGN_ROOT}/dense${RESOLUTION}/seed${SEED}/dense${RESOLUTION}_seed${SEED}"
PROFILE_SCRATCH_DIR="${POWER_SCRATCH_ROOT}/job${SLURM_JOB_ID}_dense${RESOLUTION}_seed${SEED}_formal"
POWER_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i "${SLURM_JOB_GPUS}" | tr -d '[:space:]')"
[[ "${POWER_UUID}" == GPU-* ]] || fail "could not resolve allocated GPU UUID"
if ! (
  cd "${TRAINING_ROOT}"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    taskset -c "${DETECTOR_CPUS}" \
    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="s1-profile-${SLURM_JOB_ID}-${RESOLUTION}-${SEED}" \
      "${ROOT}/tools/bata/profile_spatial_zoom_s1.py" \
      "${BOUND_CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --manifest "${MANIFEST}" \
      --annotation "${ANNOTATION}" \
      --split test \
      --test-open-certificate "${TEST_OPEN}" \
      --test-evidence "${TEST_EVIDENCE}" \
      --profile-recovery-certificate "${PROFILE_RECOVERY}" \
      --sidecar-gate-evidence "${SIDECAR_GATE}" \
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
      --power-interval-ms 20 \
      --power-scratch-root "${POWER_SCRATCH_ROOT}" \
      --allocated-cpus "${ALLOCATED_CPUS}" \
      --detector-cpus "${DETECTOR_CPUS}" \
      --sidecar-cpu "${SIDECAR_CPU}"
); then
  python "${ROOT}/tools/bata/spatial_zoom_s1_power.py" salvage \
    --scratch-dir "${PROFILE_SCRATCH_DIR}" \
    --attempt-prefix "${PROFILE_PREFIX}" \
    --expected-uuid "${POWER_UUID}" \
    --interval-ms 20 \
    --sidecar-cpu-id "${SIDECAR_CPU}" \
    --detector-cpus "${DETECTOR_CPUS}" \
    --allocated-cpus "${ALLOCATED_CPUS}" || true
  fail "formal S1 profile failed after sealing its sidecar attempt"
fi

PROFILE="${PROFILE_PREFIX}.summary.json"
DESCRIPTOR="${CAMPAIGN_ROOT}/descriptors/dense${RESOLUTION}_seed${SEED}.run.json"
mkdir -p "$(dirname "${DESCRIPTOR}")"
(
  cd "${TRAINING_ROOT}"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${ROOT}/tools/bata/build_spatial_zoom_s1_run_descriptor.py" \
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
)

printf '[SPATIAL_ZOOM_S1_TEST_PROFILE] PASS resolution=%s seed=%s descriptor=%s\n' \
  "${RESOLUTION}" "${SEED}" "${DESCRIPTOR}"
