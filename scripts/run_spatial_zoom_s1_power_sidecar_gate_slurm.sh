#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_SIDECAR_GATE][FAIL] %s\n' "$*" >&2
  exit 2
}

has_sidecar_runtime_evidence() {
  local scratch_dir="$1"
  local path
  for path in \
    "${scratch_dir}/pid.json" \
    "${scratch_dir}/ready.json" \
    "${scratch_dir}/power.jsonl" \
    "${scratch_dir}/result.json"; do
    [[ -e "${path}" ]] && return 0
  done
  return 1
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
RESOLUTION=256
SEED=3408
export PYTHONDONTWRITEBYTECODE=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Gate requires one Slurm allocation"
if [[ "${SPATIAL_ZOOM_S1_SINGLE_GPU_STEP:-0}" != "1" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a JOB_GPU_ARRAY <<< "${SLURM_JOB_GPUS:-}"
  if [[ "${#JOB_GPU_ARRAY[@]}" -gt 1 ]]; then
    export SPATIAL_ZOOM_S1_SINGLE_GPU_STEP=1
    exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
      bash "${ROOT}/scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh"
  fi
fi
SCOPED_GPU_ID="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "Gate requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "Gate requires one GPU"
[[ -n "${SCOPED_GPU_ID}" && "${SCOPED_GPU_ID}" != *,* ]] || \
  fail "Gate requires one step-scoped physical GPU identity"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "Gate requires exactly five CPUs"
command -v taskset >/dev/null 2>&1 || fail "Gate requires taskset"
case "${POWER_SCRATCH_ROOT}" in
  /tmp/*|/var/tmp/*) ;;
  *) fail "Gate sidecar scratch must be node-local" ;;
esac
mkdir -p "${POWER_SCRATCH_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

(
  cd "${ROOT}"
  python -c 'from tools.bata.spatial_zoom_s1_training import require_slurm_memory_limit_mb, require_slurm_single_gpu_allocation; require_slurm_single_gpu_allocation(); print(require_slurm_memory_limit_mb(minimum_mb=90000))'
)

ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
IFS=',' read -r -a CPU_ARRAY <<< "${ALLOCATED_CPUS}"
[[ "${#CPU_ARRAY[@]}" == "5" ]] || fail "Slurm affinity does not expose five CPUs"
DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]},${CPU_ARRAY[2]},${CPU_ARRAY[3]}"
SIDECAR_CPU="${CPU_ARRAY[4]}"

TRAINING_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_code_commit"])' "${PROFILE_RECOVERY}")"
PROFILE_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["profile_code_commit"])' "${PROFILE_RECOVERY}")"
CAMPAIGN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_root"])' "${PROFILE_RECOVERY}")"
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${PROFILE_COMMIT}" ]] || \
  fail "profile checkout differs from the recovery certificate"
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "profile checkout must be clean"
[[ "$(git -C "${TRAINING_ROOT}" rev-parse HEAD)" == "${TRAINING_COMMIT}" ]] || \
  fail "training checkout differs from the recovery certificate"
[[ -z "$(git -C "${TRAINING_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "training checkout must be clean"

WORK_DIR="${RUN_ROOT}/dense${RESOLUTION}/seed${SEED}"
BOUND_CONFIG="${RUN_ROOT}/control/dense${RESOLUTION}_seed${SEED}.py"
SELECTION="${WORK_DIR}/checkpoint_selection.json"
TEST_EVIDENCE="${WORK_DIR}/gpu1_id0/test_evidence/test.evidence.json"
for path in \
  "${MANIFEST}" "${ANNOTATION}" "${TEST_OPEN}" "${PROFILE_RECOVERY}" \
  "${BOUND_CONFIG}" "${SELECTION}" "${TEST_EVIDENCE}"; do
  [[ -f "${path}" ]] || fail "required Gate artifact does not exist: ${path}"
done
CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint_path"])' "${SELECTION}")"
[[ -f "${CHECKPOINT}" ]] || fail "selected checkpoint is missing"

GATE_EVIDENCE="${CAMPAIGN_ROOT}/sidecar_gate.json"
GATE_PREFIX="${CAMPAIGN_ROOT}/sidecar_gate/dense256_seed3408_long_full_path"
[[ ! -e "${GATE_EVIDENCE}" ]] || fail "sidecar Gate evidence already exists"
for suffix in started.json summary.json samples.jsonl power.jsonl power_attempt.json power_attempt.jsonl; do
  [[ ! -e "${GATE_PREFIX}.${suffix}" ]] || fail "sidecar Gate namespace already started"
done
TEST_EVIDENCE_SHA_BEFORE="$(sha256sum "${TEST_EVIDENCE}" | awk '{print $1}')"
GATE_SCRATCH_DIR="${POWER_SCRATCH_ROOT}/job${SLURM_JOB_ID}_dense256_seed3408_gate"
POWER_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i "${SCOPED_GPU_ID}" | tr -d '[:space:]')"
[[ "${POWER_UUID}" == GPU-* ]] || fail "could not resolve allocated GPU UUID"

if ! (
  cd "${TRAINING_ROOT}"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    taskset -c "${DETECTOR_CPUS}" \
    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="s1-sidecar-gate-${SLURM_JOB_ID}" \
      "${ROOT}/tools/bata/profile_spatial_zoom_s1.py" \
      "${BOUND_CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --manifest "${MANIFEST}" \
      --annotation "${ANNOTATION}" \
      --split test \
      --test-open-certificate "${TEST_OPEN}" \
      --test-evidence "${TEST_EVIDENCE}" \
      --profile-recovery-certificate "${PROFILE_RECOVERY}" \
      --sidecar-gate-evidence "${GATE_EVIDENCE}" \
      --sidecar-gate \
      --output-prefix "${GATE_PREFIX}" \
      --device cuda:0 \
      --seed "${SEED}" \
      --samples 0 \
      --warmup-samples 50 \
      --batch-size 1 \
      --loader-workers 0 \
      --amp \
      --use-ema \
      --sample-power \
      --power-gpu-id "${SCOPED_GPU_ID}" \
      --power-interval-ms 20 \
      --power-scratch-root "${POWER_SCRATCH_ROOT}" \
      --allocated-cpus "${ALLOCATED_CPUS}" \
      --detector-cpus "${DETECTOR_CPUS}" \
      --sidecar-cpu "${SIDECAR_CPU}"
); then
  if has_sidecar_runtime_evidence "${GATE_SCRATCH_DIR}"; then
    if python "${ROOT}/tools/bata/spatial_zoom_s1_power.py" salvage \
      --scratch-dir "${GATE_SCRATCH_DIR}" \
      --attempt-prefix "${GATE_PREFIX}" \
      --expected-uuid "${POWER_UUID}" \
      --interval-ms 20 \
      --sidecar-cpu-id "${SIDECAR_CPU}" \
      --detector-cpus "${DETECTOR_CPUS}" \
      --allocated-cpus "${ALLOCATED_CPUS}"; then
      fail "long sidecar Gate failed after sealing its sidecar attempt"
    fi
    fail "long sidecar Gate failed and sidecar salvage also failed"
  fi
  if [[ -e "${GATE_PREFIX}.started.json" ]]; then
    fail "long sidecar Gate consumed its namespace before sidecar startup"
  fi
  fail "long sidecar Gate failed before the sidecar published attempt evidence"
fi

[[ -f "${GATE_EVIDENCE}" ]] || fail "Gate evidence was not published"
TEST_EVIDENCE_SHA_AFTER="$(sha256sum "${TEST_EVIDENCE}" | awk '{print $1}')"
[[ "${TEST_EVIDENCE_SHA_BEFORE}" == "${TEST_EVIDENCE_SHA_AFTER}" ]] || \
  fail "Gate changed the existing official test evidence"
for suffix in summary.json samples.jsonl power.jsonl; do
  [[ ! -e "${GATE_PREFIX}.${suffix}" ]] || fail "Gate published a paper profile"
done
printf '[SPATIAL_ZOOM_S1_SIDECAR_GATE] PASS evidence=%s\n' "${GATE_EVIDENCE}"
