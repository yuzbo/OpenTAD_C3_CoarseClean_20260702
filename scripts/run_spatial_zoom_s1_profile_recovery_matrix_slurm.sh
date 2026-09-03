#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_PROFILE_RECOVERY_MATRIX][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT:?set SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
PROFILE_RECOVERY="${SPATIAL_ZOOM_S1_PROFILE_RECOVERY:?set SPATIAL_ZOOM_S1_PROFILE_RECOVERY}"
POWER_SCRATCH_ROOT="${SPATIAL_ZOOM_S1_POWER_SCRATCH_ROOT:?set SPATIAL_ZOOM_S1_POWER_SCRATCH_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal recovery matrix requires one Slurm allocation"
if [[ "${SPATIAL_ZOOM_S1_SINGLE_GPU_STEP:-0}" != "1" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a JOB_GPU_ARRAY <<< "${SLURM_JOB_GPUS:-}"
  if [[ "${#JOB_GPU_ARRAY[@]}" -gt 1 ]]; then
    export SPATIAL_ZOOM_S1_SINGLE_GPU_STEP=1
    exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
      bash "${ROOT}/scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh"
  fi
fi
SCOPED_GPU_ID="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
[[ -f "${PROFILE_RECOVERY}" ]] || fail "profile recovery certificate does not exist"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "sidecar matrix requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "sidecar matrix step requires one GPU"
[[ -n "${SCOPED_GPU_ID}" && "${SCOPED_GPU_ID}" != *,* ]] || \
  fail "sidecar matrix requires one step-scoped physical GPU identity"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "sidecar matrix requires five CPUs"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
MEMORY_LIMIT_MB="$(
  cd "${ROOT}"
  python -c 'from tools.bata.spatial_zoom_s1_training import require_slurm_memory_limit_mb; print(require_slurm_memory_limit_mb(minimum_mb=90000))'
)"
command -v taskset >/dev/null 2>&1 || fail "formal recovery matrix requires taskset"
ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
IFS=',' read -r -a CPU_ARRAY <<< "${ALLOCATED_CPUS}"
[[ "${#CPU_ARRAY[@]}" == "5" ]] || fail "Slurm affinity does not expose five CPUs"
DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]},${CPU_ARRAY[2]},${CPU_ARRAY[3]}"
SIDECAR_CPU="${CPU_ARRAY[4]}"

PROFILE_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["profile_code_commit"])' "${PROFILE_RECOVERY}")"
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || \
  fail "profile source root is not a Git checkout: ${ROOT}"
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${PROFILE_COMMIT}" ]] || \
  fail "profile source root differs from the certificate-bound commit"
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "profile source root must be clean"

EXPECTED_ORDER="$(
  cd "${ROOT}"
  python -c 'from tools.bata.spatial_zoom_s1_contract import build_s1_profile_order; print(" ".join("{}:{}".format(row["resolution"], row["seed"]) for row in build_s1_profile_order()))'
)"
FROZEN_ORDER="256:3408 224:3409 256:3409 224:3407 160:3407 224:3408 160:3408 160:3409 256:3407"
[[ "${EXPECTED_ORDER}" == "${FROZEN_ORDER}" ]] || fail "profile order differs from the frozen contract"

CAMPAIGN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_root"])' "${PROFILE_RECOVERY}")"
[[ -f "${CAMPAIGN_ROOT}/sidecar_gate.json" ]] || fail "passed long sidecar Gate is missing"
(
  cd "${ROOT}"
  python - "${PROFILE_RECOVERY}" "${CAMPAIGN_ROOT}/sidecar_gate.json" <<'PY'
import json
import sys
from pathlib import Path

from tools.bata.spatial_zoom_s1_sidecar_gate import (
    validate_sidecar_gate_evidence,
)

recovery = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
validate_sidecar_gate_evidence(gate, recovery=recovery, verify_artifacts=True)
PY
)
[[ ! -d "${CAMPAIGN_ROOT}/descriptors" ]] || \
  fail "refusing to duplicate an already-started sidecar matrix"
case "${POWER_SCRATCH_ROOT}" in
  /tmp/*|/var/tmp/*) ;;
  *) fail "matrix sidecar scratch must be node-local" ;;
esac

export SPATIAL_ZOOM_S1_MATRIX_DRY_RUN=1
for cell in ${FROZEN_ORDER}; do
  export SPATIAL_ZOOM_S1_RESOLUTION="${cell%%:*}"
  export SPATIAL_ZOOM_S1_SEED="${cell##*:}"
  bash "${ROOT}/scripts/run_spatial_zoom_s1_test_profile_slurm.sh"
done
unset SPATIAL_ZOOM_S1_MATRIX_DRY_RUN

export SCOPED_GPU_ID MEMORY_LIMIT_MB ALLOCATED_CPUS DETECTOR_CPUS SIDECAR_CPU
START_RECEIPT_CANDIDATE="$(
  mktemp "${POWER_SCRATCH_ROOT%/}/s1_matrix_start_${SLURM_JOB_ID}_${SLURM_STEP_ID:-nostep}_XXXXXX.json"
)"
export START_RECEIPT_CANDIDATE PROFILE_RECOVERY PROFILE_COMMIT CAMPAIGN_ROOT
trap 'rm -f -- "${START_RECEIPT_CANDIDATE:-}"' EXIT
(
  cd "${ROOT}"
  taskset -c "${DETECTOR_CPUS}" python - <<'PY'
import json
import os
from pathlib import Path

import torch

from tools.bata.profile_spatial_zoom_s1 import (
    _hardware_identity,
    _software_identity,
)
from tools.bata.spatial_zoom_s1_contract import (
    build_s1_profile_order,
    canonical_sha256,
)
from tools.bata.spatial_zoom_s1_matrix import (
    build_profile_matrix_start_receipt,
    validate_profile_matrix_start_receipt,
)
from tools.bata.spatial_zoom_s1_training import (
    require_slurm_memory_limit_mb,
    require_slurm_single_gpu_allocation,
)

certificate_path = Path(os.environ["PROFILE_RECOVERY"]).resolve()
gate_path = Path(os.environ["CAMPAIGN_ROOT"]).resolve() / "sidecar_gate.json"
certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
physical_gpu_id = require_slurm_single_gpu_allocation()
if physical_gpu_id != os.environ["SCOPED_GPU_ID"]:
    raise RuntimeError("S1 matrix shell and Python resolved different step GPUs")
memory_limit_mb = require_slurm_memory_limit_mb(minimum_mb=90000)
if memory_limit_mb != int(os.environ["MEMORY_LIMIT_MB"]):
    raise RuntimeError("S1 matrix effective memory changed after shell preflight")

allocated_cpu_ids = tuple(
    int(value) for value in os.environ["ALLOCATED_CPUS"].split(",")
)
detector_cpu_ids = tuple(
    int(value) for value in os.environ["DETECTOR_CPUS"].split(",")
)
sidecar_cpu_id = int(os.environ["SIDECAR_CPU"])
device = torch.device("cuda:0")
torch.cuda.set_device(device)
hardware_identity = _hardware_identity(
    torch,
    device,
    physical_gpu_id=physical_gpu_id,
    allocated_cpu_ids=allocated_cpu_ids,
    detector_cpu_ids=detector_cpu_ids,
    sidecar_cpu_id=sidecar_cpu_id,
    memory_limit_mb=memory_limit_mb,
)
software_fingerprint = canonical_sha256(_software_identity(torch))
receipt = build_profile_matrix_start_receipt(
    recovery=certificate_path,
    sidecar_gate=gate_path,
    hardware_identity=hardware_identity,
    software_fingerprint=software_fingerprint,
    profile_code_commit=certificate["profile_code_commit"],
    frozen_order=build_s1_profile_order(),
)
validate_profile_matrix_start_receipt(
    receipt,
    recovery=certificate_path,
    verify_runtime=True,
    hardware_identity=hardware_identity,
    software_fingerprint=software_fingerprint,
    effective_memory_limit_mb=memory_limit_mb,
)
candidate_path = Path(os.environ["START_RECEIPT_CANDIDATE"]).resolve()
candidate_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(candidate_path)
PY
)

MATRIX_LOCK_DIR="${CAMPAIGN_ROOT}/matrix.lock"
if ! mkdir "${MATRIX_LOCK_DIR}"; then
  fail "profile matrix lock already exists; refusing a concurrent or repeated matrix"
fi
export MATRIX_LOCK_DIR
(
  cd "${ROOT}"
  python - <<'PY'
import json
import os
from pathlib import Path

from tools.bata.spatial_zoom_s1_contract import atomic_publish_json
from tools.bata.spatial_zoom_s1_matrix import (
    canonical_matrix_start_path,
    validate_profile_matrix_start_receipt,
)

certificate_path = Path(os.environ["PROFILE_RECOVERY"]).resolve()
candidate_path = Path(os.environ["START_RECEIPT_CANDIDATE"]).resolve()
receipt = json.loads(candidate_path.read_text(encoding="utf-8"))
validate_profile_matrix_start_receipt(receipt, recovery=certificate_path)
receipt_path = canonical_matrix_start_path(certificate_path)
if receipt_path.parent != Path(os.environ["MATRIX_LOCK_DIR"]).resolve():
    raise RuntimeError("S1 matrix lock differs from the canonical receipt namespace")
atomic_publish_json(receipt_path, receipt)
validate_profile_matrix_start_receipt(receipt_path, recovery=certificate_path)
print(receipt_path)
PY
)
MATRIX_STARTED="${MATRIX_LOCK_DIR}/matrix.started.json"
[[ -f "${MATRIX_STARTED}" ]] || fail "matrix start receipt was not published"
export SPATIAL_ZOOM_S1_MATRIX_STARTED="${MATRIX_STARTED}"

for cell in ${FROZEN_ORDER}; do
  export SPATIAL_ZOOM_S1_RESOLUTION="${cell%%:*}"
  export SPATIAL_ZOOM_S1_SEED="${cell##*:}"
  bash "${ROOT}/scripts/run_spatial_zoom_s1_test_profile_slurm.sh"
done

mapfile -t DESCRIPTOR_PATHS < <(
  find "${CAMPAIGN_ROOT}/descriptors" -maxdepth 1 -type f -name '*.run.json' | sort
)
[[ "${#DESCRIPTOR_PATHS[@]}" == "9" ]] || \
  fail "recovery campaign did not publish nine descriptors"
(
  cd "${ROOT}"
  python - "${MATRIX_STARTED}" "${PROFILE_RECOVERY}" "${DESCRIPTOR_PATHS[@]}" <<'PY'
import sys
from pathlib import Path

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
)
from tools.bata.spatial_zoom_s1_matrix import (
    build_profile_matrix_completion_receipt,
    canonical_matrix_completion_path,
    validate_profile_matrix_completion_receipt,
)

start_path = Path(sys.argv[1]).resolve()
recovery_path = Path(sys.argv[2]).resolve()
descriptor_paths = [Path(value).resolve() for value in sys.argv[3:]]
receipt = build_profile_matrix_completion_receipt(
    start_receipt_path=start_path,
    recovery=recovery_path,
    descriptor_paths=descriptor_paths,
)
completion_path = canonical_matrix_completion_path(recovery_path)
atomic_publish_json(completion_path, receipt)
validate_profile_matrix_completion_receipt(
    completion_path,
    recovery=recovery_path,
    descriptor_paths=descriptor_paths,
)
print(completion_path)
PY
)
printf '[SPATIAL_ZOOM_S1_PROFILE_RECOVERY_MATRIX] PASS campaign_root=%s\n' "${CAMPAIGN_ROOT}"
