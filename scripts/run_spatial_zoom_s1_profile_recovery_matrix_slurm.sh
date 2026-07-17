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
MATRIX_LOCK_DIR="${CAMPAIGN_ROOT}/matrix.lock"
if ! mkdir "${MATRIX_LOCK_DIR}"; then
  fail "profile matrix lock already exists; refusing a concurrent or repeated matrix"
fi
[[ ! -d "${CAMPAIGN_ROOT}/descriptors" ]] || \
  fail "refusing to duplicate an already-started sidecar matrix"
case "${POWER_SCRATCH_ROOT}" in
  /tmp/*|/var/tmp/*) ;;
  *) fail "matrix sidecar scratch must be node-local" ;;
esac

export MATRIX_LOCK_DIR PROFILE_RECOVERY PROFILE_COMMIT FROZEN_ORDER CAMPAIGN_ROOT SCOPED_GPU_ID MEMORY_LIMIT_MB
(
  cd "${ROOT}"
  python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
)

certificate_path = Path(os.environ["PROFILE_RECOVERY"]).resolve()
gate_path = Path(os.environ["CAMPAIGN_ROOT"]).resolve() / "sidecar_gate.json"
certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
record = {
    "schema_version": "spatial_zoom_s1_profile_matrix_start_v1",
    "status": "RUNNING",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
    "scoped_gpu_id": os.environ["SCOPED_GPU_ID"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
    "slurm_memory_limit_mb": int(os.environ["MEMORY_LIMIT_MB"]),
    "profile_code_commit": os.environ["PROFILE_COMMIT"],
    "profile_recovery_certificate_path": str(certificate_path),
    "profile_recovery_certificate_file_sha256": sha256_file(certificate_path),
    "profile_recovery_certificate_sha256": certificate["certificate_sha256"],
    "profile_recovery_campaign_id": certificate["campaign_id"],
    "sidecar_gate_path": str(gate_path),
    "sidecar_gate_file_sha256": sha256_file(gate_path),
    "sidecar_gate_sha256": gate["gate_sha256"],
    "frozen_order": os.environ["FROZEN_ORDER"].split(),
}
record["matrix_sha256"] = canonical_sha256(record)
atomic_publish_json(
    Path(os.environ["MATRIX_LOCK_DIR"]) / "matrix.started.json",
    record,
)
PY
)

for cell in ${FROZEN_ORDER}; do
  export SPATIAL_ZOOM_S1_RESOLUTION="${cell%%:*}"
  export SPATIAL_ZOOM_S1_SEED="${cell##*:}"
  bash "${ROOT}/scripts/run_spatial_zoom_s1_test_profile_slurm.sh"
done

DESCRIPTOR_COUNT="$(find "${CAMPAIGN_ROOT}/descriptors" -maxdepth 1 -type f -name '*.run.json' | wc -l)"
[[ "${DESCRIPTOR_COUNT}" == "9" ]] || fail "recovery campaign did not publish nine descriptors"
export DESCRIPTOR_COUNT
(
  cd "${ROOT}"
  python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
)

campaign_root = Path(os.environ["CAMPAIGN_ROOT"]).resolve()
started_path = Path(os.environ["MATRIX_LOCK_DIR"]) / "matrix.started.json"
started = json.loads(started_path.read_text(encoding="utf-8"))
started_hash = started.pop("matrix_sha256", None)
if (
    not started_hash
    or canonical_sha256(started) != started_hash
    or started["slurm_job_id"] != os.environ["SLURM_JOB_ID"]
):
    raise RuntimeError("S1 matrix start receipt identity mismatch")
started["matrix_sha256"] = started_hash
descriptors = []
for path in sorted((campaign_root / "descriptors").glob("*.run.json")):
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptors.append(
        {
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "resolution": int(descriptor["resolution"]),
            "seed": int(descriptor["seed"]),
        }
    )
if len(descriptors) != int(os.environ["DESCRIPTOR_COUNT"]):
    raise RuntimeError("S1 matrix descriptor count changed during sealing")
descriptor_cells = {
    f'{row["resolution"]}:{row["seed"]}' for row in descriptors
}
if descriptor_cells != set(started["frozen_order"]):
    raise RuntimeError("S1 matrix descriptors differ from the frozen cells")
record = {
    "schema_version": "spatial_zoom_s1_profile_matrix_completion_v1",
    "status": "PASS",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "profile_code_commit": os.environ["PROFILE_COMMIT"],
    "profile_recovery_certificate_sha256": started[
        "profile_recovery_certificate_sha256"
    ],
    "sidecar_gate_sha256": started["sidecar_gate_sha256"],
    "matrix_started_path": str(started_path.resolve()),
    "matrix_started_file_sha256": sha256_file(started_path),
    "matrix_started_sha256": started["matrix_sha256"],
    "frozen_order": started["frozen_order"],
    "descriptors": descriptors,
}
record["matrix_sha256"] = canonical_sha256(record)
atomic_publish_json(
    Path(os.environ["MATRIX_LOCK_DIR"]) / "matrix.completed.json",
    record,
)
PY
)
printf '[SPATIAL_ZOOM_S1_PROFILE_RECOVERY_MATRIX] PASS campaign_root=%s\n' "${CAMPAIGN_ROOT}"
