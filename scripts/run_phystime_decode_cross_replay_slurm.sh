#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime decode cross replay] ERROR: $*" >&2
  exit 1
}

BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?runtime commit is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?runtime tree is required}"
SOURCE_COMMIT="${PHYSTIME_SOURCE_COMMIT:?source commit is required}"
SOURCE_TREE="${PHYSTIME_SOURCE_TREE:?source tree is required}"
ARM="${PHYSTIME_DECODE_ARM:?arm is required}"
WEIGHTS_SOURCE="${PHYSTIME_DECODE_WEIGHTS_SOURCE:?weights source is required}"
CONFIG="${PHYSTIME_DECODE_CONFIG:?config is required}"
SELECTED_CONFIG="${PHYSTIME_DECODE_SELECTED_CONFIG:?selected config is required}"
PHYSICAL_CONFIG="${PHYSTIME_DECODE_PHYSICAL_CONFIG:?physical config is required}"
RUN_DIR="${PHYSTIME_DECODE_RUN_DIR:?run dir is required}"
SOURCE_DIR="${PHYSTIME_DECODE_SOURCE_DIR:?source dir is required}"
SELECTED_SOURCE_DIR="${PHYSTIME_SELECTED_SOURCE_DIR:?selected source dir is required}"
PHYSICAL_SOURCE_DIR="${PHYSTIME_PHYSICAL_SOURCE_DIR:?physical source dir is required}"
CHECKPOINT="${PHYSTIME_DECODE_CHECKPOINT:?checkpoint is required}"
SELECTED_CHECKPOINT="${PHYSTIME_SELECTED_CHECKPOINT:?selected checkpoint is required}"
PHYSICAL_CHECKPOINT="${PHYSTIME_PHYSICAL_CHECKPOINT:?physical checkpoint is required}"
VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?VideoMAE checkpoint is required}"
P0_RUN_ROOT="${PHYSTIME_P0_RUN_ROOT:?P0 run root is required}"
P0_COMPLETION="${PHYSTIME_DECODE_P0_COMPLETION:?P0 completion is required}"
GATE="${PHYSTIME_DECODE_GATE_OUTPUT:?gate is required}"
SEED="${PHYSTIME_SEED:-42}"
EVALUATION_EPOCH="${PHYSTIME_EVALUATION_EPOCH:-59}"
JOB_VARIANT="${PHYSTIME_JOB_VARIANT:?job variant is required}"
SBATCH_PATH="${PHYSTIME_SBATCH_PATH:?sbatch path is required}"
EXPECTED_DEPENDENCY="${PHYSTIME_EXPECTED_DEPENDENCY:?dependency is required}"
SLURM_LOG_ROOT="${PHYSTIME_DECODE_SLURM_LOG_ROOT:?Slurm log root is required}"
PREFLIGHT="${PHYSTIME_DECODE_PREFLIGHT:?preflight manifest is required}"
PREFLIGHT_SHA256="${PHYSTIME_DECODE_PREFLIGHT_SHA256:?preflight SHA256 is required}"
DAG_TOKEN="${PHYSTIME_DAG_TOKEN:?DAG token is required}"
EXPECTED_COMMENT="${PHYSTIME_EXPECTED_JOB_COMMENT:?job comment is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "replay must run inside Slurm"
[[ -n "${SLURM_JOB_NAME:-}" ]] || fail "Slurm job name is missing"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a GPU"
[[ "${SLURM_JOB_NAME}" == "pt_dc_${JOB_VARIANT}" ]] \
  || fail "replay Slurm job name mismatch"
[[ "$(scontrol show job -o "${SLURM_JOB_ID}")" == *"Comment=${EXPECTED_COMMENT}"* ]] \
  || fail "replay Slurm comment mismatch"
[[ "${SEED}" == "42" && "${EVALUATION_EPOCH}" == "59" ]] \
  || fail "replay is fixed to seed 42 and epoch 59"
for path in \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${VIDEOMAE_CHECKPOINT}" \
  "${PREFLIGHT}" \
  "${P0_COMPLETION}" \
  "${GATE}" \
  "${SOURCE_DIR}/FULL_COMPLETE.json" \
  "${SOURCE_DIR}/run_manifest.json"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done
[[ ! -e "${RUN_DIR}" ]] || fail "run directory already exists: ${RUN_DIR}"

case "${ARM}|${WEIGHTS_SOURCE}|$(basename "${CONFIG}")" in
  selected_axis\|online\|phystime_g1a_selected_axis_native_j192_decode_replay.py) ;;
  selected_axis\|ema\|phystime_g1a_selected_axis_native_j192_decode_replay.py) ;;
  physical_metric\|online\|phystime_g1a_physical_metric_native_j192_decode_replay.py) ;;
  physical_metric\|ema\|phystime_g1a_physical_metric_native_j192_decode_replay.py) ;;
  *) fail "arm/weights/config condition is outside the frozen suite" ;;
esac

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PHYSTIME_CHECKPOINT_PATH="${CHECKPOINT}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] \
  || fail "runtime commit mismatch"
[[ "$(git rev-parse 'HEAD^{tree}')" == "${EXPECTED_TREE}" ]] \
  || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

mkdir -p "${RUN_DIR}"
RUNTIME_PREFLIGHT="${RUN_DIR}/runtime_preflight_manifest.json"
"${PYTHON}" tools/bata/preflight_phystime_decode_cross.py \
  --selected-config "${SELECTED_CONFIG}" \
  --physical-config "${PHYSICAL_CONFIG}" \
  --selected-checkpoint "${SELECTED_CHECKPOINT}" \
  --physical-checkpoint "${PHYSICAL_CHECKPOINT}" \
  --videomae-checkpoint "${VIDEOMAE_CHECKPOINT}" \
  --selected-source-dir "${SELECTED_SOURCE_DIR}" \
  --physical-source-dir "${PHYSICAL_SOURCE_DIR}" \
  --p0-run-root "${P0_RUN_ROOT}" \
  --expected-runtime-commit "${EXPECTED_COMMIT}" \
  --expected-runtime-tree "${EXPECTED_TREE}" \
  --output "${RUNTIME_PREFLIGHT}"
RUNTIME_PREFLIGHT_SHA256="$(sha256sum "${RUNTIME_PREFLIGHT}" | awk '{print $1}')"
[[ "${RUNTIME_PREFLIGHT_SHA256}" == "${PREFLIGHT_SHA256}" ]] \
  || fail "replay-time full preflight differs from submission preflight"

"${PYTHON}" - \
  "${GATE}" "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" \
  "${ARM}" "${WEIGHTS_SOURCE}" "${CHECKPOINT}" "${P0_COMPLETION}" \
  "${PREFLIGHT}" "${PREFLIGHT_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    gate_path,
    commit,
    tree,
    arm,
    weights,
    checkpoint,
    p0_path,
    preflight_path,
    preflight_sha,
) = sys.argv[1:]
gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
if gate.get("gate_pass") is not True:
    raise SystemExit("decode cross gate did not pass")
if gate["runtime"]["commit"] != commit or gate["runtime"]["git_tree"] != tree:
    raise SystemExit("decode cross gate snapshot mismatch")
preflight_digest = hashlib.sha256(
    Path(preflight_path).read_bytes()
).hexdigest()
if (
    preflight_digest != preflight_sha
    or gate.get("preflight", {}).get("sha256") != preflight_sha
    or Path(gate["preflight"]["path"]).resolve()
    != Path(preflight_path).resolve()
):
    raise SystemExit("decode cross preflight binding mismatch")
condition = (
    ("selected" if arm == "selected_axis" else "physical")
    + "_"
    + weights
)
real_window = gate.get("real_windows", {}).get(condition, {})
if (
    gate.get("all_native_direct_exact_equivalence") is not True
    or real_window.get("native_direct_exact_equivalence") is not True
    or real_window.get("raw_tensors_immutable") is not True
):
    raise SystemExit("decode cross four-condition gate mismatch")
p0 = json.loads(Path(p0_path).read_text(encoding="utf-8"))
if (
    p0.get("schema_version")
    != "phystime_p0_fullprecision_completion_v2"
    or p0.get("runtime_commit")
    != "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
    or p0.get("runtime_tree")
    != "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
    or p0["artifacts"]["gate"]["sha256"]
    != "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
    or p0.get("validation_pass") is not True
    or p0.get("arm") != arm
    or p0.get("weights_source") != weights
):
    raise SystemExit("P0 provenance condition mismatch")
digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
if p0["artifacts"]["checkpoint"]["sha256"] != digest:
    raise SystemExit("checkpoint differs from P0 provenance")
PY

USE_EMA="False"
if [[ "${WEIGHTS_SOURCE}" == "ema" ]]; then
  USE_EMA="True"
fi

"${PYTHON}" - \
  "${RUN_DIR}" "${CONFIG}" "${GATE}" "${SOURCE_DIR}" "${CHECKPOINT}" \
  "${VIDEOMAE_CHECKPOINT}" "${P0_COMPLETION}" "${ARM}" \
  "${WEIGHTS_SOURCE}" "${USE_EMA}" "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" \
  "${SOURCE_COMMIT}" "${SOURCE_TREE}" "${SEED}" "${JOB_VARIANT}" \
  "${SBATCH_PATH}" "${EXPECTED_DEPENDENCY}" "${SLURM_LOG_ROOT}" \
  "${PREFLIGHT}" "${PREFLIGHT_SHA256}" "${DAG_TOKEN}" \
  "${EXPECTED_COMMENT}" "${RUNTIME_PREFLIGHT}" \
  "${RUNTIME_PREFLIGHT_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from mmengine.config import Config

from opentad.cores.phystime_decode_replay_capture import (
    decode_replay_effective_config_sha256,
)
from opentad.utils import update_workdir

(
    run_dir,
    config_path,
    gate_path,
    source_dir,
    checkpoint,
    videomae_checkpoint,
    p0_completion,
    arm,
    weights_source,
    use_ema,
    runtime_commit,
    runtime_tree,
    source_commit,
    source_tree,
    seed,
    job_variant,
    sbatch_path,
    expected_dependency,
    slurm_log_root,
    preflight_path,
    preflight_sha256,
    dag_token,
    expected_comment,
    runtime_preflight_path,
    runtime_preflight_sha256,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def normalize_state_dict(state):
    state = dict(state)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state

def state_dict_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(
            tensor.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
    return digest.hexdigest()

cfg = Config.fromfile(config_path, lazy_import=False)
cfg.merge_from_dict(
    {
        "work_dir": str(Path(run_dir, "direct_work").resolve()),
        "solver.ema": use_ema == "True",
        "model.backbone.custom.pretrain": str(
            Path(videomae_checkpoint).resolve()
        ),
        "inference.phystime_decode_replay_capture.weights_source": (
            weights_source
        ),
    }
)
cfg = update_workdir(cfg, 0, 1)
checkpoint_payload = torch.load(checkpoint, map_location="cpu")
checkpoint_state_key = (
    "state_dict_ema" if weights_source == "ema" else "state_dict"
)
checkpoint_state_sha256 = state_dict_sha256(
    normalize_state_dict(checkpoint_payload[checkpoint_state_key])
)
manifest = {
    "schema_version": "phystime_decode_cross_run_manifest_v1",
    "track": "frozen_epoch59_same_raw_tensor_dual_axis_decode",
    "arm": arm,
    "weights_source": weights_source,
    "solver_ema": use_ema == "True",
    "runtime_commit": runtime_commit,
    "runtime_tree": runtime_tree,
    "source_commit": source_commit,
    "source_tree": source_tree,
    "config": str(Path(config_path).resolve()),
    "effective_config_sha256": decode_replay_effective_config_sha256(cfg),
    "gate": str(Path(gate_path).resolve()),
    "gate_sha256": sha256_file(gate_path),
    "source_dir": str(Path(source_dir).resolve()),
    "source_completion": str(
        Path(source_dir, "FULL_COMPLETE.json").resolve()
    ),
    "source_completion_sha256": sha256_file(
        Path(source_dir, "FULL_COMPLETE.json")
    ),
    "source_manifest": str(Path(source_dir, "run_manifest.json").resolve()),
    "source_manifest_sha256": sha256_file(
        Path(source_dir, "run_manifest.json")
    ),
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": sha256_file(checkpoint),
    "checkpoint_state_key": checkpoint_state_key,
    "checkpoint_state_dict_sha256": checkpoint_state_sha256,
    "videomae_checkpoint": str(Path(videomae_checkpoint).resolve()),
    "videomae_checkpoint_sha256": sha256_file(videomae_checkpoint),
    "p0_completion": str(Path(p0_completion).resolve()),
    "p0_completion_sha256": sha256_file(p0_completion),
    "preflight_manifest": str(Path(preflight_path).resolve()),
    "preflight_manifest_sha256": preflight_sha256,
    "runtime_preflight_manifest": str(
        Path(runtime_preflight_path).resolve()
    ),
    "runtime_preflight_manifest_sha256": runtime_preflight_sha256,
    "evaluation_epoch": 59,
    "seed": int(seed),
    "new_training": False,
    "frozen_checkpoint_replay": True,
    "shared_raw_tensor_dual_decode": True,
    "slurm": {
        "job_id": os.environ["SLURM_JOB_ID"],
        "job_name": os.environ["SLURM_JOB_NAME"],
        "job_variant": job_variant,
        "dag_token": dag_token,
        "comment": expected_comment,
        "expected_dependency": expected_dependency,
        "sbatch_path": str(Path(sbatch_path).resolve()),
        "sbatch_sha256": sha256_file(sbatch_path),
        "stdout": str(
            Path(
                slurm_log_root,
                f"{os.environ['SLURM_JOB_NAME']}_{os.environ['SLURM_JOB_ID']}.out",
            ).resolve()
        ),
        "stderr": str(
            Path(
                slurm_log_root,
                f"{os.environ['SLURM_JOB_NAME']}_{os.environ['SLURM_JOB_ID']}.err",
            ).resolve()
        ),
    },
    "environment": {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
    },
    "started_at_unix": time.time(),
}
manifest_path = Path(run_dir, "run_manifest.json")
temporary = manifest_path.with_name(
    f"{manifest_path.name}.tmp.{os.getpid()}"
)
data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
    "utf-8"
)
with temporary.open("xb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, manifest_path)
directory_fd = os.open(manifest_path.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

MASTER_PORT="$((20000 + (10#${SLURM_JOB_ID} % 20000)))"
set +e
"${TORCHRUN}" --nnodes=1 --nproc_per_node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --seed "${SEED}" \
  --id 0 \
  --cfg-options \
    "work_dir=${RUN_DIR}/direct_work" \
    "solver.ema=${USE_EMA}" \
    "model.backbone.custom.pretrain=${VIDEOMAE_CHECKPOINT}" \
    "inference.phystime_decode_replay_capture.weights_source=${WEIGHTS_SOURCE}" \
  2>&1 | tee "${RUN_DIR}/inference.out"
INFERENCE_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${INFERENCE_STATUS}" == "0" ]] \
  || fail "frozen direct inference failed with ${INFERENCE_STATUS}"

DIRECT_DIR="${RUN_DIR}/direct_work/gpu1_id0"
CAPTURE="${DIRECT_DIR}/decode_replay_inputs.npz"
CAPTURE_MANIFEST="${DIRECT_DIR}/decode_replay_manifest.json"
DIRECT_PRE="${DIRECT_DIR}/pre_cross_window_detections.json.gz"
DIRECT_RESULT="${DIRECT_DIR}/result_detection.json"
DIRECT_METRICS="${DIRECT_DIR}/evaluation_metrics.json"
DIRECT_AUDIT="${DIRECT_DIR}/post_processing_audit.json"
for path in \
  "${CAPTURE}" \
  "${CAPTURE_MANIFEST}" \
  "${DIRECT_PRE}" \
  "${DIRECT_RESULT}" \
  "${DIRECT_METRICS}" \
  "${DIRECT_AUDIT}"; do
  [[ -f "${path}" ]] || fail "direct inference artifact is missing: ${path}"
done

"${PYTHON}" - \
  "${RUN_DIR}/DIRECT_INFERENCE_COMPLETE" "${CAPTURE}" \
  "${CAPTURE_MANIFEST}" "${DIRECT_PRE}" "${DIRECT_RESULT}" \
  "${DIRECT_METRICS}" "${DIRECT_AUDIT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
artifacts = {}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for raw_path in sys.argv[2:]:
    path = Path(raw_path).resolve()
    artifacts[path.name] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
payload = {
    "schema_version": "phystime_decode_cross_direct_marker_v1",
    "validation_pass": True,
    "artifacts": artifacts,
}
temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

"${PYTHON}" tools/bata/replay_phystime_decode_cross.py \
  --config "${CONFIG}" \
  --artifact "${CAPTURE}" \
  --manifest "${CAPTURE_MANIFEST}" \
  --run-manifest "${RUN_DIR}/run_manifest.json" \
  --direct-pre-cross "${DIRECT_PRE}" \
  --direct-result "${DIRECT_RESULT}" \
  --direct-metrics "${DIRECT_METRICS}" \
  --checkpoint "${CHECKPOINT}" \
  --source-completion "${SOURCE_DIR}/FULL_COMPLETE.json" \
  --source-manifest "${SOURCE_DIR}/run_manifest.json" \
  --p0-completion "${P0_COMPLETION}" \
  --output-dir "${RUN_DIR}/replay" \
  --arm "${ARM}" \
  --weights-source "${WEIGHTS_SOURCE}" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-tree "${SOURCE_TREE}" \
  --expected-runtime-commit "${EXPECTED_COMMIT}" \
  --expected-runtime-tree "${EXPECTED_TREE}" \
  --evaluation-epoch "${EVALUATION_EPOCH}" \
  2>&1 | tee "${RUN_DIR}/replay.out"

"${PYTHON}" tools/bata/validate_phystime_decode_cross_replay.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/DECODE_CROSS_COMPLETE.json" \
  2>&1 | tee "${RUN_DIR}/validator.out"

"${PYTHON}" - "${RUN_DIR}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
completion = json.loads(
    (run_dir / "DECODE_CROSS_COMPLETE.json").read_text(encoding="utf-8")
)
if completion.get("validation_pass") is not True:
    raise SystemExit("production-semantic decode-cross validator did not pass")
summary = {
    "schema_version": "phystime_decode_cross_runtime_summary_v1",
    "validation_pass": True,
    "status": completion["status"],
    "arm": completion["arm"],
    "weights_source": completion["weights_source"],
    "native_axis": completion["native_axis"],
    "evaluation_epoch": completion["evaluation_epoch"],
    "new_training": False,
    "mode_metrics": completion["mode_metrics"],
    "physical_minus_uniform_percentage_points": completion[
        "physical_minus_uniform_percentage_points"
    ],
    "native_direct_exact_equivalence": completion[
        "native_direct_exact_equivalence"
    ],
}

def atomic_json(path, payload):
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

summary_path = run_dir / "runtime_summary.json"
atomic_json(summary_path, summary)
completion_path = run_dir / "DECODE_CROSS_COMPLETE.json"
atomic_json(
    run_dir / "DECODE_CROSS_VALIDATED",
    {
        "schema_version": "phystime_decode_cross_validated_marker_v1",
        "validation_pass": True,
        "completion_sha256": hashlib.sha256(
            completion_path.read_bytes()
        ).hexdigest(),
        "runtime_summary_sha256": hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest(),
    },
)
PY

echo "[PhysTime decode cross replay] COMPLETE arm=${ARM} weights=${WEIGHTS_SOURCE}"
