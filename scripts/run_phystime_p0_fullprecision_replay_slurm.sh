#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime P0 full-precision replay] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"
SOURCE_COMMIT="${PHYSTIME_SOURCE_COMMIT:?PHYSTIME_SOURCE_COMMIT is required}"
SOURCE_TREE="${PHYSTIME_SOURCE_TREE:?PHYSTIME_SOURCE_TREE is required}"
ARM="${PHYSTIME_P0_ARM:?PHYSTIME_P0_ARM is required}"
WEIGHTS_SOURCE="${PHYSTIME_P0_WEIGHTS_SOURCE:?PHYSTIME_P0_WEIGHTS_SOURCE is required}"
CONFIG="${PHYSTIME_P0_CONFIG:?PHYSTIME_P0_CONFIG is required}"
RUN_DIR="${PHYSTIME_P0_RUN_DIR:?PHYSTIME_P0_RUN_DIR is required}"
SOURCE_DIR="${PHYSTIME_P0_SOURCE_DIR:?PHYSTIME_P0_SOURCE_DIR is required}"
CHECKPOINT="${PHYSTIME_P0_CHECKPOINT:?PHYSTIME_P0_CHECKPOINT is required}"
VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
GATE="${PHYSTIME_P0_GATE_OUTPUT:?PHYSTIME_P0_GATE_OUTPUT is required}"
SEED="${PHYSTIME_SEED:-42}"
EVALUATION_EPOCH="${PHYSTIME_EVALUATION_EPOCH:-59}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "replay must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
[[ "${SEED}" == "42" ]] || fail "P0 replay requires seed 42"
[[ "${EVALUATION_EPOCH}" == "59" ]] || fail "P0 replay requires epoch 59"
[[ -f "${CONFIG}" && -f "${CHECKPOINT}" && -f "${VIDEOMAE_CHECKPOINT}" \
    && -f "${GATE}" ]] \
  || fail "config/checkpoint/VideoMAE checkpoint/gate is missing"
[[ -f "${SOURCE_DIR}/FULL_COMPLETE.json" ]] \
  || fail "source FULL_COMPLETE.json is missing"
[[ -f "${SOURCE_DIR}/run_manifest.json" ]] \
  || fail "source run_manifest.json is missing"
[[ ! -e "${RUN_DIR}" ]] || fail "run directory already exists: ${RUN_DIR}"

case "${ARM}|${WEIGHTS_SOURCE}|$(basename "${CONFIG}")" in
  selected_axis\|online\|phystime_g1a_selected_axis_native_j192_p0_replay.py) ;;
  selected_axis\|ema\|phystime_g1a_selected_axis_native_j192_p0_replay.py) ;;
  physical_metric\|online\|phystime_g1a_physical_metric_native_j192_p0_replay.py) ;;
  physical_metric\|ema\|phystime_g1a_physical_metric_native_j192_p0_replay.py) ;;
  *) fail "arm/weights/config combination is outside the fixed P0 suite" ;;
esac

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
  PHYSTIME_ENV_INIT_MODE="module_cuda11.8_miniforge3_24.11"
else
  PHYSTIME_ENV_INIT_MODE="fixed_conda_path_no_module_command"
fi
export PHYSTIME_ENV_INIT_MODE
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit mismatch"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

"${PYTHON}" - \
  "${GATE}" "${COMMIT}" "${TREE}" "${ARM}" "${CHECKPOINT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

gate_path, commit, tree, arm, checkpoint = sys.argv[1:]
gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
if gate.get("gate_pass") is not True:
    raise SystemExit("P0 gate did not pass")
if gate["runtime"]["commit"] != commit or gate["runtime"]["git_tree"] != tree:
    raise SystemExit("P0 gate runtime snapshot mismatch")
source = gate["source_full60"]["arms"][arm]
digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
if digest != source["checkpoint_sha256"]:
    raise SystemExit("checkpoint differs from P0 gate")
PY

mkdir -p "${RUN_DIR}"
USE_EMA="False"
if [[ "${WEIGHTS_SOURCE}" == "ema" ]]; then
  USE_EMA="True"
fi

"${PYTHON}" - \
  "${RUN_DIR}" "${CONFIG}" "${GATE}" "${SOURCE_DIR}" "${CHECKPOINT}" \
  "${VIDEOMAE_CHECKPOINT}" \
  "${ARM}" "${WEIGHTS_SOURCE}" "${USE_EMA}" "${COMMIT}" "${TREE}" \
  "${SOURCE_COMMIT}" "${SOURCE_TREE}" "${SEED}" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from mmengine.config import Config

(
    run_dir,
    config_path,
    gate_path,
    source_dir,
    checkpoint,
    videomae_checkpoint,
    arm,
    weights_source,
    use_ema,
    runtime_commit,
    runtime_tree,
    source_commit,
    source_tree,
    seed,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

cfg = Config.fromfile(config_path, lazy_import=False)
cfg.merge_from_dict(
    {
        "work_dir": str(Path(run_dir, "direct_work").resolve()),
        "solver.ema": use_ema == "True",
        "model.backbone.custom.pretrain": str(
            Path(videomae_checkpoint).resolve()
        ),
    }
)
canonical = json.dumps(
    cfg.to_dict(),
    sort_keys=True,
    separators=(",", ":"),
    default=lambda value: value.item(),
).encode("utf-8")
manifest = {
    "schema_version": "phystime_p0_inference_manifest_v1",
    "track": "p0_fullprecision_nms_frozen_epoch59",
    "arm": arm,
    "weights_source": weights_source,
    "runtime_commit": runtime_commit,
    "runtime_tree": runtime_tree,
    "source_commit": source_commit,
    "source_tree": source_tree,
    "config": str(Path(config_path).resolve()),
    "effective_config_sha256": hashlib.sha256(canonical).hexdigest(),
    "gate": str(Path(gate_path).resolve()),
    "gate_sha256": sha256_file(gate_path),
    "source_dir": str(Path(source_dir).resolve()),
    "source_completion": str(Path(source_dir, "FULL_COMPLETE.json").resolve()),
    "source_completion_sha256": sha256_file(
        Path(source_dir, "FULL_COMPLETE.json")
    ),
    "source_manifest": str(Path(source_dir, "run_manifest.json").resolve()),
    "source_manifest_sha256": sha256_file(
        Path(source_dir, "run_manifest.json")
    ),
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": sha256_file(checkpoint),
    "videomae_checkpoint": str(Path(videomae_checkpoint).resolve()),
    "videomae_checkpoint_sha256": sha256_file(videomae_checkpoint),
    "evaluation_epoch": 59,
    "seed": int(seed),
    "new_training": False,
    "frozen_checkpoint_replay": True,
    "direct_policy": "fullprecision_filtered",
    "environment": {
        "init_mode": os.environ.get("PHYSTIME_ENV_INIT_MODE"),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
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
Path(run_dir, "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
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
  2>&1 | tee "${RUN_DIR}/inference.out"
INFERENCE_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${INFERENCE_STATUS}" == "0" ]] \
  || fail "direct full-precision inference failed with ${INFERENCE_STATUS}"
touch "${RUN_DIR}/DIRECT_INFERENCE_COMPLETE"

DIRECT_DIR="${RUN_DIR}/direct_work/gpu1_id0"
PRE_CROSS="${DIRECT_DIR}/pre_cross_window_detections.json.gz"
DIRECT_RESULT="${DIRECT_DIR}/result_detection.json"
DIRECT_METRICS="${DIRECT_DIR}/evaluation_metrics.json"
DIRECT_AUDIT="${DIRECT_DIR}/post_processing_audit.json"
[[ -f "${PRE_CROSS}" && -f "${DIRECT_RESULT}" && -f "${DIRECT_METRICS}" \
    && -f "${DIRECT_AUDIT}" ]] \
  || fail "direct inference did not produce the complete P0 artifact set"

"${PYTHON}" tools/bata/replay_phystime_p0_fullprecision_nms.py \
  --config "${CONFIG}" \
  --pre-cross-window "${PRE_CROSS}" \
  --direct-result "${DIRECT_RESULT}" \
  --direct-metrics "${DIRECT_METRICS}" \
  --checkpoint "${CHECKPOINT}" \
  --source-completion "${SOURCE_DIR}/FULL_COMPLETE.json" \
  --source-manifest "${SOURCE_DIR}/run_manifest.json" \
  --output-dir "${RUN_DIR}/replay" \
  --arm "${ARM}" \
  --weights-source "${WEIGHTS_SOURCE}" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-tree "${SOURCE_TREE}" \
  --expected-runtime-commit "${COMMIT}" \
  --expected-runtime-tree "${TREE}" \
  --evaluation-epoch "${EVALUATION_EPOCH}" \
  2>&1 | tee "${RUN_DIR}/replay.out"

"${PYTHON}" tools/bata/validate_phystime_p0_fullprecision_replay.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/P0_COMPLETE.json" \
  2>&1 | tee "${RUN_DIR}/validator.out"

"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
completion = json.loads(
    (run_dir / "P0_COMPLETE.json").read_text(encoding="utf-8")
)
if completion.get("validation_pass") is not True:
    raise SystemExit("P0 artifact validator did not pass")
(run_dir / "runtime_summary.json").write_text(
    json.dumps(
        {
            "validation_pass": True,
            "arm": completion["arm"],
            "weights_source": completion["weights_source"],
            "evaluation_epoch": completion["evaluation_epoch"],
            "new_training": False,
            "direct_fullprecision_filtered_metrics": completion[
                "direct_fullprecision_filtered_metrics"
            ],
            "mode_metrics": completion["mode_metrics"],
            "delta_report": completion["delta_report"],
            "direct_audit_aggregate": completion["direct_audit_aggregate"],
            "source_legacy_ema_equivalence": completion[
                "source_legacy_ema_equivalence"
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "[PhysTime P0 full-precision replay] COMPLETE arm=${ARM} weights=${WEIGHTS_SOURCE}"
