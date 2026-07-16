#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1b SDPQ pilot] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
CONFIG="${PHYSTIME_G1B_CONFIG:-configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py}"
RUN_DIR="${PHYSTIME_G1B_RUN_DIR:?PHYSTIME_G1B_RUN_DIR is required}"
GATE_JSON="${PHYSTIME_G1B_GATE_OUTPUT:?PHYSTIME_G1B_GATE_OUTPUT is required}"
CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
SEED="${PHYSTIME_SEED:-42}"
PILOT_EPOCHS="${PHYSTIME_G1B_PILOT_EPOCHS:-20}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the G1b SDPQ pilot must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
(( PILOT_EPOCHS >= 6 )) || fail "G1b SDPQ run requires at least six epochs"
[[ -f "${GATE_JSON}" && -f "${CHECKPOINT}" && -f "${CONFIG}" ]] || fail "gate/config/checkpoint missing"
[[ ! -e "${RUN_DIR}" ]] || fail "pilot run directory already exists: ${RUN_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is not clean"
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit changed after submission"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree changed after submission"

"${PYTHON}" - "${GATE_JSON}" "${COMMIT}" "${TREE}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("gate_pass") is not True:
    raise SystemExit("G1b SDPQ pilot requires a passing gate")
if gate.get("git_commit") != sys.argv[2] or gate.get("git_tree") != sys.argv[3]:
    raise SystemExit("G1b SDPQ gate does not match pilot snapshot")
if gate.get("feature_interpolation") is not False:
    raise SystemExit("G1b SDPQ gate did not prove native no-interpolation")
if int(gate.get("gt_without_assigned_query", -1)) != 0:
    raise SystemExit("G1b SDPQ gate left GT without assigned query")
PY

mkdir -p "${RUN_DIR}"
"${PYTHON}" - "${CONFIG}" "${CHECKPOINT}" "${RUN_DIR}" "${COMMIT}" "${TREE}" "${PILOT_EPOCHS}" <<'PY'
import json
import sys
import time
from pathlib import Path

from mmengine.config import Config

config, checkpoint, run_dir, commit, tree, epochs = sys.argv[1:]
cfg = Config.fromfile(config, lazy_import=False)
post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
if "Interpolate" in post_types:
    raise SystemExit("G1b SDPQ pilot forbids feature interpolation")
manifest = {
    "schema_version": "phystime_g1b_sdpq_pilot_manifest_v1",
    "commit": commit,
    "git_tree": tree,
    "config": str(Path(config).resolve()),
    "checkpoint": str(Path(checkpoint).resolve()),
    "run_dir": str(Path(run_dir).resolve()),
    "started_at_unix": time.time(),
    "pilot_epochs": int(epochs),
    "K_raw_observations": 384,
    "J_native_tubelet_tokens": 192,
    "head": "SupportDecoupledPhysicalQueryHead",
    "feature_interpolation": False,
}
Path(run_dir, "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

MASTER_PORT="$((20000 + (10#${SLURM_JOB_ID} % 20000)))"
set +e
"${TORCHRUN}" --nnodes=1 --nproc_per_node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" --seed "${SEED}" --id 0 \
  --cfg-options \
    "model.backbone.custom.pretrain=${CHECKPOINT}" \
    "work_dir=${RUN_DIR}/work_dir" \
    "workflow.end_epoch=${PILOT_EPOCHS}" \
    "workflow.val_start_epoch=1" \
    "workflow.val_eval_interval=1" \
    "workflow.checkpoint_interval=${PILOT_EPOCHS}" \
    "workflow.checkpoint_save_mode=lightweight" \
    "workflow.checkpoint_include_ema=False" \
    "post_processing.save_dict=True" \
  2>&1 | tee "${RUN_DIR}/train.out"
STATUS="${PIPESTATUS[0]}"
set -e
[[ "${STATUS}" == "0" ]] || fail "pilot training failed with exit code ${STATUS}"
touch "${RUN_DIR}/PILOT_TRAINING_COMPLETE"
"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
pilot_epochs = int(manifest["pilot_epochs"])
final_epoch = pilot_epochs - 1
metrics_files = list(run_dir.glob("work_dir/**/evaluation_metrics.json"))
checkpoint_files = list(run_dir.glob(f"work_dir/**/checkpoint/epoch_{final_epoch}.pth"))
if len(metrics_files) != 1:
    raise SystemExit(f"expected exactly one evaluation_metrics.json, got {metrics_files}")
if len(checkpoint_files) != 1:
    raise SystemExit(f"expected exactly one final epoch_{final_epoch}.pth, got {checkpoint_files}")
metrics_path = metrics_files[0]
checkpoint_path = checkpoint_files[0]
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
required = ["average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"]
missing = [key for key in required if key not in metrics]
if missing:
    raise SystemExit(f"evaluation metrics missing required keys: {missing}")
values = {key: float(metrics[key]) for key in required}
if not all(math.isfinite(value) for value in values.values()):
    raise SystemExit("evaluation metrics contain non-finite values")
evaluation_epoch = int(metrics.get("evaluation_epoch", -1))
if evaluation_epoch != final_epoch:
    raise SystemExit(f"expected final evaluation_epoch={final_epoch}, got {evaluation_epoch}")
if checkpoint_path.stat().st_size <= 0:
    raise SystemExit(f"final checkpoint is empty: {checkpoint_path}")
payload = {
    "schema_version": "phystime_g1b_sdpq_pilot_complete_v1",
    "validation_pass": True,
    "run_dir": str(run_dir),
    "training_complete": True,
    "metrics": values,
    "observed_average_mAP": [values["average_mAP"]],
    "best_average_mAP": values["average_mAP"],
    "evaluation_epoch": evaluation_epoch,
    "pilot_epochs": pilot_epochs,
    "metrics_path": str(metrics_path.resolve()),
    "checkpoint_path": str(checkpoint_path.resolve()),
    "checkpoint_size_bytes": checkpoint_path.stat().st_size,
}
(run_dir / "PILOT_COMPLETE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
echo "[PhysTime G1b SDPQ pilot] complete run_dir=${RUN_DIR}"
