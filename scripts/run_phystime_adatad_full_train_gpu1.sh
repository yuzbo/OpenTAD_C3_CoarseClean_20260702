#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime-AdaTAD train] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
cd "${WORK_DIR}"

BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
CONFIG="${PHYSTIME_CONFIG:?PHYSTIME_CONFIG is required}"
RUN_DIR="${PHYSTIME_RUN_DIR:?PHYSTIME_RUN_DIR is required}"
PHYSTIME_REAL_GATE_JSON="${PHYSTIME_REAL_GATE_JSON:?PHYSTIME_REAL_GATE_JSON is required}"
PHYSTIME_VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
SEED="${PHYSTIME_SEED:-42}"
STABILITY_GATE="${PHYSTIME_STABILITY_GATE:-0}"
[[ "${STABILITY_GATE}" == "0" || "${STABILITY_GATE}" == "1" ]] || \
  fail "PHYSTIME_STABILITY_GATE must be 0 or 1"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "non-Slurm debug is restricted to physical GPU1"
  fail "formal training must run inside Slurm"
fi
export CUDA_VISIBLE_DEVICES

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"

[[ -x "${PYTHON}" ]] || fail "Python environment not found: ${PYTHON}"
[[ -x "${TORCHRUN}" ]] || fail "torchrun not found: ${TORCHRUN}"
[[ -f "${CONFIG}" ]] || fail "config not found: ${CONFIG}"
[[ -f "${PHYSTIME_REAL_GATE_JSON}" ]] || fail "real gate artifact not found"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "VideoMAE-S checkpoint not found"
[[ -f "${OPENTAD_THUMOS14_ANNOTATION}" ]] || fail "annotation file not found"
[[ -f "${OPENTAD_THUMOS14_CLASS_MAP}" ]] || fail "class map not found"
[[ -d "${OPENTAD_THUMOS14_TRAIN_VIDEOS}" ]] || fail "training video directory not found"
[[ -d "${OPENTAD_THUMOS14_TEST_VIDEOS}" ]] || fail "test video directory not found"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required for cost accounting"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi
source "${BASE}/conda_envs/opentad/bin/activate"

case "$(basename "${CONFIG}")" in
  selected_axis_adatad_sparse_k384.py|physical_grid_adatad_sparse_k384.py|phystime_adatad_sparse_k384.py) ;;
  *) fail "formal Phase 1 accepts only the three matched K384 raw-video configs" ;;
esac

while IFS='=' read -r name _value; do
  upper_name="${name^^}"
  if [[ "${upper_name}" == *FEATURE* && "${upper_name}" == *PATH* ]]; then
    fail "feature archive path variables are forbidden in raw-video formal training: ${name}"
  fi
done < <(env)

COMMIT="$(git rev-parse HEAD)"
"${PYTHON}" - "${PHYSTIME_REAL_GATE_JSON}" "${COMMIT}" "${PHYSTIME_VIDEOMAE_CHECKPOINT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.run_phystime_adatad_real_gate import validate_gate_report

gate_path, commit, checkpoint = sys.argv[1:]
payload = json.loads(Path(gate_path).read_text(encoding="utf-8"))
validate_gate_report(payload)
checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
if not payload.get("gate_pass") is True:
    raise SystemExit("real gate did not pass")
if not payload.get("git_commit") == commit:
    raise SystemExit("real gate commit does not match the training snapshot")
if not payload.get("checkpoint_sha256") == checkpoint_sha256:
    raise SystemExit("real gate checkpoint does not match formal training")
PY

mkdir -p "${RUN_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export PHYSTIME_VIDEOMAE_CHECKPOINT
export OPENTAD_THUMOS14_ANNOTATION OPENTAD_THUMOS14_CLASS_MAP
export OPENTAD_THUMOS14_TRAIN_VIDEOS OPENTAD_THUMOS14_TEST_VIDEOS

"${PYTHON}" - "${CONFIG}" "${PHYSTIME_VIDEOMAE_CHECKPOINT}" "${RUN_DIR}" "${COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from mmengine.config import Config

config_path, checkpoint, run_dir, commit = sys.argv[1:]
cfg = Config.fromfile(config_path)
cfg.model.backbone.custom.pretrain = str(Path(checkpoint).resolve())
payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
pipeline_types = {
    split: [str(step["type"]) for step in cfg.dataset[split].pipeline]
    for split in ("train", "val", "test")
}
feature_loader = "Load" + "Feats"
if any(feature_loader in types for types in pipeline_types.values()):
    raise SystemExit("formal raw-video config contains a feature loader")
if any("mmaction.DecordDecode" not in types for types in pipeline_types.values()):
    raise SystemExit("formal raw-video config is missing DecordDecode")
manifest = {
    "schema_version": "phystime_adatad_formal_run_v2",
    "commit": commit,
    "config": str(Path(config_path).resolve()),
    "resolved_config_sha256": hashlib.sha256(payload).hexdigest(),
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "sampling": "deterministic_random_fixed_subsample",
    "logical_window": 768,
    "decoded_frame_budget": 384,
    "input_source": "raw_thumos_mp4",
    "pipeline_types": pipeline_types,
    "stability_gate": bool(int(__import__("os").environ.get("PHYSTIME_STABILITY_GATE", "0"))),
}
Path(run_dir, "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

MASTER_PORT="$((20000 + (10#${SLURM_JOB_ID} % 20000)))"
export PHYSTIME_MASTER_PORT="${MASTER_PORT}"
env | grep -E '^(PHYSTIME_|OPENTAD_THUMOS14_|CUDA_VISIBLE_DEVICES=)' | sort > "${RUN_DIR}/environment.txt"
printf '%s\n' "${COMMIT}" > "${RUN_DIR}/commit.txt"

GPU_MEMORY_LOG="${RUN_DIR}/gpu_memory.tsv"
GPU_MONITOR_ID="${CUDA_VISIBLE_DEVICES%%,*}"
printf 'unix_time\tmemory_used_mb\n' > "${GPU_MEMORY_LOG}"
monitor_gpu() {
  while true; do
    value="$(nvidia-smi --id="${GPU_MONITOR_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    [[ "${value}" =~ ^[0-9]+$ ]] || value=0
    printf '%s\t%s\n' "$(date +%s)" "${value}" >> "${GPU_MEMORY_LOG}"
    sleep 5
  done
}

monitor_gpu &
MONITOR_PID=$!
cleanup_monitor() {
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

START_TIME="$(date +%s)"
set +e
CFG_OPTIONS=(
  "model.backbone.custom.pretrain=${PHYSTIME_VIDEOMAE_CHECKPOINT}"
  "work_dir=${RUN_DIR}/work_dir"
)
if [[ "${STABILITY_GATE}" == "1" ]]; then
  CFG_OPTIONS+=(
    "workflow.end_epoch=2"
    "workflow.disable_checkpoint=True"
    "workflow.val_start_epoch=2"
    "workflow.val_eval_interval=-1"
  )
fi
"${TORCHRUN}" --nnodes=1 --nproc_per_node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" \
  --seed "${SEED}" --id 0 \
  --cfg-options "${CFG_OPTIONS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.out"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
END_TIME="$(date +%s)"
cleanup_monitor
trap - EXIT

PEAK_GPU_MEMORY_MB="$(awk 'NR > 1 && $2 > max {max=$2} END {print max+0}' "${GPU_MEMORY_LOG}")"
WALL_TIME_SEC="$((END_TIME - START_TIME))"
"${PYTHON}" - "${RUN_DIR}" "${TRAIN_STATUS}" "${WALL_TIME_SEC}" "${PEAK_GPU_MEMORY_MB}" <<'PY'
import json
import sys
from pathlib import Path

run_dir, status, wall_time, peak_memory = sys.argv[1:]
payload = {
    "training_exit_code": int(status),
    "wall_time_sec": int(wall_time),
    "peak_gpu_memory_mb": int(peak_memory),
}
Path(run_dir, "runtime_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

[[ "${TRAIN_STATUS}" == "0" ]] || fail "training failed with exit code ${TRAIN_STATUS}"
if [[ "${STABILITY_GATE}" == "1" ]]; then
  touch "${RUN_DIR}/STABILITY_GATE_COMPLETE"
else
  touch "${RUN_DIR}/TRAINING_COMPLETE"
fi
echo "[PhysTime-AdaTAD train] complete config=$(basename "${CONFIG}") run_dir=${RUN_DIR}"
