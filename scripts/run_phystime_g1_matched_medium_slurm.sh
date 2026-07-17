#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1 matched medium] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
VARIANT="${PHYSTIME_MEDIUM_VARIANT:?PHYSTIME_MEDIUM_VARIANT is required}"
CONFIG="${PHYSTIME_MEDIUM_CONFIG:?PHYSTIME_MEDIUM_CONFIG is required}"
RUN_DIR="${PHYSTIME_MEDIUM_RUN_DIR:?PHYSTIME_MEDIUM_RUN_DIR is required}"
G1A_GATE="${PHYSTIME_G1A_GATE_OUTPUT:?PHYSTIME_G1A_GATE_OUTPUT is required}"
G1B_GATE="${PHYSTIME_G1B_GATE_OUTPUT:?PHYSTIME_G1B_GATE_OUTPUT is required}"
CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"
EPOCHS="${PHYSTIME_MEDIUM_EPOCHS:-20}"
SEED="${PHYSTIME_SEED:-42}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "medium training must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
[[ "${EPOCHS}" == "20" ]] || fail "matched medium contract requires exactly 20 epochs"
[[ "${SEED}" == "42" ]] || fail "matched medium contract requires seed 42"
[[ -f "${CONFIG}" && -f "${G1A_GATE}" && -f "${G1B_GATE}" && -f "${CHECKPOINT}" ]] \
  || fail "config/gate/checkpoint missing"
[[ ! -e "${RUN_DIR}" ]] || fail "run directory already exists: ${RUN_DIR}"

case "${VARIANT}|$(basename "${CONFIG}")" in
  selected_axis\|phystime_g1a_selected_axis_native_j192.py) ;;
  physical_metric\|phystime_g1a_physical_metric_native_j192.py) ;;
  g1b_sdpq\|phystime_g1b_sdpq_pool_native_j192.py) ;;
  *) fail "variant/config pair is not part of the fixed matched suite" ;;
esac

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
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit differs from submission"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree differs from submission"

CONFIG="$(readlink -f "${CONFIG}")"
case "${CONFIG}" in
  "${WORK_DIR}/configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py"|\
  "${WORK_DIR}/configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py"|\
  "${WORK_DIR}/configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py") ;;
  *) fail "config must come from the fixed runtime snapshot" ;;
esac

"${PYTHON}" - \
  "${VARIANT}" "${CONFIG}" "${CHECKPOINT}" "${G1A_GATE}" "${G1B_GATE}" \
  "${COMMIT}" "${TREE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from mmengine.config import Config

from tools.bata.run_phystime_g1a_real_gate import (
    _canonical_sha256,
    build_dataset_manifest,
    validate_gate_report,
)

variant, config_path, checkpoint, g1a_path, g1b_path, commit, tree = sys.argv[1:]
g1a = json.loads(Path(g1a_path).read_text(encoding="utf-8"))
g1b = json.loads(Path(g1b_path).read_text(encoding="utf-8"))
validate_gate_report(g1a)
if g1b.get("schema_version") != "phystime_g1b_sdpq_real_gate_v1":
    raise SystemExit("G1b gate schema mismatch")
if g1b.get("gate_pass") is not True:
    raise SystemExit("G1b gate did not pass")
for gate in (g1a, g1b):
    if gate.get("git_commit") != commit or gate.get("git_tree") != tree:
        raise SystemExit("gate does not match medium runtime snapshot")
if g1b.get("feature_interpolation") is not False:
    raise SystemExit("G1b gate did not prove no interpolation")
if int(g1b.get("gt_without_assigned_query", -1)) != 0:
    raise SystemExit("G1b gate left GT without assignment")
if int(g1b.get("short_gt_without_assigned_query", -1)) != 0:
    raise SystemExit("G1b gate left short GT without assignment")

cfg = Config.fromfile(config_path, lazy_import=False)
config_sha = _canonical_sha256(cfg.to_dict())
if variant in {"selected_axis", "physical_metric"}:
    observed = g1a.get("variants", {}).get(variant, {}).get("canonical_config_sha256")
    if observed != config_sha:
        raise SystemExit("G1a gate config differs from medium config")
else:
    gate_config = Path(g1b.get("config", ""))
    if not gate_config.is_absolute():
        gate_config = Path.cwd() / gate_config
    if gate_config.resolve() != Path(config_path).resolve():
        raise SystemExit("G1b gate config differs from medium config")
_, dataset_sha = build_dataset_manifest(cfg, g1a["evaluation_ground_truth_filename"])
if dataset_sha != g1a.get("dataset_manifest_sha256"):
    raise SystemExit("medium dataset differs from G1a gate dataset")
if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != g1a.get("checkpoint_sha256"):
    raise SystemExit("medium checkpoint differs from G1a gate checkpoint")
PY

mkdir -p "${RUN_DIR}"
"${PYTHON}" - \
  "${VARIANT}" "${CONFIG}" "${CHECKPOINT}" "${G1A_GATE}" "${G1B_GATE}" \
  "${RUN_DIR}" "${COMMIT}" "${TREE}" "${EPOCHS}" "${SEED}" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

from mmengine.config import Config

from tools.bata.run_phystime_g1a_real_gate import (
    _canonical_sha256,
    build_dataset_manifest,
)

(
    variant,
    config_path,
    checkpoint,
    g1a_path,
    g1b_path,
    run_dir,
    commit,
    tree,
    epochs,
    seed,
) = sys.argv[1:]
cfg = Config.fromfile(config_path, lazy_import=False)
post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
if "Interpolate" in post_types:
    raise SystemExit("matched medium suite forbids feature interpolation")
g1a = json.loads(Path(g1a_path).read_text(encoding="utf-8"))
_, dataset_sha = build_dataset_manifest(cfg, g1a["evaluation_ground_truth_filename"])
manifest = {
    "schema_version": "phystime_g1_matched_medium_manifest_v1",
    "variant": variant,
    "commit": commit,
    "git_tree": tree,
    "runtime_root": str(Path.cwd().resolve()),
    "config": str(Path(config_path).resolve()),
    "config_sha256": _canonical_sha256(cfg.to_dict()),
    "pretrained_checkpoint": str(Path(checkpoint).resolve()),
    "pretrained_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "g1a_gate": str(Path(g1a_path).resolve()),
    "g1a_gate_sha256": hashlib.sha256(Path(g1a_path).read_bytes()).hexdigest(),
    "g1b_gate": str(Path(g1b_path).resolve()),
    "g1b_gate_sha256": hashlib.sha256(Path(g1b_path).read_bytes()).hexdigest(),
    "dataset_manifest_sha256": dataset_sha,
    "run_dir": str(Path(run_dir).resolve()),
    "started_at_unix": time.time(),
    "epochs": int(epochs),
    "final_epoch": int(epochs) - 1,
    "seed": int(seed),
    "K_raw_observations": 384,
    "J_native_tubelet_tokens": 192,
    "feature_interpolation": False,
    "sampling": "random_fixed_subsample_k384_from_logical_768",
    "scheduler_type": str(cfg.scheduler.type),
    "scheduler_warmup_epoch": int(cfg.scheduler.warmup_epoch),
    "scheduler_max_epoch": int(cfg.scheduler.max_epoch),
    "validation_start_epoch": 1,
    "validation_interval": 1,
    "checkpoint_save_mode": "lightweight_final_only_with_ema",
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
  tools/train.py "${CONFIG}" --seed "${SEED}" --id 0 \
  --cfg-options \
    "model.backbone.custom.pretrain=${CHECKPOINT}" \
    "work_dir=${RUN_DIR}/work_dir" \
    "workflow.end_epoch=${EPOCHS}" \
    "workflow.val_start_epoch=1" \
    "workflow.val_eval_interval=1" \
    "workflow.checkpoint_interval=${EPOCHS}" \
    "workflow.checkpoint_save_mode=lightweight" \
    "workflow.checkpoint_include_ema=True" \
    "post_processing.save_dict=True" \
  2>&1 | tee "${RUN_DIR}/train.out"
STATUS="${PIPESTATUS[0]}"
set -e
[[ "${STATUS}" == "0" ]] || fail "medium training failed with exit code ${STATUS}"
touch "${RUN_DIR}/MEDIUM_TRAINING_COMPLETE"
"${PYTHON}" tools/bata/validate_phystime_g1_matched_medium_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/MEDIUM_COMPLETE.json"
"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
completion = json.loads((run_dir / "MEDIUM_COMPLETE.json").read_text(encoding="utf-8"))
if completion.get("validation_pass") is not True:
    raise SystemExit("medium artifact validation did not pass")
(run_dir / "runtime_summary.json").write_text(
    json.dumps(
        {
            "training_exit_code": 0,
            "variant": completion["variant"],
            "epochs": completion["epochs"],
            "evaluation_epoch": completion["evaluation_epoch"],
            "metrics": completion["metrics"],
            "prediction_path": completion["artifacts"]["predictions"]["path"],
            "checkpoint_path": completion["artifacts"]["checkpoint"]["path"],
            "evaluation_artifacts_valid": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "[PhysTime G1 matched medium] complete variant=${VARIANT} run_dir=${RUN_DIR}"
