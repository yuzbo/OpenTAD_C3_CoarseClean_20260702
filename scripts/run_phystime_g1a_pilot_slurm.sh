#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1a pilot] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
CONFIG="${PHYSTIME_G1A_CONFIG:?PHYSTIME_G1A_CONFIG is required}"
RUN_DIR="${PHYSTIME_G1A_RUN_DIR:?PHYSTIME_G1A_RUN_DIR is required}"
GATE_JSON="${PHYSTIME_G1A_GATE_JSON:?PHYSTIME_G1A_GATE_JSON is required}"
CONTRACT_JSON="${PHYSTIME_G1A_CONTRACT_JSON:?PHYSTIME_G1A_CONTRACT_JSON is required}"
STATIC_G0_JSON="${PHYSTIME_G0_OUTPUT:?PHYSTIME_G0_OUTPUT is required}"
CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
SEED="${PHYSTIME_SEED:-42}"
PILOT_EPOCHS="${PHYSTIME_G1A_PILOT_EPOCHS:-6}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the formal G1a pilot must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
[[ "${PILOT_EPOCHS}" == "6" ]] || fail "the matched G1a pilot is fixed to exactly six epochs"
: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
[[ -x "${PYTHON}" && -x "${TORCHRUN}" ]] || fail "OpenTAD Python/torchrun environment is unavailable"
[[ -f "${CONFIG}" && -f "${GATE_JSON}" && -f "${CONTRACT_JSON}" && -f "${STATIC_G0_JSON}" && -f "${CHECKPOINT}" ]] || fail "config/gate/contract/G0/checkpoint missing"
[[ ! -e "${RUN_DIR}" ]] || fail "pilot run directory already exists: ${RUN_DIR}"

case "$(basename "${CONFIG}")" in
  phystime_g1a_selected_axis_native_j192.py|phystime_g1a_physical_metric_native_j192.py) ;;
  *) fail "G1a pilot accepts only the two native-J192 matched configs" ;;
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
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit changed after submission"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree changed after submission"

CONFIG="$(readlink -f "${CONFIG}")"
case "${CONFIG}" in
  "${WORK_DIR}/configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py"|\
  "${WORK_DIR}/configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py") ;;
  *) fail "G1a config must come from the fixed runtime snapshot" ;;
esac

"${PYTHON}" - "${GATE_JSON}" "${CONTRACT_JSON}" "${STATIC_G0_JSON}" "${COMMIT}" "${TREE}" "${CHECKPOINT}" "${CONFIG}" <<'PY'
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

gate_path, contract_path, static_g0_path, commit, tree, checkpoint, config_path = sys.argv[1:]
payload = json.loads(Path(gate_path).read_text(encoding="utf-8"))
validate_gate_report(payload)
if payload.get("git_commit") != commit:
    raise SystemExit("G1a gate commit does not match pilot snapshot")
if payload.get("git_tree") != tree:
    raise SystemExit("G1a gate tree does not match pilot snapshot")
if payload.get("checkpoint_sha256") != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
    raise SystemExit("G1a gate checkpoint does not match pilot checkpoint")
if payload.get("contract_sha256") != hashlib.sha256(Path(contract_path).read_bytes()).hexdigest():
    raise SystemExit("G1a gate contract does not match pilot contract")
if payload.get("static_g0_sha256") != hashlib.sha256(Path(static_g0_path).read_bytes()).hexdigest():
    raise SystemExit("G1a gate static G0 does not match pilot G0")
contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
variant = "selected_axis" if "selected_axis" in Path(config_path).name else "physical_metric"
cfg = Config.fromfile(config_path, lazy_import=False)
config_sha256 = _canonical_sha256(cfg.to_dict())
if contract.get("config_sha256", {}).get(variant) != config_sha256:
    raise SystemExit("G1a pilot config differs from the static contract")
if payload.get("variants", {}).get(variant, {}).get("canonical_config_sha256") != config_sha256:
    raise SystemExit("G1a pilot config differs from the real gate")
manifest, manifest_sha256 = build_dataset_manifest(cfg, payload["evaluation_ground_truth_filename"])
if manifest_sha256 != payload.get("dataset_manifest_sha256"):
    raise SystemExit("G1a pilot dataset inventory differs from the real gate")
PY

mkdir -p "${RUN_DIR}"
"${PYTHON}" - "${CONFIG}" "${CHECKPOINT}" "${GATE_JSON}" "${CONTRACT_JSON}" "${STATIC_G0_JSON}" "${RUN_DIR}" "${COMMIT}" "${TREE}" "${PILOT_EPOCHS}" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

from mmengine.config import Config
from tools.bata.run_phystime_g1a_real_gate import _canonical_sha256

config_path, checkpoint, gate_path, contract_path, static_g0_path, run_dir, commit, tree, epochs = sys.argv[1:]
cfg = Config.fromfile(config_path, lazy_import=False)
canonical_config_sha256 = _canonical_sha256(cfg.to_dict())
variant = "selected_axis" if "selected_axis" in Path(config_path).name else "physical_metric"
cfg.model.backbone.custom.pretrain = str(Path(checkpoint).resolve())
cfg.work_dir = str(Path(run_dir, "work_dir").resolve())
cfg.workflow.end_epoch = int(epochs)
cfg.workflow.val_start_epoch = 1
cfg.workflow.val_eval_interval = 1
cfg.workflow.checkpoint_interval = 1
cfg.post_processing.save_dict = True
post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
if "Interpolate" in post_types:
    raise SystemExit("G1a pilot forbids J192-to-K384 interpolation")
manifest = {
    "schema_version": "phystime_g1a_pilot_manifest_v3",
    "commit": commit,
    "git_tree": tree,
    "runtime_root": str(Path.cwd().resolve()),
    "variant": variant,
    "started_at_unix": time.time(),
    "config": str(Path(config_path).resolve()),
    "config_sha256": canonical_config_sha256,
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "gate": str(Path(gate_path).resolve()),
    "gate_sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest(),
    "contract": str(Path(contract_path).resolve()),
    "contract_sha256": hashlib.sha256(Path(contract_path).read_bytes()).hexdigest(),
    "static_g0": str(Path(static_g0_path).resolve()),
    "static_g0_sha256": hashlib.sha256(Path(static_g0_path).read_bytes()).hexdigest(),
    "K_raw_observations": 384,
    "J_native_tubelet_tokens": 192,
    "Q0_base_candidates": 192,
    "Q_total_candidates": 378,
    "feature_interpolation": False,
    "dataset_manifest_sha256": json.loads(Path(gate_path).read_text(encoding="utf-8"))[
        "dataset_manifest_sha256"
    ],
    "pilot_epochs": int(epochs),
    "warmup_epochs": int(cfg.scheduler.warmup_epoch),
    "post_warmup_epoch_present": int(epochs) > int(cfg.scheduler.warmup_epoch),
    "sampling": "deterministic_random_fixed_subsample",
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
    "workflow.checkpoint_interval=1" \
    "post_processing.save_dict=True" \
  2>&1 | tee "${RUN_DIR}/train.out"
STATUS="${PIPESTATUS[0]}"
set -e
[[ "${STATUS}" == "0" ]] || fail "pilot training failed with exit code ${STATUS}"
"${PYTHON}" tools/bata/validate_phystime_g1a_pilot_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/PILOT_COMPLETE.json"
"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
completion_path = run_dir / "PILOT_COMPLETE.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))
if completion.get("validation_pass") is not True:
    raise SystemExit("pilot artifact validation did not pass")
(run_dir / "runtime_summary.json").write_text(
    json.dumps(
        {
            "training_exit_code": 0,
            "effective_work_dir": completion["effective_work_dir"],
            "result_detection": completion["artifacts"]["predictions"]["path"],
            "evaluation_metrics": completion["artifacts"]["metrics"]["path"],
            "checkpoint": completion["artifacts"]["checkpoint"]["path"],
            "evaluation_artifacts_valid": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "[PhysTime G1a pilot] complete config=$(basename "${CONFIG}") run_dir=${RUN_DIR}"
