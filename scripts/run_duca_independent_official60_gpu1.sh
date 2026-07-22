#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_INDEPENDENT_OFFICIAL60][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

VARIANT="${DUCA_INDEPENDENT_VARIANT:-}"
case "${VARIANT}" in
  two_stage_exact_uniform)
    CONFIG="configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
    P0_CONFIG=""
    ;;
  gaussian_matched_g0)
    CONFIG="configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_gaussian_frontend_pretrain_matched_fixed384.py"
    ;;
  boundary_burst_r2q3_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py"
    ;;
  boundary_burst_r2q3_soft_detached_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_boundary_burst_soft_detached_frontend_pretrain_fixed384.py"
    ;;
  boundary_burst_r2q3_hard_detached_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_boundary_burst_hard_detached_frontend_pretrain_fixed384.py"
    ;;
  boundary_burst_r2q3_soft_adapted_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384.py"
    ;;
  boundary_burst_r4q5_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
    P0_CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py"
    ;;
  *) fail "unknown independent variant: ${VARIANT}" ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
ARM_ROOT="${DUCA_INDEPENDENT_ARM_ROOT:-}"
FROZEN_PRETRAIN_PATH="${DUCA_ADATAD_PRETRAIN_PATH:-}"
FROZEN_PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "clean tree required"
[[ -n "${ARM_ROOT}" && ! -e "${ARM_ROOT}" ]] || fail "fresh ARM_ROOT is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == 1 ]] \
  || fail "exactly one Slurm-visible GPU is required"

"${PYTHON}" - "${ADATAD_PRETRAIN_PATH}" "${FROZEN_PRETRAIN_PATH}" \
  "${FROZEN_PRETRAIN_SHA256}" <<'PY'
import sys
from tools.bata.duca_selected_axis_training import validate_frozen_pretrain_binding

validate_frozen_pretrain_binding(
    runtime_path=sys.argv[1], expected_path=sys.argv[2], expected_sha256=sys.argv[3]
)
PY

mkdir -p "${ARM_ROOT}/gate/contracts" "${ARM_ROOT}/gate/full_model" \
  "${ARM_ROOT}/official60" "${ARM_ROOT}/p0"

P0_CHECKPOINT=""
P0_CHECKPOINT_SHA256=""
if [[ -n "${P0_CONFIG}" ]]; then
  EMPTY_BLOCK_LIST="${ARM_ROOT}/p0/no_blocked_training_videos.txt"
  : > "${EMPTY_BLOCK_LIST}"
  export DUCA_FRONTEND_TRAIN_BLOCK_LIST="${EMPTY_BLOCK_LIST}"
  P0_WORK="${ARM_ROOT}/p0/work"
  "${PYTHON}" -m tools.bata.validate_duca_frontend_p0_contract \
    --config "${P0_CONFIG}" --output-json "${ARM_ROOT}/p0/contract.json"
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-independent-${SLURM_JOB_ID}-${VARIANT}-p0" \
    tools/train.py "${P0_CONFIG}" --id 0 --seed 3407 --cfg-options \
      "work_dir=${P0_WORK}" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${ARM_ROOT}/p0/train.out"
  P0_CHECKPOINT="${P0_WORK}/gpu1_id0/checkpoint/epoch_19.pth"
  [[ -f "${P0_CHECKPOINT}" ]] || fail "terminal P0 epoch-19 checkpoint is missing"
  P0_CHECKPOINT_SHA256="$(sha256sum "${P0_CHECKPOINT}" | awk '{print $1}')"
  export DUCA_FRONTEND_CHECKPOINT="${P0_CHECKPOINT}"
  export DUCA_FRONTEND_CHECKPOINT_SHA256="${P0_CHECKPOINT_SHA256}"
  export DUCA_FRONTEND_CHECKPOINT_EPOCH=19
else
  unset DUCA_FRONTEND_CHECKPOINT DUCA_FRONTEND_CHECKPOINT_SHA256 \
    DUCA_FRONTEND_CHECKPOINT_EPOCH
fi

CONFIG_STEM="$(basename "${CONFIG}" .py)"
CONTRACT_JSON="${ARM_ROOT}/gate/contracts/${VARIANT}.json"
FULL_GATE_JSON="${ARM_ROOT}/gate/full_model/${CONFIG_STEM}.json"
"${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
  --config "${CONFIG}" --output-json "${CONTRACT_JSON}"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-independent-${SLURM_JOB_ID}-${VARIANT}-gate" \
  tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
  --config "${CONFIG}" --expected-commit "${EXPECTED_COMMIT}" \
  --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
  --adatad-pretrain-sha256 "${FROZEN_PRETRAIN_SHA256}" \
  --output-json "${FULL_GATE_JSON}" \
  2>&1 | tee "${ARM_ROOT}/gate/full_model_gate.out"

GATE_SUITE="${ARM_ROOT}/gate/gate_suite.json"
"${PYTHON}" - "${GATE_SUITE}" "${EXPECTED_COMMIT}" "${CONTRACT_JSON}" \
  "${FULL_GATE_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.duca_selected_axis_training import atomic_write_json

out = Path(sys.argv[1]).resolve()
artifacts = []
for value in sys.argv[3:]:
    path = Path(value).resolve()
    artifacts.append(
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
atomic_write_json(
    out,
    {
        "schema": "duca_selected_axis_optimization_gate_v1",
        "ok": True,
        "formal_training_unlocked": True,
        "task": "offline_temporal_action_detection",
        "git_commit": sys.argv[2],
        "artifacts": artifacts,
    },
)
PY
export DUCA_SELECTED_OPT_GATE_SUITE="${GATE_SUITE}"
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="$(sha256sum "${GATE_SUITE}" | awk '{print $1}')"
export DUCA_SELECTED_OPT_VARIANT="${VARIANT}"

WORK_DIR="${ARM_ROOT}/official60/work"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-independent-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py "${CONFIG}" --id 0 --seed 3407 --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${ARM_ROOT}/official60/train.out"

CHECKPOINT="${WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth"
EVALUATION_JSON="${ARM_ROOT}/official60/terminal_evaluation.json"
[[ -f "${CHECKPOINT}" ]] || fail "terminal official60 epoch-59 checkpoint is missing"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-independent-${SLURM_JOB_ID}-${VARIANT}-eval" \
  tools/test.py "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 \
  --metrics-json "${EVALUATION_JSON}" --id 0 --seed 3407 --cfg-options \
    "work_dir=${ARM_ROOT}/official60/eval" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${ARM_ROOT}/official60/eval.out"

"${PYTHON}" - "${ARM_ROOT}/completion.json" "${EXPECTED_COMMIT}" \
  "${VARIANT}" "${CONFIG}" "${CHECKPOINT}" "${EVALUATION_JSON}" \
  "${P0_CHECKPOINT}" "${P0_CHECKPOINT_SHA256}" \
  "${DUCA_SELECTED_OPT_GATE_SUITE_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.duca_selected_axis_training import atomic_write_json, canonical_sha256

(
    output, commit, variant, config, checkpoint, evaluation_path,
    p0_checkpoint, p0_sha256, gate_sha256,
) = sys.argv[1:]
evaluation_file = Path(evaluation_path).resolve()
evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
unsigned = dict(evaluation)
self_hash = unsigned.pop("evaluation_sha256", None)
if self_hash != canonical_sha256(unsigned):
    raise SystemExit("terminal evaluation self-hash mismatch")
expected = {
    "schema_version": "duca_selected_axis_terminal_evaluation_v1",
    "git_commit": commit,
    "task": "offline_temporal_action_detection",
    "variant": variant,
    "seed": 3407,
    "checkpoint_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
}
for key, value in expected.items():
    if evaluation.get(key) != value:
        raise SystemExit(f"terminal evaluation identity mismatch: {key}")
evaluation_config = evaluation.get("evaluation_config", {})
if (
    evaluation_config.get("type") != "mAP"
    or evaluation_config.get("subset") != "validation"
    or evaluation_config.get("tiou_thresholds") != [0.3, 0.4, 0.5, 0.6, 0.7]
    or evaluation_config.get("blocked_videos") is not None
):
    raise SystemExit("terminal evaluation is not full official validation mAP")
identity = evaluation.get("training_identity", {})
if identity.get("successful_optimizer_updates") != 6000:
    raise SystemExit("terminal training did not complete 6000 updates")
def digest(value):
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()
payload = {
    "schema": "duca_independent_official60_completion_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "variant": variant,
    "seed": 3407,
    "config": str(Path(config).resolve()),
    "config_sha256": digest(config),
    "p0_training_scope": "full_THUMOS_training_subset" if p0_checkpoint else None,
    "p0_fixed_terminal_epoch": 19 if p0_checkpoint else None,
    "p0_checkpoint": str(Path(p0_checkpoint).resolve()) if p0_checkpoint else None,
    "p0_checkpoint_sha256": p0_sha256 or None,
    "gate_suite_sha256": gate_sha256,
    "terminal_checkpoint": str(Path(checkpoint).resolve()),
    "terminal_checkpoint_sha256": digest(checkpoint),
    "terminal_evaluation": str(evaluation_file),
    "terminal_evaluation_sha256": digest(evaluation_file),
    "official_validation_comparable": True,
    "checkpoint_selection_used_validation": False,
    "metrics": evaluation["metrics"],
    "evaluator": evaluation["evaluator"],
    "evaluation_config": evaluation_config,
}
atomic_write_json(Path(output).resolve(), payload)
PY

echo "[DUCA_INDEPENDENT_OFFICIAL60] completed ${ARM_ROOT}/completion.json"
