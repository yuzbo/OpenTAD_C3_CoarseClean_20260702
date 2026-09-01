#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_SELECTED_OPT_TRAIN][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

VARIANT="${DUCA_SELECTED_OPT_VARIANT:-}"
case "${VARIANT}" in
  exact_uniform)
    CONFIG="configs/adatad/thumos/duca_exact_uniform_fixed384_official60.py"
    ;;
  direct025)
    CONFIG="configs/adatad/thumos/duca_protected_e2e_direct025_fixed384_official60.py"
    ;;
  homotopy025)
    CONFIG="configs/adatad/thumos/duca_protected_e2e_homotopy025_fixed384_official60.py"
    ;;
  homotopy_uni_companion025)
    CONFIG="configs/adatad/thumos/duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py"
    ;;
  *)
    fail "unknown selected-axis optimization variant"
    ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
GATE_SUITE="${DUCA_SELECTED_OPT_GATE_SUITE:-}"
GATE_SUITE_SHA256="${DUCA_SELECTED_OPT_GATE_SUITE_SHA256:-}"
RUN_DIR="${RUN_DIR:-}"
WORK_DIR="${WORK_DIR:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${GATE_SUITE}" ]] || fail "selected-axis gate suite is missing"
[[ "$(sha256sum "${GATE_SUITE}" | awk '{print $1}')" == "${GATE_SUITE_SHA256}" ]] \
  || fail "selected-axis gate suite hash drift"
[[ -n "${RUN_DIR}" && ! -e "${RUN_DIR}" ]] || fail "fresh RUN_DIR is required"
[[ -n "${WORK_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh WORK_DIR is required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

mkdir -p "${RUN_DIR}" "${WORK_DIR}"
CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
cat > "${RUN_DIR}/launch_manifest.json" <<EOF
{
  "schema": "duca_selected_axis_optimization_launch_v1",
  "task": "offline_temporal_action_detection",
  "git_commit": "${EXPECTED_COMMIT}",
  "variant": "${VARIANT}",
  "seed": 3407,
  "config": "${CONFIG}",
  "config_sha256": "${CONFIG_SHA256}",
  "gate_suite_sha256": "${GATE_SUITE_SHA256}",
  "training_profile": "official60",
  "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
  "checkpoint_interval": 5,
  "slurm_job_id": "${SLURM_JOB_ID}"
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-selected-opt-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py "${CONFIG}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_59.pth"
EVAL_ROOT="${RUN_DIR}/terminal_eval"
EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
[[ -f "${CHECKPOINT}" ]] || fail "terminal epoch_59 checkpoint is missing"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-selected-opt-${SLURM_JOB_ID}-${VARIANT}-eval" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 \
  --metrics-json "${EVAL_JSON}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${EVAL_ROOT}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" - "${RUN_DIR}" "${CHECKPOINT}" "${EVAL_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
evaluation = Path(sys.argv[3]).resolve()
payload = {
    "schema": "duca_selected_axis_optimization_completion_v1",
    "ok": True,
    "status": "terminal_epoch_59_ema_evaluated",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "evaluation": str(evaluation),
    "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
}
(run_dir / "completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[DUCA_SELECTED_OPT_TRAIN] completed: ${RUN_DIR}/completion.json"
