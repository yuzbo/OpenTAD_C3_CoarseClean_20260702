#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_TRAIN][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

VARIANT="${DUCA_SELECTED_OPT_VARIANT:-}"
case "${VARIANT}" in
  two_stage_exact_uniform)
    CONFIG="configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
    ;;
  two_stage_scratch)
    CONFIG="configs/adatad/thumos/duca_two_stage_scratch_fixed384_official60.py"
    ;;
  two_stage_pretrained_joint)
    CONFIG="configs/adatad/thumos/duca_two_stage_pretrained_joint_fixed384_official60.py"
    ;;
  two_stage_pretrained_frozen)
    CONFIG="configs/adatad/thumos/duca_two_stage_pretrained_frozen_fixed384_official60.py"
    ;;
  *)
    fail "unknown two-stage variant: ${VARIANT}"
    ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
DECISION="${DUCA_FRONTEND_DECISION_JSON:-}"
DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256:-}"
GATE_SUITE="${DUCA_SELECTED_OPT_GATE_SUITE:-}"
GATE_SUITE_SHA256="${DUCA_SELECTED_OPT_GATE_SUITE_SHA256:-}"
RUN_DIR="${RUN_DIR:-}"
WORK_DIR="${WORK_DIR:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${DECISION}" ]] || fail "frontend decision is missing"
[[ "$(sha256sum "${DECISION}" | awk '{print $1}')" == "${DECISION_SHA256}" ]] \
  || fail "frontend decision hash drift"
[[ -f "${GATE_SUITE}" ]] || fail "two-stage gate suite is missing"
[[ "$(sha256sum "${GATE_SUITE}" | awk '{print $1}')" == "${GATE_SUITE_SHA256}" ]] \
  || fail "two-stage gate suite hash drift"
[[ -n "${RUN_DIR}" && ! -e "${RUN_DIR}" ]] || fail "fresh RUN_DIR is required"
[[ -n "${WORK_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh WORK_DIR is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

readarray -t winner < <("${PYTHON}" - "${DECISION}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("ok") is not True or payload.get("test_subset_consumed") is not False:
    raise SystemExit("frontend decision did not authorize training")
manifest = json.loads(Path(payload["candidate_manifest_path"]).read_text(encoding="utf-8"))
if manifest.get("git_commit") != sys.argv[2]:
    raise SystemExit("frontend decision commit mismatch")
winner = payload["winner"]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
)
export DUCA_FRONTEND_CHECKPOINT="${winner[0]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${winner[1]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${winner[2]}"
[[ -f "${DUCA_FRONTEND_CHECKPOINT}" ]] || fail "selected frontend checkpoint is missing"
[[ "$(sha256sum "${DUCA_FRONTEND_CHECKPOINT}" | awk '{print $1}')" == "${DUCA_FRONTEND_CHECKPOINT_SHA256}" ]] \
  || fail "selected frontend checkpoint hash drift"

mkdir -p "${RUN_DIR}" "${WORK_DIR}"
CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
cat > "${RUN_DIR}/launch_manifest.json" <<EOF
{
  "schema": "duca_two_stage_curriculum_launch_v1",
  "task": "offline_temporal_action_detection",
  "git_commit": "${EXPECTED_COMMIT}",
  "variant": "${VARIANT}",
  "seed": 3407,
  "config": "${CONFIG}",
  "config_sha256": "${CONFIG_SHA256}",
  "frontend_decision_sha256": "${DECISION_SHA256}",
  "frontend_checkpoint_sha256": "${DUCA_FRONTEND_CHECKPOINT_SHA256}",
  "frontend_checkpoint_epoch_zero_based": ${DUCA_FRONTEND_CHECKPOINT_EPOCH},
  "gate_suite_sha256": "${GATE_SUITE_SHA256}",
  "uniform_detector_warmup_successful_updates": 1000,
  "official_training_successful_updates": 6000,
  "detector_extra_updates": 0,
  "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
  "checkpoint_interval": 5,
  "slurm_job_id": "${SLURM_JOB_ID}"
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-two-stage-${SLURM_JOB_ID}-${VARIANT}-train" \
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
  --rdzv_id="duca-two-stage-${SLURM_JOB_ID}-${VARIANT}-eval" \
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

"${PYTHON}" - "${RUN_DIR}" "${VARIANT}" "${EXPECTED_COMMIT}" \
  "${CHECKPOINT}" "${EVAL_JSON}" "${DECISION_SHA256}" "${GATE_SUITE_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir, variant, commit, checkpoint, evaluation, decision_sha, gate_sha = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
metrics = json.loads(Path(evaluation).read_text(encoding="utf-8"))
if metrics.get("schema_version") != "duca_selected_axis_terminal_evaluation_v1":
    raise SystemExit("terminal evaluation schema mismatch")
payload = {
    "schema": "duca_two_stage_curriculum_completion_v1",
    "ok": True,
    "git_commit": commit,
    "variant": variant,
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": digest(checkpoint),
    "evaluation_path": str(Path(evaluation).resolve()),
    "evaluation_sha256": digest(evaluation),
    "metrics": metrics["metrics"],
    "frontend_decision_sha256": decision_sha,
    "gate_suite_sha256": gate_sha,
}
Path(run_dir, "completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "[DUCA_TWO_STAGE_TRAIN] completed ${RUN_DIR}/completion.json"
