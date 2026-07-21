#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_FRONTEND_PRETRAIN][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

VARIANT="${DUCA_FRONTEND_VARIANT:-}"
case "${VARIANT}" in
  lr_control_c25_a50_s100)
    CONFIG="configs/adatad/thumos/duca_frontend_pretrain_lr_control_c25_a50_s100.py"
    COARSE_TRUNK_LR=2.5e-5
    ACTION_HEAD_LR=5.0e-5
    TRANSITION_SCORER_LR=1.0e-4
    ;;
  lr_coarse50_action100_scorer25)
    CONFIG="configs/adatad/thumos/duca_frontend_pretrain_lr_coarse50_action100_scorer25.py"
    COARSE_TRUNK_LR=5.0e-5
    ACTION_HEAD_LR=1.0e-4
    TRANSITION_SCORER_LR=2.5e-5
    ;;
  lr_coarse100_action200_scorer50)
    CONFIG="configs/adatad/thumos/duca_frontend_pretrain_lr_coarse100_action200_scorer50.py"
    COARSE_TRUNK_LR=1.0e-4
    ACTION_HEAD_LR=2.0e-4
    TRANSITION_SCORER_LR=5.0e-5
    ;;
  gaussian_matched)
    CONFIG="configs/adatad/thumos/duca_gaussian_frontend_pretrain_matched_fixed384.py"
    COARSE_TRUNK_LR=5.0e-5
    ACTION_HEAD_LR=1.0e-4
    TRANSITION_SCORER_LR=1.0e-4
    ;;
  burst_r2q3)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py"
    COARSE_TRUNK_LR=5.0e-5
    ACTION_HEAD_LR=1.0e-4
    TRANSITION_SCORER_LR=1.0e-4
    ;;
  burst_r4q5)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py"
    COARSE_TRUNK_LR=5.0e-5
    ACTION_HEAD_LR=1.0e-4
    TRANSITION_SCORER_LR=1.0e-4
    ;;
  *)
    fail "unknown frontend learning-rate variant: ${VARIANT}"
    ;;
esac
ACTION_WEIGHT=1.0
TRANSITION_WEIGHT=0.10
if [[ "${VARIANT}" == gaussian_matched || "${VARIANT}" == burst_* ]]; then
  BOUNDARY_WEIGHT=2.0
else
  BOUNDARY_WEIGHT=16.0
fi

QUALITY_CONFIG="configs/adatad/thumos/duca_frontend_holdout_quality_fixed384.py"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_DIR="${RUN_DIR:-}"
WORK_DIR="${WORK_DIR:-}"
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${SPLIT_MANIFEST}" ]] || fail "frontend split manifest is missing"
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == "${SPLIT_SHA256}" ]] \
  || fail "frontend split manifest hash drift"
[[ -f "${DUCA_FRONTEND_TRAIN_BLOCK_LIST:-}" ]] || fail "frontend train block list is missing"
[[ -f "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST:-}" ]] || fail "frontend holdout block list is missing"
"${PYTHON}" -m tools.bata.create_duca_frontend_split \
  --validate-manifest "${SPLIT_MANIFEST}" \
  --expected-manifest-sha256 "${SPLIT_SHA256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --train-block-list "${DUCA_FRONTEND_TRAIN_BLOCK_LIST}" \
  --holdout-block-list "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}" \
  > /dev/null
[[ -n "${RUN_DIR}" && ! -e "${RUN_DIR}" ]] || fail "fresh RUN_DIR is required"
[[ -n "${WORK_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh WORK_DIR is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

mkdir -p "${RUN_DIR}/quality" "${WORK_DIR}"
"${PYTHON}" -m tools.bata.validate_duca_frontend_p0_contract \
  --config "${CONFIG}" \
  --output-json "${RUN_DIR}/p0_contract.json"
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-frontend-${SLURM_JOB_ID}-${VARIANT}" \
  tools/train.py "${CONFIG}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
candidate_files=()
for item in "5:4" "10:9" "15:14" "20:19"; do
  epoch_one="${item%%:*}"
  epoch_zero="${item##*:}"
  checkpoint="${ACTUAL_WORK_DIR}/checkpoint/epoch_${epoch_zero}.pth"
  quality_dir="${RUN_DIR}/quality/epoch_${epoch_one}"
  records="${quality_dir}/selection_quality_records.jsonl"
  export_summary="${quality_dir}/selection_quality_export.json"
  summary="${quality_dir}/selection_quality_summary.json"
  candidate="${quality_dir}/candidate.json"
  [[ -f "${checkpoint}" ]] || fail "missing checkpoint ${checkpoint}"
  mkdir -p "${quality_dir}"
  "${PYTHON}" -m tools.bata.export_duca_selection_quality \
    --config "${QUALITY_CONFIG}" \
    --selector-config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --output-jsonl "${records}" \
    --summary-json "${export_summary}" \
    --split val \
    --device cuda:0 \
    --use-ema true \
    --seed 3407 \
    2>&1 | tee "${quality_dir}/export.out"
  "${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
    --records-jsonl "${records}" \
    --output-dir "${quality_dir}" \
    --bootstrap-samples 2000 \
    --random-seed 3407 \
    2>&1 | tee "${quality_dir}/analyze.out"
  "${PYTHON}" - "${candidate}" "${VARIANT}" "${epoch_one}" \
    "${checkpoint}" "${summary}" "${records}" \
    "${ACTION_WEIGHT}" "${TRANSITION_WEIGHT}" "${BOUNDARY_WEIGHT}" \
    "${COARSE_TRUNK_LR}" "${ACTION_HEAD_LR}" "${TRANSITION_SCORER_LR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    out,
    variant,
    epoch,
    checkpoint,
    summary,
    records,
    action,
    transition,
    boundary,
    coarse_lr,
    action_lr,
    scorer_lr,
) = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "variant": variant,
    "epoch_one_based": int(epoch),
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": digest(checkpoint),
    "summary_path": str(Path(summary).resolve()),
    "summary_sha256": digest(summary),
    "records_path": str(Path(records).resolve()),
    "records_sha256": digest(records),
    "loss_weights": {
        "actionness": float(action),
        "transition": float(transition),
        "transition_boundary": float(boundary),
    },
    "component_lrs": {
        "coarse_trunk": float(coarse_lr),
        "action_head": float(action_lr),
        "transition_scorer": float(scorer_lr),
    },
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  candidate_files+=("${candidate}")
done

"${PYTHON}" - "${RUN_DIR}/completion.json" "${EXPECTED_COMMIT}" "${VARIANT}" \
  "${SPLIT_SHA256}" "${ACTION_WEIGHT}" "${TRANSITION_WEIGHT}" "${BOUNDARY_WEIGHT}" \
  "${COARSE_TRUNK_LR}" "${ACTION_HEAD_LR}" "${TRANSITION_SCORER_LR}" \
  "${RUN_DIR}/p0_contract.json" \
  "${candidate_files[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    out,
    commit,
    variant,
    split_sha,
    action,
    transition,
    boundary,
    coarse_lr,
    action_lr,
    scorer_lr,
    contract_path,
    *candidate_paths,
) = sys.argv[1:]
candidates = [json.loads(Path(path).read_text(encoding="utf-8")) for path in candidate_paths]
contract = Path(contract_path).resolve()
payload = {
    "schema": "duca_frontend_variant_completion_v1",
    "ok": True,
    "git_commit": commit,
    "variant": variant,
    "split_manifest_sha256": split_sha,
    "test_subset_consumed": False,
    "loss_weights": {
        "actionness": float(action),
        "transition": float(transition),
        "transition_boundary": float(boundary),
    },
    "component_lrs": {
        "coarse_trunk": float(coarse_lr),
        "action_head": float(action_lr),
        "transition_scorer": float(scorer_lr),
    },
    "p0_contract_path": str(contract),
    "p0_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "candidates": candidates,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[DUCA_FRONTEND_PRETRAIN] completed ${RUN_DIR}/completion.json"
