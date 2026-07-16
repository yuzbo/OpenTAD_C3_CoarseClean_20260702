#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_VARIANT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
VARIANT="${DUCA_CELLCF_VARIANT:-}"
case "${VARIANT}" in
  uniform) CONFIG="configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py" ;;
  transition_beta0) CONFIG="configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py" ;;
  cellcf) CONFIG="configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py" ;;
  *) fail "variant must be uniform, transition_beta0, or cellcf" ;;
esac

SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-logs/duca_cellcf_${VARIANT}_seed${SEED}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_cellcf/${VARIANT}/seed${SEED}}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
GATE_JSON="${DUCA_CELLCF_GATE_JSON:-}"
PILOT_JSON="${DUCA_CELLCF_DDP_PILOT_JSON:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal training must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ "$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal training requires a clean tree"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] || fail "official ASFormer source is missing"
[[ -f "${GATE_JSON}" && -f "${PILOT_JSON}" ]] || fail "gate or pilot artifact is missing"
for name in \
  DUCA_CELLCF_RESOLVED_CONFIG_SHA256 DUCA_CELLCF_PROTOCOL_SHA256 \
  DUCA_CELLCF_ORDER_SHA256 DUCA_CELLCF_ANNOTATION_SHA256 \
  DUCA_CELLCF_CLASS_MAP_SHA256 DUCA_CELLCF_EVALUATION_CONFIG_SHA256 \
  DUCA_CELLCF_GATE_SHA256 DUCA_CELLCF_DDP_PILOT_SHA256; do
  [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} is missing or invalid"
done
[[ "$(sha256sum "${GATE_JSON}" | awk '{print $1}')" == "${DUCA_CELLCF_GATE_SHA256}" ]] || fail "real-loader gate hash drift"
[[ "$(sha256sum "${PILOT_JSON}" | awk '{print $1}')" == "${DUCA_CELLCF_DDP_PILOT_SHA256}" ]] || fail "DDP pilot hash drift"

mkdir -p "${RUN_DIR}" "${WORK_DIR}"
RUNTIME_ENV="${RUN_DIR}/canonical_env.tsv"
duca_cellcf_canonical_env_payload > "${RUNTIME_ENV}"
[[ "$(sha256sum "${RUNTIME_ENV}" | awk '{print $1}')" == "${DUCA_CELLCF_CANONICAL_ENV_SHA256}" ]] \
  || fail "canonical environment drift"
cmp -s "${RUNTIME_ENV}" "${DUCA_CELLCF_CANONICAL_ENV_FILE}" || fail "canonical environment payload drift"

"${PYTHON}" tools/bata/validate_duca_cellcf_fixed384.py \
  --variant "${VARIANT}" --config "${CONFIG}" \
  --output-json "${RUN_DIR}/variant_validation.json"

CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
GATE_SHA256="$(sha256sum "${GATE_JSON}" | awk '{print $1}')"
PILOT_SHA256="$(sha256sum "${PILOT_JSON}" | awk '{print $1}')"
EVAL_ROOT="${RUN_DIR}/terminal_eval"
readarray -t RUNTIME_CONFIG_HASHES < <("${PYTHON}" - "${CONFIG}" "${WORK_DIR}" \
  "${EVAL_ROOT}" "${ADATAD_PRETRAIN_PATH}" <<'PY'
import sys

from tools.bata.duca_cellcf_training import expected_runtime_config_sha256

config, train_work_dir, eval_work_dir, pretrain = sys.argv[1:]
print(expected_runtime_config_sha256(
    config,
    {"work_dir": train_work_dir, "model.backbone.custom.pretrain": pretrain},
    experiment_id=0,
    gpu_num=1,
    entrypoint="tools/train.py",
))
print(expected_runtime_config_sha256(
    config,
    {
        "work_dir": eval_work_dir,
        "model.backbone.custom.pretrain": pretrain,
        "post_processing.save_dict": True,
        "inference.load_from_raw_predictions": False,
    },
    experiment_id=0,
    gpu_num=1,
    entrypoint="tools/test.py",
))
PY
)
[[ "${#RUNTIME_CONFIG_HASHES[@]}" == "2" ]] || fail "failed to freeze runtime config hashes"
DUCA_CELLCF_RUNTIME_CONFIG_SHA256="${RUNTIME_CONFIG_HASHES[0]}"
DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256="${RUNTIME_CONFIG_HASHES[1]}"
for value in "${DUCA_CELLCF_RUNTIME_CONFIG_SHA256}" "${DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid frozen runtime config hash"
done
cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "schema": "duca_cellcf_run_manifest_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "variant": "${VARIANT}",
  "seed": ${SEED},
  "task": "offline_temporal_action_detection",
  "config": "${CONFIG}",
  "config_sha256": "${CONFIG_SHA256}",
  "resolved_config_sha256": "${DUCA_CELLCF_RESOLVED_CONFIG_SHA256}",
  "runtime_config_sha256": "${DUCA_CELLCF_RUNTIME_CONFIG_SHA256}",
  "evaluation_runtime_config_sha256": "${DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256}",
  "protocol_sha256": "${DUCA_CELLCF_PROTOCOL_SHA256}",
  "ordered_exposure_sha256": "${DUCA_CELLCF_ORDER_SHA256}",
  "real_loader_gate_sha256": "${GATE_SHA256}",
  "ddp_pilot_sha256": "${PILOT_SHA256}",
  "evaluation_annotation_sha256": "${DUCA_CELLCF_ANNOTATION_SHA256}",
  "evaluation_class_map_sha256": "${DUCA_CELLCF_CLASS_MAP_SHA256}",
  "evaluation_config_sha256": "${DUCA_CELLCF_EVALUATION_CONFIG_SHA256}",
  "expected_successful_optimizer_updates": 13200,
  "checkpoint_interval": 5,
  "terminal_checkpoint": "epoch_131.pth/state_dict_ema",
  "slurm_job_id": "${SLURM_JOB_ID}"
}
EOF

export DUCA_CELLCF_VARIANT DUCA_EXPECTED_COMMIT DUCA_CELLCF_GATE_JSON DUCA_CELLCF_DDP_PILOT_JSON
export DUCA_CELLCF_GATE_SHA256 DUCA_CELLCF_DDP_PILOT_SHA256
export DUCA_CELLCF_RESOLVED_CONFIG_SHA256 DUCA_CELLCF_PROTOCOL_SHA256 DUCA_CELLCF_ORDER_SHA256
export DUCA_CELLCF_ANNOTATION_SHA256 DUCA_CELLCF_CLASS_MAP_SHA256 DUCA_CELLCF_EVALUATION_CONFIG_SHA256
export DUCA_CELLCF_RUNTIME_CONFIG_SHA256 DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="cellcf-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py "${CONFIG}" --id 0 --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
TRAINING_AUDIT="${ACTUAL_WORK_DIR}/duca_cellcf_training_audit.json"
CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_131.pth"
SIDECAR="${CHECKPOINT}.metadata.json"
EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
[[ -f "${TRAINING_AUDIT}" && -f "${CHECKPOINT}" && -f "${SIDECAR}" ]] || fail "terminal training evidence is incomplete"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="cellcf-${SLURM_JOB_ID}-${VARIANT}-eval" \
  tools/test.py "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 131 \
  --metrics-json "${EVAL_JSON}" --id 0 --seed "${SEED}" \
  --cfg-options "work_dir=${EVAL_ROOT}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" -m tools.bata.finalize_duca_cellcf_run \
  --variant "${VARIANT}" --run-manifest "${RUN_DIR}/manifest.json" \
  --training-audit "${TRAINING_AUDIT}" --checkpoint "${CHECKPOINT}" \
  --checkpoint-sidecar "${SIDECAR}" --evaluation-json "${EVAL_JSON}" \
  --output-json "${RUN_DIR}/post_run_evidence.json"
