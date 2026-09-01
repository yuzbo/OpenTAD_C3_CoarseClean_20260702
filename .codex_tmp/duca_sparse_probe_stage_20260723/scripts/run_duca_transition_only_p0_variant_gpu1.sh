#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_VARIANT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${DUCA_P0_VARIANT:-}"
case "${VARIANT}" in
  uniform) CONFIG="configs/adatad/thumos/duca_exact_uniform_fixed384_official_adatad_backend_full_train.py" ;;
  direct) CONFIG="configs/adatad/thumos/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py" ;;
  transition_beta0) CONFIG="configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py" ;;
  transition_counterfactual) CONFIG="configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py" ;;
  *) fail "DUCA_P0_VARIANT must be uniform, direct, transition_beta0, or transition_counterfactual" ;;
esac

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"
VALIDATOR="tools/bata/validate_duca_transition_only_p0_variant.py"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
DUCA_CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"
DUCA_DDP_PILOT_JSON="${DUCA_DDP_PILOT_JSON:-}"
DUCA_RESOLVED_CONFIG_SHA256="${DUCA_RESOLVED_CONFIG_SHA256:-}"
DUCA_VARIANT_CONTRACT_SHA256="${DUCA_VARIANT_CONTRACT_SHA256:-}"
DUCA_SHARED_PROTOCOL_SHA256="${DUCA_SHARED_PROTOCOL_SHA256:-}"
DUCA_CANONICAL_ENV_FILE="${DUCA_CANONICAL_ENV_FILE:-}"
DUCA_CANONICAL_ENV_SHA256="${DUCA_CANONICAL_ENV_SHA256:-}"
DUCA_EVALUATION_ANNOTATION_PATH="${DUCA_EVALUATION_ANNOTATION_PATH:-}"
DUCA_EVALUATION_ANNOTATION_SHA256="${DUCA_EVALUATION_ANNOTATION_SHA256:-}"
DUCA_EVALUATION_CLASS_MAP_PATH="${DUCA_EVALUATION_CLASS_MAP_PATH:-}"
DUCA_EVALUATION_CLASS_MAP_SHA256="${DUCA_EVALUATION_CLASS_MAP_SHA256:-}"
DUCA_EVALUATION_CONFIG_SHA256="${DUCA_EVALUATION_CONFIG_SHA256:-}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-0}"
RUN_TAG="${RUN_TAG:-duca_p0_${VARIANT}_seed${SEED}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_p0/${VARIANT}/seed${SEED}/${RUN_TAG}}"
[[ "${FULLTRAIN_CANDIDATE}" == "1" ]] || fail "FULLTRAIN_CANDIDATE=1 is required"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "P0 full train must run inside Slurm"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"
SOURCE_PATH="${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py"
REFERENCE_CONFIG="configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py"
[[ -f "${SOURCE_PATH}" ]] || fail "official ASFormer source missing"
[[ -f "${REFERENCE_CONFIG}" ]] || fail "transition-only reference config missing"
[[ -n "${DUCA_CORE_GATE_JSON}" ]] || fail "DUCA_CORE_GATE_JSON is required"
[[ -f "${DUCA_CORE_GATE_JSON}" ]] || fail "DUCA core gate JSON missing: ${DUCA_CORE_GATE_JSON}"
[[ -n "${DUCA_DDP_PILOT_JSON}" ]] || fail "DUCA_DDP_PILOT_JSON is required"
[[ -f "${DUCA_DDP_PILOT_JSON}" ]] || fail "DUCA DDP pilot JSON missing: ${DUCA_DDP_PILOT_JSON}"
[[ "${DUCA_RESOLVED_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "resolved config SHA256 is required"
[[ "${DUCA_VARIANT_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "variant contract SHA256 is required"
[[ "${DUCA_SHARED_PROTOCOL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "shared protocol SHA256 is required"
[[ -f "${DUCA_CANONICAL_ENV_FILE}" ]] || fail "canonical environment file is required"
[[ "${DUCA_CANONICAL_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "canonical environment SHA256 is required"
[[ -f "${DUCA_EVALUATION_ANNOTATION_PATH}" ]] || fail "evaluation annotation file is required"
[[ "${DUCA_EVALUATION_ANNOTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "evaluation annotation SHA256 is required"
[[ -f "${DUCA_EVALUATION_CLASS_MAP_PATH}" ]] || fail "evaluation class-map file is required"
[[ "${DUCA_EVALUATION_CLASS_MAP_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "evaluation class-map SHA256 is required"
[[ "${DUCA_EVALUATION_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "evaluation config SHA256 is required"

CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current git HEAD"
DUCA_EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
[[ "${DUCA_EXPECTED_COMMIT}" == "${CURRENT_HEAD}" ]] \
  || fail "DUCA_EXPECTED_COMMIT must match current HEAD ${CURRENT_HEAD}"
GIT_STATUS="$(git status --porcelain --untracked-files=normal)" || fail "cannot inspect git tree"
[[ -z "${GIT_STATUS}" ]] || fail "formal P0 full train requires a clean git tree"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose an allocated GPU"
VISIBLE_GPU_COUNT="$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPU_COUNT}" == "1" ]] || fail "P0 job requires exactly one Slurm-visible GPU; got ${VISIBLE_GPU_COUNT}"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

RUNTIME_CANONICAL_ENV="${RUN_DIR}/canonical_env.tsv"
duca_p0_canonical_env_payload > "${RUNTIME_CANONICAL_ENV}"
RUNTIME_CANONICAL_ENV_SHA256="$(sha256sum "${RUNTIME_CANONICAL_ENV}" | awk '{print $1}')"
PREPARED_CANONICAL_ENV_SHA256="$(sha256sum "${DUCA_CANONICAL_ENV_FILE}" | awk '{print $1}')"
[[ "${RUNTIME_CANONICAL_ENV_SHA256}" == "${DUCA_CANONICAL_ENV_SHA256}" ]] \
  || fail "runtime canonical environment hash drift"
[[ "${PREPARED_CANONICAL_ENV_SHA256}" == "${DUCA_CANONICAL_ENV_SHA256}" ]] \
  || fail "prepared canonical environment hash drift"
cmp -s "${DUCA_CANONICAL_ENV_FILE}" "${RUNTIME_CANONICAL_ENV}" \
  || fail "runtime canonical environment differs from preparation"

REFERENCE_CONFIG_SHA256="$(sha256sum "${REFERENCE_CONFIG}" | awk '{print $1}')"
SOURCE_SHA256="$(sha256sum "${SOURCE_PATH}" | awk '{print $1}')"
CHECKPOINT_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"

"${PYTHON}" "${VALIDATOR}" \
  --variant "${VARIANT}" \
  --config "${CONFIG}" \
  --core-gate-json "${DUCA_CORE_GATE_JSON}" \
  --expected-commit "${DUCA_EXPECTED_COMMIT}" \
  --expected-config-sha256 "${REFERENCE_CONFIG_SHA256}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --output-json "${RUN_DIR}/variant_validation.json"
"${PYTHON}" -m pytest tests/test_duca_transition_only_p0_matrix.py -q

RUNTIME_SUITE_MANIFEST="${RUN_DIR}/runtime_suite_manifest.json"
"${PYTHON}" -m tools.bata.validate_duca_transition_only_p0_suite \
  --repo-root "${REPO_ROOT}" \
  --seed "${SEED}" \
  --expected-commit "${DUCA_EXPECTED_COMMIT}" \
  --require-clean \
  --core-gate-json "${DUCA_CORE_GATE_JSON}" \
  --ddp-pilot-json "${DUCA_DDP_PILOT_JSON}" \
  --require-ddp-pilot \
  --output-json "${RUNTIME_SUITE_MANIFEST}" >/dev/null
"${PYTHON}" - "${RUNTIME_SUITE_MANIFEST}" "${VARIANT}" \
  "${DUCA_RESOLVED_CONFIG_SHA256}" "${DUCA_VARIANT_CONTRACT_SHA256}" \
  "${DUCA_SHARED_PROTOCOL_SHA256}" <<'PY'
import json
import sys

manifest_path, variant_name, expected_resolved, expected_contract, expected_shared = sys.argv[1:]
payload = json.load(open(manifest_path, encoding="utf-8"))
variant = next(item for item in payload["variants"] if item["name"] == variant_name)
checks = {
    "resolved config": (variant["resolved_config_sha256"], expected_resolved),
    "variant contract": (variant["variant_contract_sha256"], expected_contract),
    "shared protocol": (payload["shared_protocol_sha256"], expected_shared),
}
for label, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"runtime {label} hash drift: expected {expected}, got {actual}")
PY

CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
CORE_GATE_SHA256="$(sha256sum "${DUCA_CORE_GATE_JSON}" | awk '{print $1}')"
DDP_PILOT_SHA256="$(sha256sum "${DUCA_DDP_PILOT_JSON}" | awk '{print $1}')"

export DUCA_P0_VARIANT DUCA_EXPECTED_COMMIT DUCA_CORE_GATE_JSON DUCA_DDP_PILOT_JSON
export DUCA_RESOLVED_CONFIG_SHA256 DUCA_VARIANT_CONTRACT_SHA256 DUCA_SHARED_PROTOCOL_SHA256
export DUCA_CANONICAL_ENV_FILE DUCA_CANONICAL_ENV_SHA256
export DUCA_EVALUATION_ANNOTATION_PATH DUCA_EVALUATION_ANNOTATION_SHA256
export DUCA_EVALUATION_CLASS_MAP_PATH DUCA_EVALUATION_CLASS_MAP_SHA256
export DUCA_EVALUATION_CONFIG_SHA256

cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "git_commit": "${CURRENT_HEAD}",
  "variant": "${VARIANT}",
  "config": "${CONFIG}",
  "config_sha256": "${CONFIG_SHA256}",
  "resolved_config_sha256": "${DUCA_RESOLVED_CONFIG_SHA256}",
  "variant_contract_sha256": "${DUCA_VARIANT_CONTRACT_SHA256}",
  "shared_protocol_sha256": "${DUCA_SHARED_PROTOCOL_SHA256}",
  "canonical_env_file": "${DUCA_CANONICAL_ENV_FILE}",
  "canonical_env_sha256": "${DUCA_CANONICAL_ENV_SHA256}",
  "evaluation_annotation_path": "${DUCA_EVALUATION_ANNOTATION_PATH}",
  "evaluation_annotation_sha256": "${DUCA_EVALUATION_ANNOTATION_SHA256}",
  "evaluation_class_map_path": "${DUCA_EVALUATION_CLASS_MAP_PATH}",
  "evaluation_class_map_sha256": "${DUCA_EVALUATION_CLASS_MAP_SHA256}",
  "evaluation_config_sha256": "${DUCA_EVALUATION_CONFIG_SHA256}",
  "source": "${SOURCE_PATH}",
  "source_sha256": "${SOURCE_SHA256}",
  "checkpoint": "${ADATAD_PRETRAIN_PATH}",
  "checkpoint_sha256": "${CHECKPOINT_SHA256}",
  "core_gate_json": "${DUCA_CORE_GATE_JSON}",
  "core_gate_json_sha256": "${CORE_GATE_SHA256}",
  "core_gate_git_commit": "${DUCA_EXPECTED_COMMIT}",
  "ddp_pilot_json": "${DUCA_DDP_PILOT_JSON}",
  "ddp_pilot_json_sha256": "${DDP_PILOT_SHA256}",
  "seed": ${SEED},
  "task": "offline_temporal_action_detection",
  "budget": 384,
  "dense_window_size": 768,
  "expected_optimizer_steps": 13200,
  "slurm_job_id": "${SLURM_JOB_ID}",
  "core_gate_dependency": "precompleted_hash_bound_artifact"
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-p0-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py \
  "${CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id${RUN_ID}"
TRAINING_AUDIT="${ACTUAL_WORK_DIR}/duca_p0_training_audit.json"
TERMINAL_CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_131.pth"
CHECKPOINT_SIDECAR="${TERMINAL_CHECKPOINT}.metadata.json"
TERMINAL_EVAL_ROOT="${RUN_DIR}/terminal_eval"
TERMINAL_EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
POST_RUN_EVIDENCE="${RUN_DIR}/post_run_evidence.json"
[[ -f "${TRAINING_AUDIT}" ]] || fail "terminal training audit is missing"
[[ -f "${TERMINAL_CHECKPOINT}" ]] || fail "terminal epoch-131 checkpoint is missing"
[[ -f "${CHECKPOINT_SIDECAR}" ]] || fail "terminal checkpoint sidecar is missing"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-p0-${SLURM_JOB_ID}-${VARIANT}-terminal-eval" \
  tools/test.py \
  "${CONFIG}" \
  --checkpoint "${TERMINAL_CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 131 \
  --metrics-json "${TERMINAL_EVAL_JSON}" \
  --id 0 \
  --seed "${SEED}" \
  --cfg-options "work_dir=${TERMINAL_EVAL_ROOT}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" -m tools.bata.finalize_duca_transition_only_p0_run \
  --variant "${VARIANT}" \
  --run-manifest "${RUN_DIR}/manifest.json" \
  --training-audit "${TRAINING_AUDIT}" \
  --checkpoint "${TERMINAL_CHECKPOINT}" \
  --checkpoint-sidecar "${CHECKPOINT_SIDECAR}" \
  --evaluation-json "${TERMINAL_EVAL_JSON}" \
  --output-json "${POST_RUN_EVIDENCE}"
