#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_OFFICIAL60][FAIL] $*" >&2
  exit 1
}

external_path() {
  local label="$1"
  local requested="$2"
  local repo_real base_real target_real
  repo_real="$(realpath -e "${REPO_ROOT}")"
  base_real="$(realpath -e "${BASE}")"
  target_real="$(realpath -m "${requested}")"
  case "${target_real}/" in
    "${base_real}/"*) ;;
    *) fail "${label} must stay under BASE" ;;
  esac
  case "${target_real}/" in
    "${repo_real}/"*) fail "${label} must stay outside the worktree" ;;
  esac
  printf '%s\n' "${target_real}"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_protected_physical_env.sh"

VARIANT="${DUCA_PROTECTED_VARIANT:-}"
case "${VARIANT}" in
  exact_uniform)
    CONFIG="configs/adatad/thumos/duca_protected_physical_exact_uniform_fixed384_official60.py"
    ;;
  transition_no_bridge)
    CONFIG="configs/adatad/thumos/duca_protected_physical_transition_no_bridge_fixed384_official60.py"
    ;;
  protected_e2e)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_fixed384_official60.py"
    ;;
  protected_e2e_bridge025)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_bridge025_fixed384_official60.py"
    ;;
  protected_e2e_uni_companion)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_uni_companion_fixed384_official60.py"
    ;;
  protected_e2e_rho001)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_rho001_fixed384_official60.py"
    ;;
  *)
    fail "unknown protected physical variant"
    ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
PROTOCOL_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
PROTOCOL_SHA256="${DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256:-}"
AUTHORIZATION_JSON="${DUCA_PROTECTED_AUTHORIZATION_JSON:-}"
AUTHORIZATION_SHA256="${DUCA_PROTECTED_AUTHORIZATION_SHA256:-}"
RUN_DIR="$(external_path RUN_DIR "${RUN_DIR:-}")"
WORK_DIR="$(external_path WORK_DIR "${WORK_DIR:-}")"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one logical GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PROTOCOL_JSON}" ]] || fail "P0 manifest is missing"
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] || fail "P0 hash drift"
[[ -f "${AUTHORIZATION_JSON}" ]] || fail "P0-P3 authorization is missing"
[[ "$(sha256sum "${AUTHORIZATION_JSON}" | awk '{print $1}')" == "${AUTHORIZATION_SHA256}" ]] || fail "authorization hash drift"
[[ ! -e "${RUN_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh run and work directories are required"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

readarray -t CONFIG_BINDING < <(
  "${PYTHON}" - "${PROTOCOL_JSON}" "${VARIANT}" "${CONFIG}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
variant = sys.argv[2]
config = Path(sys.argv[3]).resolve()
record = manifest["configs"]["arms"][variant]
actual = hashlib.sha256(config.read_bytes()).hexdigest()
if actual != record["source_sha256"]:
    raise SystemExit("runtime source config differs from P0")
print(record["resolved_sha256"])
print(actual)
PY
)
[[ "${#CONFIG_BINDING[@]}" == "2" ]] || fail "cannot bind the P0 config"
export DUCA_RESOLVED_CONFIG_SHA256="${CONFIG_BINDING[0]}"
export DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
export DUCA_PROTECTED_VARIANT="${VARIANT}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON="${PROTOCOL_JSON}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256="${PROTOCOL_SHA256}"
export DUCA_PROTECTED_AUTHORIZATION_JSON="${AUTHORIZATION_JSON}"
export DUCA_PROTECTED_AUTHORIZATION_SHA256="${AUTHORIZATION_SHA256}"

cat > "${RUN_DIR}/launch_manifest.json" <<EOF
{
  "schema": "duca_protected_physical_launch_manifest_v1",
  "task": "offline_temporal_action_detection",
  "git_commit": "${EXPECTED_COMMIT}",
  "variant": "${VARIANT}",
  "seed": 3407,
  "config": "${CONFIG}",
  "config_sha256": "${CONFIG_BINDING[1]}",
  "resolved_config_sha256": "${DUCA_RESOLVED_CONFIG_SHA256}",
  "protocol_manifest_sha256": "${PROTOCOL_SHA256}",
  "authorization_sha256": "${AUTHORIZATION_SHA256}",
  "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
  "checkpoint_interval": 5,
  "slurm_job_id": "${SLURM_JOB_ID}"
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-protected-official60-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py "${CONFIG}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
TRAINING_AUDIT="${ACTUAL_WORK_DIR}/duca_protected_physical_training_audit.json"
CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_59.pth"
CHECKPOINT_SIDECAR="${CHECKPOINT}.metadata.json"
EVAL_ROOT="${RUN_DIR}/terminal_eval"
EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
POST_RUN_JSON="${RUN_DIR}/post_run_evidence.json"
[[ -f "${TRAINING_AUDIT}" ]] || fail "terminal training audit is missing"
[[ -f "${CHECKPOINT}" ]] || fail "terminal checkpoint is missing"
[[ -f "${CHECKPOINT_SIDECAR}" ]] || fail "terminal checkpoint sidecar is missing"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-protected-official60-${SLURM_JOB_ID}-${VARIANT}-eval" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 \
  --metrics-json "${EVAL_JSON}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${EVAL_ROOT}" \
    "model.backbone.custom.pretrain=${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" -m tools.bata.finalize_duca_protected_physical_run \
  --variant "${VARIANT}" \
  --protocol-manifest "${PROTOCOL_JSON}" \
  --protocol-manifest-sha256 "${PROTOCOL_SHA256}" \
  --authorization "${AUTHORIZATION_JSON}" \
  --authorization-sha256 "${AUTHORIZATION_SHA256}" \
  --training-audit "${TRAINING_AUDIT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sidecar "${CHECKPOINT_SIDECAR}" \
  --evaluation-json "${EVAL_JSON}" \
  --output-json "${POST_RUN_JSON}"
