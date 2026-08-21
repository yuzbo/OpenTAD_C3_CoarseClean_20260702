#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_TRUETIME_OFFICIAL60][FAIL] $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${ROOT}/scripts/duca_protected_physical_env.sh"

ROUTE_ARM="${DUCA_TRUETIME_ROUTE_ARM:-}"
case "${ROUTE_ARM}" in
  RANKPACK_K384) CONFIG="configs/adatad/thumos/duca_rankpack_k384_curriculum.py" ;;
  TRUETIME_K384) CONFIG="configs/adatad/thumos/duca_truetime_k384_curriculum.py" ;;
  *) fail "DUCA_TRUETIME_ROUTE_ARM must be RANKPACK_K384 or TRUETIME_K384" ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
PROTOCOL_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
PROTOCOL_SHA256="${DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256:-}"
AUTHORIZATION_JSON="${DUCA_PROTECTED_AUTHORIZATION_JSON:-}"
AUTHORIZATION_SHA256="${DUCA_PROTECTED_AUTHORIZATION_SHA256:-}"
RUN_DIR="${RUN_DIR:-}"
WORK_DIR="${WORK_DIR:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one logical GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PROTOCOL_JSON}" && -f "${AUTHORIZATION_JSON}" ]] || fail "PRE_RUN evidence is missing"
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] \
  || fail "protocol hash drift"
[[ "$(sha256sum "${AUTHORIZATION_JSON}" | awk '{print $1}')" == "${AUTHORIZATION_SHA256}" ]] \
  || fail "authorization hash drift"
[[ -n "${RUN_DIR}" && -n "${WORK_DIR}" ]] || fail "run/work directories are required"
[[ ! -e "${RUN_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh run/work directories are required"

readarray -t BINDING < <(
  "${PYTHON}" - "${PROTOCOL_JSON}" "${AUTHORIZATION_JSON}" "${CONFIG}" "${ROUTE_ARM}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
a=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
c=Path(sys.argv[3]).resolve(); route=sys.argv[4]
record=p['configs']['arms'][route]
source=hashlib.sha256(c.read_bytes()).hexdigest()
assert p['ok'] is True and p['route_arm'] == route
assert record['route_arm'] == route and record['source_sha256'] == source
assert a['ok'] is True and a['route_arm'] == route
assert a['config_hashes'][route] == source
print(record['resolved_sha256'])
print(source)
PY
)
[[ "${#BINDING[@]}" == 2 ]] || fail "cannot bind config/PRE_RUN evidence"

mkdir -p "${RUN_DIR}" "${WORK_DIR}"
export DUCA_RESOLVED_CONFIG_SHA256="${BINDING[0]}"
export DUCA_PROTECTED_VARIANT="protected_e2e_homotopy025"
export DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}"

cat > "${RUN_DIR}/launch_manifest.json" <<EOF
{
  "schema": "duca_truetime_curriculum_launch_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "route_arm": "${ROUTE_ARM}",
  "selector_arm": "protected_e2e_homotopy025",
  "seed": 3407,
  "config": "${CONFIG}",
  "source_config_sha256": "${BINDING[1]}",
  "resolved_config_sha256": "${BINDING[0]}",
  "protocol_manifest_sha256": "${PROTOCOL_SHA256}",
  "authorization_sha256": "${AUTHORIZATION_SHA256}",
  "checkpoint_interval_epochs": 5,
  "checkpoint_retention": "all_5_epoch_milestones_plus_terminal",
  "primary_checkpoint": "epoch_59.pth:state_dict_ema",
  "slurm_job_id": "${SLURM_JOB_ID}"
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-truetime-${SLURM_JOB_ID}-${ROUTE_ARM}-train" \
  tools/train.py "${CONFIG}" --id 0 --seed 3407 \
  --cfg-options "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_59.pth"
AUDIT="${ACTUAL_WORK_DIR}/duca_protected_physical_training_audit.json"
SIDECAR="${CHECKPOINT}.metadata.json"
EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
[[ -f "${CHECKPOINT}" && -f "${AUDIT}" && -f "${SIDECAR}" ]] \
  || fail "terminal training evidence is incomplete"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-truetime-${SLURM_JOB_ID}-${ROUTE_ARM}-eval" \
  tools/test.py "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 \
  --metrics-json "${EVAL_JSON}" --id 0 --seed 3407 \
  --cfg-options "work_dir=${RUN_DIR}/terminal_eval" \
    "model.backbone.custom.pretrain=${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
    "post_processing.save_dict=True" "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" -m tools.bata.finalize_duca_protected_physical_run \
  --variant protected_e2e_homotopy025 \
  --protocol-manifest "${PROTOCOL_JSON}" \
  --protocol-manifest-sha256 "${PROTOCOL_SHA256}" \
  --authorization "${AUTHORIZATION_JSON}" \
  --authorization-sha256 "${AUTHORIZATION_SHA256}" \
  --training-audit "${AUDIT}" --checkpoint "${CHECKPOINT}" \
  --checkpoint-sidecar "${SIDECAR}" --evaluation-json "${EVAL_JSON}" \
  --output-json "${RUN_DIR}/post_run_evidence.json"
