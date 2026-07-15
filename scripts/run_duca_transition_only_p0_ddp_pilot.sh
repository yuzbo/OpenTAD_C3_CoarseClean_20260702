#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_DDP_PILOT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"
CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_p0_ddp_pilot_${CURRENT_HEAD:0:7}}"
SEED="${SEED:-0}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "DDP pilot must run inside Slurm"
[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "DDP pilot requires a clean git tree"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"
[[ -n "${CORE_GATE_JSON}" && -f "${CORE_GATE_JSON}" ]] || fail "DUCA_CORE_GATE_JSON is missing"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose an allocated GPU"
VISIBLE_GPU_COUNT="$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPU_COUNT}" == "1" ]] || fail "pilot requires exactly one Slurm-visible GPU"

mkdir -p "${RUN_ROOT}/probes" "${RUN_ROOT}/work_dirs" "${RUN_ROOT}/logs"
"${PYTHON}" -m tools.bata.validate_duca_transition_only_p0_suite \
  --repo-root "${REPO_ROOT}" \
  --seed "${SEED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --core-gate-json "${CORE_GATE_JSON}" \
  --output-json "${RUN_ROOT}/formal_protocol.json" \
  > "${RUN_ROOT}/formal_protocol.out"
SHARED_PROTOCOL_SHA256="$(${PYTHON} -c "import json; print(json.load(open('${RUN_ROOT}/formal_protocol.json'))['shared_protocol_sha256'])")"
PILOT_NONCE="${SLURM_JOB_ID}-${CURRENT_HEAD}-$(date +%s%N)"
CANONICAL_ENV_FILE="${RUN_ROOT}/canonical_env.tsv"
REFERENCE_CONFIG="${REPO_ROOT}/configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py"
duca_p0_canonical_env_payload > "${CANONICAL_ENV_FILE}"
cat > "${RUN_ROOT}/manifest.json" <<EOF
{
  "schema_version": "duca_p0_ddp_pilot_run_v1",
  "git_commit": "${CURRENT_HEAD}",
  "core_gate_json": "${CORE_GATE_JSON}",
  "core_gate_json_sha256": "$(sha256sum "${CORE_GATE_JSON}" | awk '{print $1}')",
  "shared_protocol_sha256": "${SHARED_PROTOCOL_SHA256}",
  "pilot_nonce": "${PILOT_NONCE}",
  "seed": ${SEED},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "checkpoint_path": "${ADATAD_PRETRAIN_PATH}",
  "checkpoint_sha256": "$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')",
  "official_asformer_source": "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py",
  "official_asformer_source_sha256": "$(sha256sum "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" | awk '{print $1}')",
  "reference_config_path": "${REFERENCE_CONFIG}",
  "reference_config_sha256": "$(sha256sum "${REFERENCE_CONFIG}" | awk '{print $1}')",
  "canonical_env_path": "${CANONICAL_ENV_FILE}",
  "canonical_env_sha256": "$(sha256sum "${CANONICAL_ENV_FILE}" | awk '{print $1}')",
  "task": "offline_temporal_action_detection",
  "pilot_steps_per_variant": 10,
  "activation_checkpointing": false,
  "static_graph": false,
  "find_unused_parameters": true
}
EOF
RUN_MANIFEST_SHA256="$(sha256sum "${RUN_ROOT}/manifest.json" | awk '{print $1}')"

variants=(uniform direct transition_beta0 transition_counterfactual)
configs=(
  configs/adatad/thumos/duca_exact_uniform_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_direct_boundary_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_transition_only_fixed384_beta0_p0_ddp_pilot.py
  configs/adatad/thumos/duca_transition_only_fixed384_counterfactual_p0_ddp_pilot.py
)
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  config="${configs[$index]}"
  probe_json="${RUN_ROOT}/probes/${variant}.training_probe.json"
  context_json="${RUN_ROOT}/probes/${variant}.context.json"
  export DUCA_TRAINING_PROBE_JSON="${probe_json}"
  cat > "${context_json}" <<EOF
{
  "schema_version": "duca_p0_ddp_pilot_context_v1",
  "git_commit": "${CURRENT_HEAD}",
  "variant": "${variant}",
  "seed": ${SEED},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "pilot_nonce": "${PILOT_NONCE}",
  "source_config_path": "${REPO_ROOT}/${config}",
  "source_config_sha256": "$(sha256sum "${config}" | awk '{print $1}')",
  "training_probe_json": "${probe_json}",
  "context_json": "${context_json}",
  "work_dir": "${RUN_ROOT}/work_dirs/${variant}",
  "checkpoint_path": "${ADATAD_PRETRAIN_PATH}",
  "checkpoint_sha256": "$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')",
  "core_gate_json_sha256": "$(sha256sum "${CORE_GATE_JSON}" | awk '{print $1}')",
  "shared_protocol_sha256": "${SHARED_PROTOCOL_SHA256}",
  "run_manifest_path": "${RUN_ROOT}/manifest.json",
  "run_manifest_sha256": "${RUN_MANIFEST_SHA256}"
}
EOF
  export DUCA_TRAINING_PROBE_CONTEXT_JSON="${context_json}"
  echo "[DUCA_P0_DDP_PILOT] starting ${variant}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-p0-${SLURM_JOB_ID}-${variant}-pilot" \
    tools/train.py \
    "${config}" \
    --id "${index}" \
    --seed "${SEED}" \
    --cfg-options \
      "work_dir=${RUN_ROOT}/work_dirs/${variant}" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${RUN_ROOT}/logs/${variant}.train.out"
  [[ -s "${probe_json}" ]] || fail "${variant} did not emit a non-empty training probe"
done

"${PYTHON}" -m tools.bata.validate_duca_transition_only_p0_ddp_pilot \
  --repo-root "${REPO_ROOT}" \
  --probe-dir "${RUN_ROOT}/probes" \
  --core-gate-json "${CORE_GATE_JSON}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --output-json "${RUN_ROOT}/ddp_pilot_suite.json" \
  > "${RUN_ROOT}/ddp_pilot_validation.out"

"${PYTHON}" - "${RUN_ROOT}/ddp_pilot_suite.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("ok") is not True:
    raise SystemExit("DDP pilot suite did not declare ok=true")
PY

echo "[DUCA_P0_DDP_PILOT] PASS ${RUN_ROOT}/ddp_pilot_suite.json"
