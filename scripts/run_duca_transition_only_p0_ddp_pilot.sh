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
cat > "${RUN_ROOT}/manifest.json" <<EOF
{
  "git_commit": "${CURRENT_HEAD}",
  "core_gate_json": "${CORE_GATE_JSON}",
  "core_gate_json_sha256": "$(sha256sum "${CORE_GATE_JSON}" | awk '{print $1}')",
  "seed": ${SEED},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "task": "offline_temporal_action_detection",
  "pilot_steps_per_variant": 10,
  "activation_checkpointing": false,
  "static_graph": false,
  "find_unused_parameters": true
}
EOF

variants=(uniform direct transition_beta0 transition_counterfactual)
configs=(
  configs/adatad/thumos/duca_exact_uniform_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_direct_boundary_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_transition_only_fixed384_beta0_p0_ddp_pilot.py
  configs/adatad/thumos/duca_transition_only_fixed384_counterfactual_p0_ddp_pilot.py
)
ports=(30621 30622 30623 30624)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  config="${configs[$index]}"
  probe_json="${RUN_ROOT}/probes/${variant}.training_probe.json"
  export DUCA_TRAINING_PROBE_JSON="${probe_json}"
  echo "[DUCA_P0_DDP_PILOT] starting ${variant}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${ports[$index]}" \
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
