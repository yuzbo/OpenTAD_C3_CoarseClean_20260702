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
  transition_beta025) CONFIG="configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py" ;;
  *) fail "DUCA_P0_VARIANT must be uniform, direct, transition_beta0, or transition_beta025" ;;
esac

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
VALIDATOR="tools/bata/validate_duca_transition_only_p0_variant.py"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
DUCA_CORE_GATE_PASSED="${DUCA_CORE_GATE_PASSED:-0}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-0}"
MASTER_PORT="${MASTER_PORT:-30471}"
RUN_TAG="${RUN_TAG:-duca_p0_${VARIANT}_seed${SEED}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_p0/${VARIANT}/seed${SEED}/${RUN_TAG}}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
export C3_OFFICIAL_ACTION_SEG_REPOS="${C3_OFFICIAL_ACTION_SEG_REPOS:-${BASE}/projects/external_official_action_segmentation_repos_20260702}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export DUCA_ONLINE_BUDGET=384
export DUCA_OFFICIAL_ADATAD_BUDGET=384
export DUCA_ONLINE_DENSE_WINDOW_SIZE=768
export DUCA_VALIDATOR_MAX_BUDGET=384
export DUCA_BUDGET_CURVE_MODE=0
export DUCA_OFFICIAL_ADATAD_END_EPOCH=132
export DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH=100
export DUCA_LOSS_SCHEDULE_TOTAL_STEPS=13200
export DUCA_PROFILE_RUNTIME="${DUCA_PROFILE_RUNTIME:-0}"

[[ "${DUCA_CORE_GATE_PASSED}" == "1" ]] || fail "DUCA_CORE_GATE_PASSED=1 is required"
[[ "${FULLTRAIN_CANDIDATE}" == "1" ]] || fail "FULLTRAIN_CANDIDATE=1 is required"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "P0 full train must run inside Slurm"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] || fail "official ASFormer source missing"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "${CUDA_VISIBLE_DEVICES}" == "0" || "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "invalid Slurm GPU mapping"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

"${PYTHON}" "${VALIDATOR}" --variant "${VARIANT}" --config "${CONFIG}" --output-json "${RUN_DIR}/variant_validation.json"
"${PYTHON}" -m pytest tests/test_duca_transition_only_p0_matrix.py -q

cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo nogit)",
  "variant": "${VARIANT}",
  "config": "${CONFIG}",
  "seed": ${SEED},
  "task": "offline_temporal_action_detection",
  "budget": 384,
  "dense_window_size": 768,
  "expected_optimizer_steps": 13200,
  "slurm_job_id": "${SLURM_JOB_ID}",
  "core_gate_dependency_asserted": true
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"
