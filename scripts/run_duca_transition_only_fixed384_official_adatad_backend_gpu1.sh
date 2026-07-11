#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TRANSITION_ONLY][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py}"
VALIDATOR="${VALIDATOR:-tools/bata/validate_duca_transition_only_fixed384_official_adatad_backend.py}"
PROOF="${PROOF:-tools/bata/run_duca_transition_only_official_adatad_one_step_grad_proof.py}"
COST_PROFILER="${COST_PROFILER:-tools/bata/profile_duca_transition_only_cost.py}"
RUN_TAG="${RUN_TAG:-duca_transition_only_fixed384_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-30371}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
export C3_OFFICIAL_ACTION_SEG_REPOS="${C3_OFFICIAL_ACTION_SEG_REPOS:-${BASE}/projects/external_official_action_segmentation_repos_20260702}"

export DUCA_ONLINE_BUDGET=384
export DUCA_OFFICIAL_ADATAD_BUDGET=384
export DUCA_ONLINE_DENSE_WINDOW_SIZE=768
export DUCA_VALIDATOR_MAX_BUDGET=384
export DUCA_BUDGET_CURVE_MODE=0
export DUCA_OFFICIAL_ADATAD_END_EPOCH=132
export DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH=100
export DUCA_LOSS_SCHEDULE_TOTAL_STEPS=13200
export DUCA_PROFILE_RUNTIME="${DUCA_PROFILE_RUNTIME:-0}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "0" || "${CUDA_VISIBLE_DEVICES}" == "1" ]] \
    || fail "expected one Slurm-bound logical GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] \
    || fail "outside Slurm this launcher is restricted to physical GPU1"
fi

[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${CONFIG}" ]] || fail "config missing: ${CONFIG}"
[[ -f "${VALIDATOR}" ]] || fail "validator missing: ${VALIDATOR}"
[[ -f "${PROOF}" ]] || fail "one-step proof missing: ${PROOF}"
[[ -f "${COST_PROFILER}" ]] || fail "cost profiler missing: ${COST_PROFILER}"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] \
  || fail "official ASFormer source missing under ${C3_OFFICIAL_ACTION_SEG_REPOS}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true

RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_transition_only_fixed384/${RUN_TAG}}"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

echo "[DUCA_TRANSITION_ONLY] repo=${REPO_ROOT}"
echo "[DUCA_TRANSITION_ONLY] head=$(git rev-parse HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_TRANSITION_ONLY] task=offline_tad selector=transition_only dense=768 budget=384 max_hole=15"
echo "[DUCA_TRANSITION_ONLY] official_asformer_root=${C3_OFFICIAL_ACTION_SEG_REPOS}"
echo "[DUCA_TRANSITION_ONLY] precheck_only=${PRECHECK_ONLY} fulltrain_candidate=${FULLTRAIN_CANDIDATE}"

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  "${CONFIG}" \
  "${VALIDATOR}" \
  "${PROOF}" \
  "${COST_PROFILER}" \
  opentad/models/duca/transition_only.py \
  opentad/models/duca/acquisition.py \
  opentad/models/selectors/duca_online_frame_selector.py \
  tools/bata/train_lowres_action_probe.py
"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}" --output-json "${RUN_DIR}/contract_validation.json"
"${PYTHON}" -m pytest \
  tests/test_duca_transition_only.py \
  tests/test_duca_official_asformer_hidden.py \
  tests/test_duca_optimizer_exact_coverage.py \
  tests/test_duca_transition_only_fixed384_official_adatad_backend.py \
  tests/test_duca_transition_only_optimizer_groups.py \
  tests/test_duca_transition_only_p0_matrix.py \
  tests/test_duca_transition_only_cost_profile.py \
  -q

PROOF_DEVICE=cpu
if "${PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  PROOF_DEVICE=cuda
fi
"${PYTHON}" "${PROOF}" \
  --config "${CONFIG}" \
  --device "${PROOF_DEVICE}" \
  --output-json "${RUN_DIR}/one_step_grad_proof.json"
"${PYTHON}" "${COST_PROFILER}" \
  --config "${CONFIG}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --device "${PROOF_DEVICE}" \
  --temporal-len 16 \
  --budget 8 \
  --height 16 \
  --width 16 \
  --probe-spatial-size 16 \
  --warmup 1 \
  --repeats 2 \
  --output "${RUN_DIR}/cost_profile_smoke.json"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[DUCA_TRANSITION_ONLY] PRECHECK_ONLY complete"
  exit 0
fi

[[ "${FULLTRAIN_CANDIDATE}" == "1" ]] || fail "FULLTRAIN_CANDIDATE=1 is required beyond precheck"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal full train must run inside Slurm"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"

cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo nogit)",
  "config": "${CONFIG}",
  "task": "offline_temporal_action_detection",
  "selector_variant": "transition_only",
  "budget": 384,
  "dense_window_size": 768,
  "max_unselected_hole": 15,
  "expected_steps_per_epoch": 100,
  "expected_total_steps": 13200,
  "workflow_epochs": 132,
  "slurm_job_id": "${SLURM_JOB_ID}",
  "seed": ${SEED}
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
