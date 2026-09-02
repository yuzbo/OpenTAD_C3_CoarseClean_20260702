#!/usr/bin/env bash
# =============================================================================
# DUCA Evidence Recovery 8-Arm Single-Seed (Seed 8261) Deployment Pipeline
# =============================================================================
# Deploys 8-Arm full matrix on single seed 8261 on N16R4 cluster with Slurm DAG.
# 1. Cost profiling gate
# 2. 8-Arm Training Array (--array=0-7)
# 3. 8-Arm Evaluation Array (--array=0-7)
# 4. Statistical Analysis & Decision Gate Output
# =============================================================================

source /etc/profile
set -euo pipefail

N16R4_BASE="${DUCA_N16R4_BASE:-/data/run01/sczc063/yuzibo}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${N16R4_BASE}}"
export DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-${YUZIBO_ROOT}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
export DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800}"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/test/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_LEDGER_SOURCE="${DUCA_H65_LEDGER_SOURCE:-}"
export DUCA_H65_LEDGER_CONFIG_HASH="${DUCA_H65_LEDGER_CONFIG_HASH:-}"
REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"

[[ "$(git branch --show-current)" == "codex/duca-evidence-recovery-numerical-correction-20260902" ]] || {
  echo "Evidence formal submit requires the numerical-correction branch" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "Evidence checkout is not clean" >&2; exit 2; }

if [[ ! -f "${DUCA_VIDEOMAE_PRETRAIN}" ]]; then
  echo "[ERROR] VideoMAE checkpoint not found: ${DUCA_VIDEOMAE_PRETRAIN}" >&2
  exit 2
fi
for ledger in "${DUCA_H65_TRAIN_LEDGER_PATH}" "${DUCA_H65_VAL_LEDGER_PATH}" "${DUCA_H65_TEST_LEDGER_PATH}"; do
  if [[ ! -f "${ledger}" ]]; then
    echo "[ERROR] H65 replay ledger not found: ${ledger}" >&2
    exit 2
  fi
done
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[PRECHECK] repo=${REPO_ROOT} checkpoint=${DUCA_VIDEOMAE_PRETRAIN}"
  exit 0
fi

SHORT_COMMIT="$(git rev-parse --short HEAD)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${N16R4_BASE}/duca_evidence_recovery_single_seed_8261_${SHORT_COMMIT}_${TIMESTAMP}"
mkdir -p "${RUN_ROOT}/slurm_logs"
DAG_MANIFEST="${RUN_ROOT}/dag_state.env"

PRIMARY_SEED="${1:-8261}"

echo "================================================================="
echo "DUCA Evidence Recovery 8-Arm Single-Seed Pipeline (Seed ${PRIMARY_SEED})"
echo "Nonce: DUCA-EVIDENCE-RECOVERY-SINGLE-SEED-v001-${TIMESTAMP}"
echo "Commit: $(git rev-parse HEAD)"
echo "Run Root: ${RUN_ROOT}"
echo "================================================================="

export DUCA_RUN_ROOT="${RUN_ROOT}"
export DUCA_SEEDS="${PRIMARY_SEED}"

COST_JOB_ID=""
TRAIN_JOB_ID=""
EVAL_JOB_ID=""
STATS_JOB_ID=""

cancel_if_active() {
  local job_id="$1"
  [[ -n "${job_id}" ]] || return 0
  if squeue -h -j "${job_id}" 2>/dev/null | grep -q .; then
    scancel "${job_id}" || true
  fi
}

cleanup_submitted_jobs() {
  local status=$?
  trap - ERR
  cancel_if_active "${STATS_JOB_ID}"
  cancel_if_active "${EVAL_JOB_ID}"
  cancel_if_active "${TRAIN_JOB_ID}"
  cancel_if_active "${COST_JOB_ID}"
  echo "[ERROR] DUCA single-seed DAG failed (status=${status}); active submitted jobs were cancelled." >&2
  exit "${status}"
}
trap cleanup_submitted_jobs ERR

submit_job() {
  local label="$1"
  shift
  local output
  if ! output="$(sbatch --parsable "$@" 2>&1)"; then
    echo "[ERROR] ${label} submission failed: ${output}" >&2
    return 1
  fi
  printf '%s' "${output}"
}

write_manifest() {
  local state="$1"
  {
    printf 'STATE=%q\n' "${state}"
    printf 'RUN_ROOT=%q\n' "${RUN_ROOT}"
    printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
    printf 'YUZIBO_ROOT=%q\n' "${YUZIBO_ROOT}"
    printf 'DUCA_VIDEOMAE_PRETRAIN=%q\n' "${DUCA_VIDEOMAE_PRETRAIN}"
    printf 'DUCA_H65_LEDGER_ROOT=%q\n' "${DUCA_H65_LEDGER_ROOT}"
    printf 'DUCA_H65_TRAIN_LEDGER_PATH=%q\n' "${DUCA_H65_TRAIN_LEDGER_PATH}"
    printf 'DUCA_H65_VAL_LEDGER_PATH=%q\n' "${DUCA_H65_VAL_LEDGER_PATH}"
    printf 'DUCA_H65_TEST_LEDGER_PATH=%q\n' "${DUCA_H65_TEST_LEDGER_PATH}"
    printf 'DUCA_SEEDS=%q\n' "${PRIMARY_SEED}"
    printf 'ARRAY_MAX=%q\n' "7"
    printf 'COST_JOB_ID=%q\n' "${COST_JOB_ID}"
    printf 'TRAIN_JOB_ID=%q\n' "${TRAIN_JOB_ID}"
    printf 'EVAL_JOB_ID=%q\n' "${EVAL_JOB_ID}"
    printf 'STATS_JOB_ID=%q\n' "${STATS_JOB_ID}"
  } > "${DAG_MANIFEST}"
}

is_qos_limit() {
  [[ "$1" == *AssocMaxSubmitJobLimit* || "$1" == *QOS* || "$1" == *job.submit.limit* ]]
}

# 1. Submit required cost profiling gate
echo ""
echo "[1/4] Submitting Cost Profiling Gate..."
COST_JOB_OUT=$(submit_job "Cost profiling" \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}",DUCA_REPO_ROOT="${REPO_ROOT}",YUZIBO_ROOT="${YUZIBO_ROOT}",DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}",DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT}",DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH}",DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH}",DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH}" \
  scripts/run_duca_evidence_recovery_cost_array_n16r4.sbatch)
COST_JOB_ID="${COST_JOB_OUT}"
echo "Cost Profiling Job ID: ${COST_JOB_ID}"

# 2. Submit 8-Arm Training Array (tasks 0-7, afterok:COST_JOB_ID)
echo ""
echo "[2/4] Submitting 8-Arm Training Array (Seed ${PRIMARY_SEED})..."
TRAIN_JOB_OUT=$(submit_job "Training array" \
  --dependency="afterok:${COST_JOB_ID}" \
  --array=0-7 \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}",DUCA_REPO_ROOT="${REPO_ROOT}",YUZIBO_ROOT="${YUZIBO_ROOT}",DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}",DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT}",DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH}",DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH}",DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH}" \
  scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)
TRAIN_JOB_ID="${TRAIN_JOB_OUT}"
echo "Training Array Job ID: ${TRAIN_JOB_ID}"
write_manifest "TRAIN_SUBMITTED"

# 3. Submit 8-Arm Evaluation Array (tasks 0-7, afterok:TRAIN_JOB_ID)
echo ""
echo "[3/4] Submitting 8-Arm Evaluation Array..."
if EVAL_JOB_OUT="$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --array=0-7 \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}",DUCA_REPO_ROOT="${REPO_ROOT}",YUZIBO_ROOT="${YUZIBO_ROOT}",DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}",DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT}",DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH}",DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH}",DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH}" \
  scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch 2>&1)"; then
  EVAL_JOB_ID="${EVAL_JOB_OUT}"
  echo "Evaluation Array Job ID: ${EVAL_JOB_ID}"
  write_manifest "EVAL_SUBMITTED"
else
  if is_qos_limit "${EVAL_JOB_OUT}"; then
    write_manifest "EVAL_DEFERRED"
    echo "[DEFERRED] Evaluation array deferred by Slurm QOS; manifest=${DAG_MANIFEST}"
    exit 0
  fi
  echo "[ERROR] Evaluation array submission failed: ${EVAL_JOB_OUT}" >&2
  false
fi

# 4. Submit Statistical / Summary Analysis (afterok:EVAL_JOB_ID and COST_JOB_ID)
echo ""
echo "[4/4] Submitting Statistical Analysis & Gate Evaluator..."
if STATS_JOB_OUT="$(sbatch --parsable \
  --dependency="afterok:${EVAL_JOB_ID}:${COST_JOB_ID}" \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}",DUCA_REPO_ROOT="${REPO_ROOT}",YUZIBO_ROOT="${YUZIBO_ROOT}",DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}",DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT}",DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH}",DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH}",DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH}" \
  scripts/run_duca_evidence_recovery_stats_n16r4.sbatch 2>&1)"; then
  STATS_JOB_ID="${STATS_JOB_OUT}"
  echo "Statistical Analysis Job ID: ${STATS_JOB_ID}"
  write_manifest "SUBMITTED"
else
  if is_qos_limit "${STATS_JOB_OUT}"; then
    write_manifest "STATS_DEFERRED"
    echo "[DEFERRED] Statistics job deferred by Slurm QOS; manifest=${DAG_MANIFEST}"
    exit 0
  fi
  echo "[ERROR] Statistics submission failed: ${STATS_JOB_OUT}" >&2
  false
fi

echo ""
echo "================================================================="
echo "8-ARM SINGLE-SEED DAG SUBMITTED TO SLURM CLUSTER"
echo "  1. Cost Gate:       ${COST_JOB_ID}"
echo "  2. Train Array (8): ${TRAIN_JOB_ID}"
echo "  3. Eval Array (8):  ${EVAL_JOB_ID}"
echo "  4. Stats & Gates:   ${STATS_JOB_ID}"
echo "  Run Root:           ${RUN_ROOT}"
echo "================================================================="
echo "Monitor with:"
echo "  squeue -u sczc063 -j ${COST_JOB_ID},${TRAIN_JOB_ID},${EVAL_JOB_ID},${STATS_JOB_ID}"
echo "================================================================="
