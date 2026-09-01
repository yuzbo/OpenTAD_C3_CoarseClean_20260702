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
SHORT_COMMIT="$(git rev-parse --short HEAD)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${N16R4_BASE}/duca_evidence_recovery_single_seed_8261_${SHORT_COMMIT}_${TIMESTAMP}"
mkdir -p "${RUN_ROOT}/slurm_logs"

PRIMARY_SEED="${1:-8261}"

echo "================================================================="
echo "DUCA Evidence Recovery 8-Arm Single-Seed Pipeline (Seed ${PRIMARY_SEED})"
echo "Nonce: DUCA-EVIDENCE-RECOVERY-SINGLE-SEED-v001-${TIMESTAMP}"
echo "Commit: $(git rev-parse HEAD)"
echo "Run Root: ${RUN_ROOT}"
echo "================================================================="

export DUCA_RUN_ROOT="${RUN_ROOT}"
export DUCA_SEEDS="${PRIMARY_SEED}"

# 1. Submit required cost profiling gate
echo ""
echo "[1/4] Submitting Cost Profiling Gate..."
COST_JOB_OUT=$(sbatch --parsable \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
  scripts/run_duca_evidence_recovery_cost_array_n16r4.sbatch)
COST_JOB_ID="${COST_JOB_OUT}"
echo "Cost Profiling Job ID: ${COST_JOB_ID}"

# 2. Submit 8-Arm Training Array (tasks 0-7, afterok:COST_JOB_ID)
echo ""
echo "[2/4] Submitting 8-Arm Training Array (Seed ${PRIMARY_SEED})..."
TRAIN_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${COST_JOB_ID}" \
  --array=0-7 \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
  scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)
TRAIN_JOB_ID="${TRAIN_JOB_OUT}"
echo "Training Array Job ID: ${TRAIN_JOB_ID}"

# 3. Submit 8-Arm Evaluation Array (tasks 0-7, afterok:TRAIN_JOB_ID)
echo ""
echo "[3/4] Submitting 8-Arm Evaluation Array..."
EVAL_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --array=0-7 \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
  scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch)
EVAL_JOB_ID="${EVAL_JOB_OUT}"
echo "Evaluation Array Job ID: ${EVAL_JOB_ID}"

# 4. Submit Statistical / Summary Analysis (afterok:EVAL_JOB_ID and COST_JOB_ID)
echo ""
echo "[4/4] Submitting Statistical Analysis & Gate Evaluator..."
STATS_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${EVAL_JOB_ID}:${COST_JOB_ID}" \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
  scripts/run_duca_evidence_recovery_stats_n16r4.sbatch)
STATS_JOB_ID="${STATS_JOB_OUT}"
echo "Statistical Analysis Job ID: ${STATS_JOB_ID}"

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
