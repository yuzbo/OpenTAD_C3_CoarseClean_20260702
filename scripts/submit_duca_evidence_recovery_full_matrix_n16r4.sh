#!/usr/bin/env bash
# Submit complete DUCA Evidence Recovery 24-cell matrix DAG on N16R4 cluster.
source /etc/profile
set -euo pipefail

N16R4_BASE="${DUCA_N16R4_BASE:-/data/run01/sczc063/yuzibo}"
SHORT_COMMIT="$(git rev-parse --short HEAD)"
RUN_STAMP="${DUCA_RUN_STAMP:-$(date +%Y%m%d)}"
RUN_ROOT="${N16R4_BASE}/duca_evidence_recovery_${SHORT_COMMIT}_${RUN_STAMP}"
mkdir -p "${RUN_ROOT}"
mkdir -p slurm_logs

echo "================================================================="
echo "DUCA Evidence Recovery Full Matrix Submission (24 Tasks)"
echo "Nonce: DUCA-EVIDENCE-RECOVERY-FULL-MATRIX-v001-${RUN_STAMP}"
echo "Commit: $(git rev-parse HEAD)"
echo "Run Root: ${RUN_ROOT}"
echo "================================================================="

export DUCA_RUN_ROOT="${RUN_ROOT}"

# 1. Submit required cost profiling gate
echo "[1/4] Submitting Cost Profiling Gate..."
COST_JOB_OUT=$(sbatch --parsable \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}" \
  scripts/run_duca_evidence_recovery_cost_array_n16r4.sbatch)
COST_JOB_ID="${COST_JOB_OUT}"
echo "Cost Profiling Job ID: ${COST_JOB_ID}"

# 2. Submit 24-task Training Array (afterok:COST_JOB_ID)
echo "[2/4] Submitting 24-task Training Array..."
TRAIN_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${COST_JOB_ID}" \
  --array=0-23 \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}" \
  scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)
TRAIN_JOB_ID="${TRAIN_JOB_OUT}"
echo "Training Array Job ID: ${TRAIN_JOB_ID}"

# 3. Submit 24-task Evaluation Array (afterok:TRAIN_JOB_ID)
echo "[3/4] Submitting 24-task Evaluation Array..."
EVAL_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --array=0-23 \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}" \
  scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch)
EVAL_JOB_ID="${EVAL_JOB_OUT}"
echo "Evaluation Array Job ID: ${EVAL_JOB_ID}"

# 4. Submit Statistical Analysis (afterok:EVAL_JOB_ID and COST_JOB_ID)
echo "[4/4] Submitting Statistical Analysis..."
STATS_JOB_OUT=$(sbatch --parsable \
  --dependency="afterok:${EVAL_JOB_ID}:${COST_JOB_ID}" \
  --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}" \
  scripts/run_duca_evidence_recovery_stats_n16r4.sbatch)
STATS_JOB_ID="${STATS_JOB_OUT}"
echo "Statistical Analysis Job ID: ${STATS_JOB_ID}"


echo "================================================================="
echo "ALL DAG JOBS SUCCESSFULLY SUBMITTED TO SLURM"
echo "Cost Gate Job ID:       ${COST_JOB_ID}"
echo "Train Array Job ID:     ${TRAIN_JOB_ID}"
echo "Eval Array Job ID:      ${EVAL_JOB_ID}"
echo "Stats Job ID:           ${STATS_JOB_ID}"
echo "Run Root:               ${RUN_ROOT}"
echo "================================================================="
