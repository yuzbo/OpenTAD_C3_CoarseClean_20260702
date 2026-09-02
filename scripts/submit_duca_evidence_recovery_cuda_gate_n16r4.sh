#!/usr/bin/env bash
set -euo pipefail
source /etc/profile
BASE="${DUCA_N16R4_BASE:-/data/run01/sczc063/yuzibo}"
REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd)}"
test -d "${REPO_ROOT}/.git"
exec sbatch --parsable --export=ALL,DUCA_REPO_ROOT="${REPO_ROOT}",YUZIBO_ROOT="${BASE}" \
  "${REPO_ROOT}/scripts/run_duca_evidence_recovery_cuda_gate_n16r4.sbatch"
