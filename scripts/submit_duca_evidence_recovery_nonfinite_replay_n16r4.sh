#!/usr/bin/env bash
set -euo pipefail

source /etc/profile
BASE="${DUCA_N16R4_BASE:-/data/run01/sczc063/yuzibo}"
REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd)}"
OLD_RUN_ROOT="${DUCA_OLD_RUN_ROOT:?DUCA_OLD_RUN_ROOT is required}"
PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
test -d "${REPO_ROOT}/.git"
test -d "${OLD_RUN_ROOT}"
test -f "${PRETRAIN}"

exec sbatch --parsable \
  --export=ALL,DUCA_REPO_ROOT="${REPO_ROOT}",DUCA_OLD_RUN_ROOT="${OLD_RUN_ROOT}",YUZIBO_ROOT="${BASE}",DUCA_VIDEOMAE_PRETRAIN="${PRETRAIN}" \
  "${REPO_ROOT}/scripts/run_duca_evidence_recovery_nonfinite_replay_n16r4.sbatch"
