#!/usr/bin/env bash
set -euo pipefail
RUN=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357
echo '=== R0 SBATCH INPUTS ==='
grep -n -E 'R0_CHECKPOINT|CHECKPOINT_EPOCH|ADATAD_PRETRAIN|run_duca_r0' \
  "${RUN}/submission/r0.sbatch" || true
echo '=== MANIFEST-LIKE FILES ==='
find "${RUN}" -maxdepth 2 -type f \( -name '*manifest*.json' -o -name '*deployment*.json' \) -print
echo '=== PRETRAIN ==='
find /data/run01/sczc063/yuzibo/pretrained -maxdepth 1 -type f \
  -name 'vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth' \
  -printf '%p|%s\n'
echo '=== CANDIDATE R0 CHECKPOINT ==='
grep -R -h -E "DUCA_R0_CHECKPOINT=|R0_CHECKPOINT='" \
  "${RUN}/submission" 2>/dev/null | head -5 || true
