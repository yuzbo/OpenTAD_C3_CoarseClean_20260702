#!/usr/bin/env bash
set -u

declare -A LOGS=(
  [R2Q3]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516/arms/boundary_burst_r2q3_g0/official60/train.out
  [R4Q5]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220/arms/boundary_burst_r4q5_g0/official60/train.out
  [soft_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60/train.out
  [hard_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/hard_detached/official60/train.out
  [soft_adapted]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_adapted/official60/train.out
)

date '+DATE %F %T %z'
for name in R2Q3 R4Q5 soft_detached hard_detached soft_adapted; do
  log=${LOGS[$name]}
  root=${log%/train.out}
  echo "=== $name ==="
  grep -E '\[Train\]: Epoch' "$log" 2>/dev/null | tail -1 || true
  grep -E '\[[0-9]{3}\]\[[0-9]{5}/[0-9]{5}\].*Loss=' "$log" 2>/dev/null | tail -1 || true
  echo '-- map lines --'
  grep -n -E 'Average-mAP|mAP at tIoU|Testing Over' "$log" "$root"/eval.out 2>/dev/null | tail -25 || true
  echo '-- checkpoints/results --'
  find "$root" -maxdepth 5 -type f \( -name 'epoch_*.pth' -o -name '*terminal*.json' -o -name 'result_detection.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -15
done

exit 0
