#!/usr/bin/env bash
set -euo pipefail

declare -A ARM_ROOTS=(
  [R2Q3]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516/arms/boundary_burst_r2q3_g0/official60
  [R4Q5]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220/arms/boundary_burst_r4q5_g0/official60
  [soft_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60
  [hard_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/hard_detached/official60
  [soft_adapted]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_adapted/official60
)

date '+DATE %F %T %z'
for name in R2Q3 R4Q5 soft_detached hard_detached soft_adapted; do
  root="${ARM_ROOTS[$name]}"
  echo "=== ${name} ==="
  echo "ROOT ${root}"
  find "${root}" -type f -path '*/checkpoint/epoch_*.pth' -printf '%f\t%s\t%p\n' 2>/dev/null |
    awk -F '\t' '$1 ~ /^epoch_(9|19|29|39|49|59)\.pth$/' |
    sort -V || true
  echo '-- config and launch evidence --'
  find "${root}" -maxdepth 3 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) -printf '%p\n' 2>/dev/null | sort | head -40
done

echo '=== TEST ENTRY SURFACES ==='
for repo in \
  /data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_ca40c9c_20260723 \
  /data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_487a178_20260723 \
  /data/run01/sczc063/yuzibo/projects/opentad_duca_t1_26ce86d_20260723; do
  [ -d "${repo}" ] || continue
  echo "--- ${repo}"
  grep -R -n -E 'tools/test.py|state_dict_ema|epoch_59|Average-mAP' \
    "${repo}/scripts" "${repo}/tools/bata" 2>/dev/null | head -100 || true
done
