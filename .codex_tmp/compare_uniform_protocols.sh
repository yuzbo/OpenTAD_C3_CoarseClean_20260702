#!/usr/bin/env bash
set -u

OLD=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/slurm_adapter_matched_diag/adapter_uniform_gridaware_densehead-1150842.out
NATIVE=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/slurm_adapter_matched_diag/adapter_stride2_uniform_dense65-1150701.out
CURRENT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048/arms/two_stage_exact_uniform/official60/eval.out

date '+DATE %F %T %z'
for spec in "GRID_AWARE:$OLD" "NATIVE_STRIDE2:$NATIVE" "CURRENT_TERMINAL:$CURRENT"; do
  name=${spec%%:*}
  path=${spec#*:}
  echo "=== $name ==="
  stat -c 'SIZE=%s MTIME=%y PATH=%n' "$path" 2>/dev/null || true
  grep -n -E 'Epoch|Average-mAP|mAP at tIoU|Testing Over|best|Best|checkpoint' "$path" 2>/dev/null | tail -120 || true
done

exit 0
