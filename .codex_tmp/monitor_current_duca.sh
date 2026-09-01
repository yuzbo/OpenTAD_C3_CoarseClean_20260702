#!/usr/bin/env bash
set -u

JOBS=1180493,1180494,1180495,1180496,1180637,1180638,1180653,1180674,1180685,1180686,1180687,1180696,1180697,1180717,1180718,1180719,1180731,1180732
ROOTS=(
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_cee4ccd_recovery_20260723_042025
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_mobilenet_e30db0f_retry_20260723_042025
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_919aa55_recovery_20260723_053508
)

date '+%F %T %z'
echo '=== QUEUE ==='
squeue -j "${JOBS}" -o '%.18i %.28j %.10T %.10M %.34R' || true
echo '=== ACCOUNTING ==='
sacct -j "${JOBS}" --format=JobIDRaw,JobName%32,State,Elapsed,ExitCode -P | grep -E '^[0-9]+\|' || true

echo '=== NEW ERRORS (35 MIN) ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f \( -name '*.err' -o -name '*.out' -o -name train.out \) -mmin -35 -print0 2>/dev/null |
    xargs -0 -r grep -Hn -E 'Traceback|CUDA out of memory|out of memory|non-finite|ValueError|\[FAIL\]|AMP replay exhausted|DependencyNeverSatisfied' 2>/dev/null |
    tail -25 || true
done

echo '=== CURRENT PROGRESS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f -name train.out -mmin -35 -print0 2>/dev/null | while IFS= read -r -d '' log; do
    line="$(grep -E '\[Train\]: Epoch|\[[0-9]{3}\]\[[0-9]{5}/[0-9]{5}\]|Average-mAP|mAP at tIoU|Training Over|successful optimizer' "${log}" 2>/dev/null | tail -1)"
    [ -z "${line}" ] || printf '%s\t%s\n' "${log}" "${line}"
  done
done

echo '=== RECENT RESULT ARTIFACTS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f \( -name '*summary*.json' -o -name '*completion*.json' -o -name '*final_results*.json' -o -name '*terminal*.json' \) -mmin -35 -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -20
done
