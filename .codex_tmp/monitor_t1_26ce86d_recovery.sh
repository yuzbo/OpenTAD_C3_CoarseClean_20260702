#!/usr/bin/env bash
set -u

JOBS=1180637,1180638,1180653,1180697,1180717,1180718,1180719
ROOTS=(
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_mobilenet_e30db0f_retry_20260723_042025
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516
)

date '+%F %T %z'
squeue -j "${JOBS}" -o '%.18i %.28j %.10T %.10M %.30R' || true
sacct -j "${JOBS}" --format=JobIDRaw,JobName%32,State,Elapsed,ExitCode -P \
  | grep -E '^[0-9]+\|' || true
echo '=== NEW ERRORS ==='
for root in "${ROOTS[@]}"; do
  [[ -d "${root}" ]] || continue
  find "${root}" -type f \( -name '*.err' -o -name '*.out' -o -name train.out \) \
    -mmin -15 -print0 2>/dev/null \
    | xargs -0 -r grep -Hn -E 'Traceback|CUDA out of memory|non-finite|ValueError|\[FAIL\]|AMP replay exhausted' 2>/dev/null \
    | tail -30 || true
done
echo '=== PROGRESS ==='
for root in "${ROOTS[@]}"; do
  [[ -d "${root}" ]] || continue
  find "${root}" -type f -name train.out -mmin -15 -print0 2>/dev/null \
    | while IFS= read -r -d '' log; do
        line="$(grep -E '\[Train\]: Epoch|\[[0-9]{3}\]\[[0-9]{5}/[0-9]{5}\]|Average-mAP|mAP at tIoU|Training Over' "${log}" 2>/dev/null | tail -1)"
        [[ -z "${line}" ]] || printf '%s\t%s\n' "${log}" "${line}"
      done
done
