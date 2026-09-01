#!/usr/bin/env bash
set -u

JOBS=1180674,1180685,1180686,1180687,1180696,1180697
ROOTS=(
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_cee4ccd_recovery_20260723_042025
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_mobilenet_e30db0f_retry_20260723_042025
)

date '+%F %T %z'
squeue -j "${JOBS}" -o '%.18i %.26j %.10T %.10M %.30R' || true
sacct -j "${JOBS}" --format=JobIDRaw,JobName%30,State,Elapsed,ExitCode -P | grep -E '^[0-9]+\|' || true

echo '=== ERRORS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f \( -name '*.err' -o -name '*.out' -o -name train.out \) -print0 2>/dev/null |
    xargs -0 -r grep -Hn -E 'Traceback|CUDA out of memory|out of memory|non-finite|ValueError|\[FAIL\]|AMP replay exhausted' 2>/dev/null |
    tail -30 || true
done

echo '=== PROGRESS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f -name train.out -print0 2>/dev/null | while IFS= read -r -d '' log; do
    line="$(grep -E '\[Train\]: Epoch|\[[0-9]{3}\]\[[0-9]{5}/[0-9]{5}\]|Average-mAP|mAP at tIoU|Training Over' "${log}" 2>/dev/null | tail -1)"
    [ -z "${line}" ] || printf '%s\t%s\n' "${log}" "${line}"
  done
done

echo '=== GATES ==='
for root in "${ROOTS[@]}"; do
  find "${root}" -type f \( -name '*full_model*.json' -o -name 'gate_suite.json' \) -print 2>/dev/null | sort
done
