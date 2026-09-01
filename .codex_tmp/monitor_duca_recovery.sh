#!/usr/bin/env bash
set -u

OLD_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220
NEW_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756

date '+%F %T %z'
squeue -j 1180674,1180685,1180686,1180687 -o '%.18i %.26j %.10T %.10M %.30R' || true
sacct -j 1180674,1180685,1180686,1180687 --format=JobIDRaw,JobName%26,State,Elapsed,ExitCode -P || true

for job_id in 1180685 1180686 1180687; do
    scontrol show job -o "${job_id}" 2>/dev/null | sed 's/ /\n/g' | grep -E '^(JobId|JobState|Reason|Dependency)=' || true
done

echo '=== R4Q5 TRAIN TAIL ==='
tail -100 "${OLD_ROOT}/arms/boundary_burst_r4q5_g0/official60/train.out" 2>/dev/null || true

echo '=== RECENT RECOVERY ARTIFACTS ==='
find "${NEW_ROOT}" -type f \
    \( -name 'formal_full_model_gate.json' -o -name 'train.out' -o -name '*.err' -o -name '*.out' \) \
    -print 2>/dev/null | sort

echo '=== ERROR SCAN ==='
grep -R -n -E 'Traceback|OOM|out of memory|non-finite|ValueError|FAIL' \
    "${OLD_ROOT}/arms/boundary_burst_r4q5_g0/official60" \
    "${OLD_ROOT}/logs/boundary_burst_r4q5_g0-1180674.err" \
    "${NEW_ROOT}" 2>/dev/null | tail -100 || true

echo '=== NEW TRAIN TAILS ==='
find "${NEW_ROOT}" -type f -name train.out -print0 2>/dev/null | while IFS= read -r -d '' train_log; do
    echo "--- ${train_log}"
    tail -60 "${train_log}"
done
