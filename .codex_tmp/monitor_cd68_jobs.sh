#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506
JOBS=1180336,1180337,1180338,1180339,1180340,1180341
ERROR_PATTERN='Traceback|OutOfMemory|CUDA out of memory|ValueError|RuntimeError|non-finite loss|nonfinite loss|FAIL|5920 vs 1184'
PROGRESS_PATTERN='Training Starts|Epoch 0 started|Epoch: 0, Step:|Average-mAP|selected_count|max_hole|optimizer'

date '+%F %T %z'
echo QUEUE
squeue -j "$JOBS" -o '%.18i %.24j %.10T %.10M %.6D %R' || true
echo ACCOUNTING
sacct -j "$JOBS" --format=JobID,JobName%24,State,Elapsed,ExitCode -P || true

echo TOP_ERRORS
grep -n -E "$ERROR_PATTERN" "$ROOT"/logs/*.out "$ROOT"/logs/*.err 2>/dev/null || true

echo CHILD_ERROR_FILES
find "$ROOT/bundles" -type f -name '*.out' -exec grep -l -E "$ERROR_PATTERN" {} + 2>/dev/null || true
find "$ROOT/bundles" -type f -name '*.err' -exec grep -l -E "$ERROR_PATTERN" {} + 2>/dev/null || true

echo TRAIN_FILES
find "$ROOT/bundles" -type f -name train.out -print 2>/dev/null | sort

echo TRAIN_PROGRESS
find "$ROOT/bundles" -type f -name train.out -exec grep -H -E "$PROGRESS_PATTERN" {} + 2>/dev/null | tail -120

echo TERMINAL_ARTIFACTS
find "$ROOT" -type f \( -name completion.json -o -name final_results.json -o -name terminal_eval.json \) -print 2>/dev/null | sort
