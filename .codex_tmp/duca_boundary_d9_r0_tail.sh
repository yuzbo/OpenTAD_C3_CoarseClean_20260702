#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357
date '+%F %T %z'
tail -100 "${ROOT}/logs/burst_r0_d9fb398-1179517.out" || true
echo STDERR
tail -60 "${ROOT}/logs/burst_r0_d9fb398-1179517.err" || true
echo PROCS
sstat -j 1179517.batch --format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize -P || true
