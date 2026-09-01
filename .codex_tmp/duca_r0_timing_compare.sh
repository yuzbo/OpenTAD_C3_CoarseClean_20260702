#!/usr/bin/env bash
set -euo pipefail

OLD=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357/r0_holdout_map
NEW=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map

echo OLD_JOB
sacct -j 1179517 --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ExitCode -P -X || true
echo OLD_TIMES
find "${OLD}" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %f\n' | sort || true
echo NEW_JOB
sacct -j 1179795 --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ExitCode -P -X || true
echo NEW_TIMES
find "${NEW}" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %f\n' | sort || true
