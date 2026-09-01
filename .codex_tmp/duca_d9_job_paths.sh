#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357
printf '%s\n' '=== jobs.tsv ==='
cat "$ROOT/jobs.tsv"
printf '%s\n' '=== files ==='
find "$ROOT" -maxdepth 3 -type f \( -name '*.sbatch' -o -name '*decision*.json' -o -name '*summary*.json' \) -print | sort
printf '%s\n' '=== p0 sbatch ==='
P0=$(find "$ROOT" -maxdepth 3 -type f -name '*p0*.sbatch' | head -1)
test -n "$P0"
sed -n '1,240p' "$P0"
