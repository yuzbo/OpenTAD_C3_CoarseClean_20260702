#!/usr/bin/env bash
set -u

for job_id in 1180557 1180654; do
  echo "=== JOB ${job_id} ==="
  scontrol write batch_script "${job_id}" - 2>/dev/null || true
done

echo '=== SPARSE STORED SCRIPTS ==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329 -type f \
  \( -name '*.sbatch' -o -path '*/jobs/*' \) -maxdepth 5 -print -exec sed -n '1,240p' {} \; 2>/dev/null || true

echo '=== MOBILE STORED SCRIPTS ==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115 -type f \
  \( -name '*.sbatch' -o -path '*/jobs/*' \) -maxdepth 5 -print -exec sed -n '1,260p' {} \; 2>/dev/null || true
