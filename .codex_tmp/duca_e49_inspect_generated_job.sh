#!/usr/bin/env bash
set -euo pipefail
R5=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r5
echo CELL_JOB
sed -n '1,240p' ${R5}/jobs/actionformer_uniform_k384_s3407.sbatch
echo COST_JOB
sed -n '1,220p' ${R5}/jobs/cost_actionformer_uniform_k384_s3407.sbatch
echo AGG_JOB
sed -n '1,220p' ${R5}/jobs/aggregate.sbatch
