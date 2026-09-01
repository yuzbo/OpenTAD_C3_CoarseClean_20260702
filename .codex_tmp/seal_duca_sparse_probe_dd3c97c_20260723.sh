#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329
COMMIT=dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45
SNAPSHOT=/data/run01/sczc063/yuzibo/projects/opentad_duca_sparse_62efd9c_20260723
GATE_JOB=1180556
SUITE_JOB=1180557
GATE_SBATCH=${ROOT}/jobs/cuda_gate.sbatch
SUITE_SBATCH=${ROOT}/suite/jobs/sparse_probe_four_stride.sbatch

printf 'role\tjob_id\tdependency\tgpus\tsbatch\n' > "${ROOT}/jobs.tsv"
printf 'cuda_gate\t%s\tnone\t1\t%s\n' "${GATE_JOB}" "${GATE_SBATCH}" >> "${ROOT}/jobs.tsv"
printf 'sparse_probe_d1_d4\t%s\tafterok:%s\t4\t%s\n' \
  "${SUITE_JOB}" "${GATE_JOB}" "${SUITE_SBATCH}" >> "${ROOT}/jobs.tsv"

jobs_sha=$(sha256sum "${ROOT}/jobs.tsv" | awk '{print $1}')
gate_sha=$(sha256sum "${GATE_SBATCH}" | awk '{print $1}')
suite_sha=$(sha256sum "${SUITE_SBATCH}" | awk '{print $1}')
printf '%s\n' "${jobs_sha}" > "${ROOT}/jobs.tsv.sha256"

cat > "${ROOT}/deployment_receipt.json" <<EOF
{
  "schema": "duca_sparse_probe_deployment_receipt_v1",
  "ok": true,
  "task": "offline_temporal_action_detection",
  "git_commit": "${COMMIT}",
  "snapshot": "${SNAPSHOT}",
  "run_root": "${ROOT}",
  "gate_job": ${GATE_JOB},
  "suite_job": ${SUITE_JOB},
  "suite_dependency": "afterok:${GATE_JOB}",
  "gate_sbatch_sha256": "${gate_sha}",
  "suite_sbatch_sha256": "${suite_sha}",
  "jobs_tsv_sha256": "${jobs_sha}",
  "strides_dense_candidates": [1, 2, 3, 4],
  "intervals_source_frames": [4, 8, 12, 16],
  "terminal_metric": "official_validation_epoch59_state_dict_ema_map"
}
EOF

sha256sum "${ROOT}/deployment_receipt.json" > "${ROOT}/deployment_receipt.json.sha256"
cat "${ROOT}/jobs.tsv"
cat "${ROOT}/deployment_receipt.json"
