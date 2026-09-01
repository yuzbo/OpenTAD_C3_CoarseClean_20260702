#!/usr/bin/env bash
set -euo pipefail

STAGE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
SPARSE_ROOT="${ROOT_BASE}/duca_sparse_cee4ccd_recovery_${STAMP}"
MOBILE_ROOT="${ROOT_BASE}/duca_mobilenet_e30db0f_retry_${STAMP}"
mkdir -p "${SPARSE_ROOT}/logs" "${SPARSE_ROOT}/jobs" \
  "${MOBILE_ROOT}/logs" "${MOBILE_ROOT}/jobs"

sparse_job="$(sbatch --parsable \
  --job-name=duca-sparse-recovery \
  --output="${SPARSE_ROOT}/logs/suite.%j.out" \
  --error="${SPARSE_ROOT}/logs/suite.%j.err" \
  --export="ALL,RUN_ROOT=${SPARSE_ROOT}" \
  "${STAGE}/recover_sparse_cee4ccd.sbatch")"
mobile_job="$(sbatch --parsable \
  --job-name=duca-mobilenet-retry \
  --output="${MOBILE_ROOT}/logs/suite.%j.out" \
  --error="${MOBILE_ROOT}/logs/suite.%j.err" \
  --export="ALL,RUN_ROOT=${MOBILE_ROOT}" \
  "${STAGE}/retry_mobilenet_e30db0f.sbatch")"

cat > "${SPARSE_ROOT}/deployment_manifest.txt" <<EOF
task=offline_tad_sparse_probe_recovery
code_commit=cee4ccd33fb20e11978e4a2a6eaa3f5845b51489
code_snapshot=/data/run01/sczc063/yuzibo/projects/opentad_duca_sparse_cee4ccd_20260723
source_p0_commit=dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45
source_p0_root=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329/suite
job_id=${sparse_job}
run_root=${SPARSE_ROOT}
EOF
cat > "${MOBILE_ROOT}/deployment_manifest.txt" <<EOF
task=offline_tad_target_train_free_mobilenet_retry
code_commit=e30db0f3987128798da6bc8ff446065b818b1a7f
code_snapshot=/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_e30db0f_20260723
job_id=${mobile_job}
run_root=${MOBILE_ROOT}
EOF
sha256sum "${SPARSE_ROOT}/deployment_manifest.txt" > "${SPARSE_ROOT}/deployment_hashes.sha256"
sha256sum "${MOBILE_ROOT}/deployment_manifest.txt" > "${MOBILE_ROOT}/deployment_hashes.sha256"

printf 'SPARSE_JOB=%s\nSPARSE_ROOT=%s\n' "${sparse_job}" "${SPARSE_ROOT}"
cat "${SPARSE_ROOT}/deployment_hashes.sha256"
printf 'MOBILE_JOB=%s\nMOBILE_ROOT=%s\n' "${mobile_job}" "${MOBILE_ROOT}"
cat "${MOBILE_ROOT}/deployment_hashes.sha256"
