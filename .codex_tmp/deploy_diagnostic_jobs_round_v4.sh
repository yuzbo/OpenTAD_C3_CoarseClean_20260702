#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
INDEPENDENT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260729_v4
SUPPORT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v6
OFFICIAL_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v4

test ! -e "${INDEPENDENT_ROOT}"
test ! -e "${SUPPORT_ROOT}"
test ! -e "${OFFICIAL_ROOT}"
mkdir "${INDEPENDENT_ROOT}" "${SUPPORT_ROOT}" "${OFFICIAL_ROOT}"

install -m 0640 \
    "${ARTIFACT_ROOT}/phystime_independent_recompute_v4.sbatch" \
    "${INDEPENDENT_ROOT}/phystime_independent_recompute_v4.sbatch"
install -m 0640 \
    "${ARTIFACT_ROOT}/sdpq_support_audit_v6.sbatch" \
    "${SUPPORT_ROOT}/sdpq_support_audit_v6.sbatch"
install -m 0640 \
    "${ARTIFACT_ROOT}/actionformer_official_eval_v4.sbatch" \
    "${OFFICIAL_ROOT}/actionformer_official_eval_v4.sbatch"

printf '%s  %s\n' \
    ce300438b7960c724d0b522586aad725c74f40f22ec8b399063ada8b3bbcf8f3 \
    "${INDEPENDENT_ROOT}/phystime_independent_recompute_v4.sbatch" \
    | sha256sum -c -
printf '%s  %s\n' \
    1cb71bcc465cf3a7517a0e2eb0f30d4993231da5711be459a67f90cf65f29378 \
    "${SUPPORT_ROOT}/sdpq_support_audit_v6.sbatch" \
    | sha256sum -c -
printf '%s  %s\n' \
    30badf400cb939e4a4c15e38d1de7cd43af84a12c04ddef5ee4139dd4f3413e2 \
    "${OFFICIAL_ROOT}/actionformer_official_eval_v4.sbatch" \
    | sha256sum -c -

test_status=0
sbatch --test-only "${OFFICIAL_ROOT}/actionformer_official_eval_v4.sbatch" \
    > "${OFFICIAL_ROOT}/sbatch_test_only.log" 2>&1 || test_status=$?
sbatch --test-only "${SUPPORT_ROOT}/sdpq_support_audit_v6.sbatch" \
    > "${SUPPORT_ROOT}/sbatch_test_only.log" 2>&1 || test_status=$?
sbatch --test-only "${INDEPENDENT_ROOT}/phystime_independent_recompute_v4.sbatch" \
    > "${INDEPENDENT_ROOT}/sbatch_test_only.log" 2>&1 || test_status=$?

sha256sum \
    "${OFFICIAL_ROOT}/sbatch_test_only.log" \
    "${SUPPORT_ROOT}/sbatch_test_only.log" \
    "${INDEPENDENT_ROOT}/sbatch_test_only.log"
exit "${test_status}"
