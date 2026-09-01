#!/usr/bin/env bash
set -euo pipefail
umask 027

OFFICIAL_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v4
SUPPORT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v6
INDEPENDENT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260729_v6
OFFICIAL_SCRIPT="${OFFICIAL_ROOT}/actionformer_official_eval_v4.sbatch"
SUPPORT_SCRIPT="${SUPPORT_ROOT}/sdpq_support_audit_v6.sbatch"
INDEPENDENT_SCRIPT="${INDEPENDENT_ROOT}/phystime_independent_recompute_v6.sbatch"

test ! -e "${OFFICIAL_ROOT}/formal_job_id.txt"
test ! -e "${SUPPORT_ROOT}/formal_job_id.txt"
test ! -e "${INDEPENDENT_ROOT}/formal_job_id.txt"
printf '%s  %s\n' \
    30badf400cb939e4a4c15e38d1de7cd43af84a12c04ddef5ee4139dd4f3413e2 \
    "${OFFICIAL_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    1cb71bcc465cf3a7517a0e2eb0f30d4993231da5711be459a67f90cf65f29378 \
    "${SUPPORT_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    1df9504628102f995d6d59921a16a2c35002cb603c745c4f1225cd70422c9b30 \
    "${INDEPENDENT_SCRIPT}" | sha256sum -c -

official_job="$(sbatch --parsable "${OFFICIAL_SCRIPT}")"
printf '%s\n' "${official_job}" > "${OFFICIAL_ROOT}/formal_job_id.txt"
support_job="$(sbatch --parsable "${SUPPORT_SCRIPT}")"
printf '%s\n' "${support_job}" > "${SUPPORT_ROOT}/formal_job_id.txt"
independent_job="$(sbatch --parsable "${INDEPENDENT_SCRIPT}")"
printf '%s\n' "${independent_job}" > "${INDEPENDENT_ROOT}/formal_job_id.txt"

sha256sum \
    "${OFFICIAL_SCRIPT}" \
    "${OFFICIAL_ROOT}/sbatch_test_only.log" \
    "${OFFICIAL_ROOT}/formal_job_id.txt" \
    > "${OFFICIAL_ROOT}/submission_sha256s.txt"
sha256sum \
    "${SUPPORT_SCRIPT}" \
    "${SUPPORT_ROOT}/sbatch_test_only.log" \
    "${SUPPORT_ROOT}/formal_job_id.txt" \
    > "${SUPPORT_ROOT}/submission_sha256s.txt"
sha256sum \
    "${INDEPENDENT_SCRIPT}" \
    "${INDEPENDENT_ROOT}/sbatch_test_only.log" \
    "${INDEPENDENT_ROOT}/formal_job_id.txt" \
    > "${INDEPENDENT_ROOT}/submission_sha256s.txt"

printf 'official_job=%s\n' "${official_job}"
printf 'support_job=%s\n' "${support_job}"
printf 'independent_job=%s\n' "${independent_job}"
