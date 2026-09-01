#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
OFFICIAL_SCRIPT="${ARTIFACT_ROOT}/actionformer_official_eval_v6.sbatch"
SUPPORT_SCRIPT="${ARTIFACT_ROOT}/sdpq_support_audit_v8.sbatch"
OFFICIAL_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260730_v6
SUPPORT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260730_v8
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v9
PREFLIGHT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/diagnostic_closure_linux_preflight_20260730_v9.log

printf '%s  %s\n' \
    4f72413d0cfc1142daf05d21d4d83da03e3de0d7d2c5cb17629b10d7aaa8e459 \
    "${OFFICIAL_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    fc9fe0de85e5845624cdb18c9ed9a44044a4845ed062f4c6d65b8340d8b8d685 \
    "${SUPPORT_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    c56b5cc3c69fa079c87d43fed73d4eb7ff8a60d4f68f2d7e35ca684c52737064 \
    "${PREFLIGHT}" | sha256sum -c -
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 53eb384f0e892812527ecc5165f5073372f2b1e8
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = e6cec6a49d47296b88215dd5e4e3baac9892198c
test -z "$(git -C "${RUNTIME}" status --porcelain)"
test ! -e "${OFFICIAL_ROOT}"
test ! -e "${SUPPORT_ROOT}"

mkdir "${OFFICIAL_ROOT}" "${SUPPORT_ROOT}"
cp --preserve=mode,timestamps "${OFFICIAL_SCRIPT}" "${OFFICIAL_ROOT}/submission.sbatch"
cp --preserve=mode,timestamps "${SUPPORT_SCRIPT}" "${SUPPORT_ROOT}/submission.sbatch"

official_test_only="$(sbatch --test-only "${OFFICIAL_ROOT}/submission.sbatch" 2>&1)"
support_test_only="$(sbatch --test-only "${SUPPORT_ROOT}/submission.sbatch" 2>&1)"
printf '%s\n' "${official_test_only}" > "${OFFICIAL_ROOT}/sbatch_test_only.txt"
printf '%s\n' "${support_test_only}" > "${SUPPORT_ROOT}/sbatch_test_only.txt"

official_job_id="$(sbatch --parsable "${OFFICIAL_ROOT}/submission.sbatch")"
support_job_id="$(sbatch --parsable "${SUPPORT_ROOT}/submission.sbatch")"
case "${official_job_id}" in
    *[!0-9]*|"") echo "invalid official job ID: ${official_job_id}" >&2; exit 1 ;;
esac
case "${support_job_id}" in
    *[!0-9]*|"") echo "invalid support job ID: ${support_job_id}" >&2; exit 1 ;;
esac

{
    printf 'formal_job_id=%s\n' "${official_job_id}"
    printf 'test_only_output=%s\n' "${official_test_only}"
    printf 'submission_sha256=%s\n' 4f72413d0cfc1142daf05d21d4d83da03e3de0d7d2c5cb17629b10d7aaa8e459
    printf 'runtime_commit=%s\n' 53eb384f0e892812527ecc5165f5073372f2b1e8
    printf 'runtime_tree=%s\n' e6cec6a49d47296b88215dd5e4e3baac9892198c
    printf 'linux_preflight_sha256=%s\n' c56b5cc3c69fa079c87d43fed73d4eb7ff8a60d4f68f2d7e35ca684c52737064
} > "${OFFICIAL_ROOT}/submission_receipt.txt"
{
    printf 'formal_job_id=%s\n' "${support_job_id}"
    printf 'test_only_output=%s\n' "${support_test_only}"
    printf 'submission_sha256=%s\n' fc9fe0de85e5845624cdb18c9ed9a44044a4845ed062f4c6d65b8340d8b8d685
    printf 'runtime_commit=%s\n' 53eb384f0e892812527ecc5165f5073372f2b1e8
    printf 'runtime_tree=%s\n' e6cec6a49d47296b88215dd5e4e3baac9892198c
    printf 'linux_preflight_sha256=%s\n' c56b5cc3c69fa079c87d43fed73d4eb7ff8a60d4f68f2d7e35ca684c52737064
} > "${SUPPORT_ROOT}/submission_receipt.txt"

sha256sum \
    "${OFFICIAL_ROOT}/submission.sbatch" \
    "${OFFICIAL_ROOT}/sbatch_test_only.txt" \
    "${OFFICIAL_ROOT}/submission_receipt.txt" \
    > "${OFFICIAL_ROOT}/submission_sha256s.txt"
sha256sum \
    "${SUPPORT_ROOT}/submission.sbatch" \
    "${SUPPORT_ROOT}/sbatch_test_only.txt" \
    "${SUPPORT_ROOT}/submission_receipt.txt" \
    > "${SUPPORT_ROOT}/submission_sha256s.txt"

printf 'official_formal_job_id=%s\n' "${official_job_id}"
printf 'official_test_only=%s\n' "${official_test_only}"
printf 'support_formal_job_id=%s\n' "${support_job_id}"
printf 'support_test_only=%s\n' "${support_test_only}"
