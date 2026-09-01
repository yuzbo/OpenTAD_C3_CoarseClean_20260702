#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
OFFICIAL_SCRIPT="${ARTIFACT_ROOT}/actionformer_official_eval_v5.sbatch"
SUPPORT_SCRIPT="${ARTIFACT_ROOT}/sdpq_support_audit_v7.sbatch"
OFFICIAL_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v5
SUPPORT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v7
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v8
PREFLIGHT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/diagnostic_closure_linux_preflight_20260729_v8.log

printf '%s  %s\n' \
    ecffe3ab226d61419c2417c16126396140de533c540e5efea825a72d856f27f0 \
    "${OFFICIAL_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    847de7210881b5fcb75e7f10aa138e451ea51b5b4e59209fb28b3f96fb614533 \
    "${SUPPORT_SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
    265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6 \
    "${PREFLIGHT}" | sha256sum -c -
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 2b074845497f6ada3314cb895f0d4ab2f4ce3eca
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 7779862c5422dc8e527b304bf881a760b0c90625
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
    printf 'submission_sha256=%s\n' ecffe3ab226d61419c2417c16126396140de533c540e5efea825a72d856f27f0
    printf 'runtime_commit=%s\n' 2b074845497f6ada3314cb895f0d4ab2f4ce3eca
    printf 'runtime_tree=%s\n' 7779862c5422dc8e527b304bf881a760b0c90625
    printf 'linux_preflight_sha256=%s\n' 265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6
} > "${OFFICIAL_ROOT}/submission_receipt.txt"
{
    printf 'formal_job_id=%s\n' "${support_job_id}"
    printf 'test_only_output=%s\n' "${support_test_only}"
    printf 'submission_sha256=%s\n' 847de7210881b5fcb75e7f10aa138e451ea51b5b4e59209fb28b3f96fb614533
    printf 'runtime_commit=%s\n' 2b074845497f6ada3314cb895f0d4ab2f4ce3eca
    printf 'runtime_tree=%s\n' 7779862c5422dc8e527b304bf881a760b0c90625
    printf 'linux_preflight_sha256=%s\n' 265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6
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
