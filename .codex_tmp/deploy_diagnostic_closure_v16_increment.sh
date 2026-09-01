#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
INCREMENT="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_e2a0d74_increment.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v16.sh"
BASE_RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v15
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v16
BRANCH=codex/sparsehead-diagnostic-closure-20260729
BASE_COMMIT=a88befdf105b794d540fa13160641e7fb6294a8b
COMMIT=e2a0d74f561b158c531d4909e72ecee69b153c16
TREE=0b6cb7996ee90f3209a78b78bbf7a55525e3badd

test ! -e "${RUNTIME}"
test "$(git -C "${BASE_RUNTIME}" rev-parse HEAD)" = "${BASE_COMMIT}"
test -z "$(git -C "${BASE_RUNTIME}" status --porcelain)"
printf '%s  %s\n' \
    89a992c7470949550886432580c9043ce04c7ce0fe42c2fe6b5cd6ec2603a88c \
    "${INCREMENT}" | sha256sum -c -

git clone "${BASE_RUNTIME}" "${RUNTIME}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = "${BASE_COMMIT}"
git -C "${RUNTIME}" fetch \
    "${INCREMENT}" \
    "refs/heads/${BRANCH}:refs/remotes/increment/candidate"
git -C "${RUNTIME}" checkout -B "${BRANCH}" "${COMMIT}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = "${COMMIT}"
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = "${TREE}"
test -z "$(git -C "${RUNTIME}" status --porcelain)"

bash "${PREFLIGHT}"
