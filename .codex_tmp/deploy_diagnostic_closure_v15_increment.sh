#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
INCREMENT="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_a88befd_increment.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v15.sh"
BASE_RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v14
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v15
BRANCH=codex/sparsehead-diagnostic-closure-20260729
BASE_COMMIT=e7a31c4f1d79eedf9672409571871db9e3e5fca9
COMMIT=a88befdf105b794d540fa13160641e7fb6294a8b
TREE=e1f1cfd36cb86ef04a73b04482440fd5fcc66842

test ! -e "${RUNTIME}"
test "$(git -C "${BASE_RUNTIME}" rev-parse HEAD)" = "${BASE_COMMIT}"
test -z "$(git -C "${BASE_RUNTIME}" status --porcelain)"
printf '%s  %s\n' \
    8cf219247282d6cf5c9399de2b9ce2f0cf422de130f01a35979f124d686af336 \
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
