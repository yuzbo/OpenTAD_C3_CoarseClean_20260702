#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
INCREMENT="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_e7a31c4_increment.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v14.sh"
BASE_RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v12
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v14
BRANCH=codex/sparsehead-diagnostic-closure-20260729
BASE_COMMIT=bd226aae0128b3e9e5d7c3a7c36c498ff28bf7b2
COMMIT=e7a31c4f1d79eedf9672409571871db9e3e5fca9
TREE=41bbae08262e78066dc9a25b97c281f1fee38351

test ! -e "${RUNTIME}"
test "$(git -C "${BASE_RUNTIME}" rev-parse HEAD)" = "${BASE_COMMIT}"
test -z "$(git -C "${BASE_RUNTIME}" status --porcelain)"
printf '%s  %s\n' \
    ccd13723e13c04df3167e2b659d6ade6d9a358710e15de762c0a6442f3fcc62a \
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
