#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v13.sh"
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v13
REPOSITORY=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
BRANCH=codex/sparsehead-diagnostic-closure-20260729
COMMIT=e7a31c4f1d79eedf9672409571871db9e3e5fca9
TREE=41bbae08262e78066dc9a25b97c281f1fee38351

test ! -e "${RUNTIME}"
test "$(git ls-remote "${REPOSITORY}" "refs/heads/${BRANCH}" | cut -f1)" = "${COMMIT}"
git clone \
    --branch "${BRANCH}" \
    --single-branch \
    "${REPOSITORY}" \
    "${RUNTIME}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = "${COMMIT}"
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = "${TREE}"
test -z "$(git -C "${RUNTIME}" status --porcelain)"

bash "${PREFLIGHT}"
