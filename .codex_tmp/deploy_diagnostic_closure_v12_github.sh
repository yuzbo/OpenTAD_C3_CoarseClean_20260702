#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v12.sh"
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v12
REPOSITORY=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
BRANCH=codex/sparsehead-diagnostic-closure-20260729
COMMIT=bd226aae0128b3e9e5d7c3a7c36c498ff28bf7b2
TREE=85d2a2f3532137ffd5540a11bd01880cb206b903

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
