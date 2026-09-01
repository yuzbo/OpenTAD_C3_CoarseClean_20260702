#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
BUNDLE="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_bd226aa.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v12.sh"
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v12

test ! -e "${RUNTIME}"
printf '%s  %s\n' \
    ffab0bb256bd2d46ebd8c7e5c880f2b807a95c9cde3b00ef571f91000efdabe9 \
    "${BUNDLE}" | sha256sum -c -
git clone \
    --branch codex/sparsehead-diagnostic-closure-20260729 \
    "${BUNDLE}" \
    "${RUNTIME}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = bd226aae0128b3e9e5d7c3a7c36c498ff28bf7b2
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 85d2a2f3532137ffd5540a11bd01880cb206b903
test -z "$(git -C "${RUNTIME}" status --porcelain)"

bash "${PREFLIGHT}"
