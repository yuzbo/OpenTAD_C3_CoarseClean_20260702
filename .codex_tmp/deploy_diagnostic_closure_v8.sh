#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
BUNDLE="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_2b07484.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v8.sh"
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v8

test ! -e "${RUNTIME}"
printf '%s  %s\n' \
    a97dc6d61e4e6a4e4fb6734fd7d3c724ada5ae8f88ea4d7ef29903fc80037686 \
    "${BUNDLE}" | sha256sum -c -
git clone \
    --branch codex/sparsehead-diagnostic-closure-20260729 \
    "${BUNDLE}" \
    "${RUNTIME}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 2b074845497f6ada3314cb895f0d4ab2f4ce3eca
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 7779862c5422dc8e527b304bf881a760b0c90625
test -z "$(git -C "${RUNTIME}" status --porcelain)"

bash "${PREFLIGHT}"
