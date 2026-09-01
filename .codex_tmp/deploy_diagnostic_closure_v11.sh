#!/usr/bin/env bash
set -euo pipefail
umask 027

ARTIFACT_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts
BUNDLE="${ARTIFACT_ROOT}/sparsehead_diagnostic_closure_7e9b9a7.bundle"
PREFLIGHT="${ARTIFACT_ROOT}/diagnostic_closure_linux_preflight_v11.sh"
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v11

test ! -e "${RUNTIME}"
printf '%s  %s\n' \
    77c7cbfcde3675120ee329e05bd3faf998b45b9fe3bf106d574eaafcfbde18b0 \
    "${BUNDLE}" | sha256sum -c -
git clone \
    --branch codex/sparsehead-diagnostic-closure-20260729 \
    "${BUNDLE}" \
    "${RUNTIME}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 7e9b9a7c692482ba9905ec4d9466312db225b3fe
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 875f4bd47ae0ade9e9d080b8409faf77f127ff37
test -z "$(git -C "${RUNTIME}" status --porcelain)"

bash "${PREFLIGHT}"
