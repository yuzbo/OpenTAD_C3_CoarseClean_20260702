#!/usr/bin/env bash
set -euo pipefail

for name in \
  DUCA_RIME_DENSE_ACTIONFORMER_CONFIG \
  DUCA_RIME_DENSE_ACTIONFORMER_ROOT; do
  [[ -n "${!name:-}" ]] || {
    echo "[DUCA_RIME_DENSE_ACTIONFORMER][FAIL] ${name} is required" >&2
    exit 1
  }
done

export DUCA_RIME_DENSE_BACKEND=ActionFormer
export DUCA_RIME_DENSE_TRIDET_CONFIG="${DUCA_RIME_DENSE_ACTIONFORMER_CONFIG}"
export DUCA_RIME_DENSE_TRIDET_ROOT="${DUCA_RIME_DENSE_ACTIONFORMER_ROOT}"
exec scripts/run_duca_rime_dense_tridet_train.sh
