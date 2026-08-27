#!/usr/bin/env bash
set -euo pipefail
fail(){ echo "[PJST_D1_MERGE][FAIL] $*" >&2; exit 1; }; : "${DUCA_REPO_ROOT:?}"; : "${DUCA_BOOTSTRAP_COMMIT:?}"; : "${DUCA_BOOTSTRAP_SHARD_ROOT:?}"; : "${DUCA_BOOTSTRAP_OUTPUT_ROOT:?}"
cd "$(cd "$DUCA_REPO_ROOT" && pwd -P)"; [[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "wrong HEAD"; [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "dirty tree"; [[ ! -e "$DUCA_BOOTSTRAP_OUTPUT_ROOT" ]] || fail "output collision"; mkdir -p "$DUCA_BOOTSTRAP_OUTPUT_ROOT"; args=(); for i in $(seq 0 15); do p="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$i").json"; [[ -r "$p" ]] || fail "missing shard $i"; args+=(--shard "$p"); done
python tools/bata/merge_duca_h65_bootstrap_shards.py "${args[@]}" --output "$DUCA_BOOTSTRAP_OUTPUT_ROOT/paired_bootstrap.json"
