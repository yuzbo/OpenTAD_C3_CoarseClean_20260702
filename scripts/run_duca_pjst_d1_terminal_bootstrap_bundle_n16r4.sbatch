#!/usr/bin/env bash
# One allocation: bounded local shard parallelism, then fail-closed merge.
set -euo pipefail
fail(){ echo "[PJST_D1_BUNDLE][FAIL] $*" >&2; exit 1; }
: "${DUCA_REPO_ROOT:?}"; : "${DUCA_BOOTSTRAP_COMMIT:?}"; : "${DUCA_BOOTSTRAP_SHARD_ROOT:?}"; : "${DUCA_BOOTSTRAP_OUTPUT_ROOT:?}"; : "${DUCA_PJST_OFF_PREDICTION:?}"; : "${DUCA_PJST_ON_PREDICTION:?}"
ROOT="$(cd "$DUCA_REPO_ROOT" && pwd -P)"; cd "$ROOT"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "wrong HEAD"; [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "dirty tree"
for p in "$DUCA_PJST_OFF_PREDICTION" "$DUCA_PJST_ON_PREDICTION"; do [[ -r "$p" ]] || fail "missing prediction: $p"; done
[[ ! -e "$DUCA_BOOTSTRAP_OUTPUT_ROOT" ]] || fail "output collision"; mkdir -p "$DUCA_BOOTSTRAP_SHARD_ROOT"
workers="${DUCA_BOOTSTRAP_WORKERS:-4}"; [[ "$workers" =~ ^[1-9][0-9]*$ ]] || fail "workers must be positive"; (( workers <= 16 )) || fail "workers exceed shard count"
run_shard(){ local i="$1" start=$((i*10000/16)) stop=$(((i+1)*10000/16)) out="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$i").json"; [[ ! -e "$out" ]] || fail "shard output collision: $out"; python tools/bata/bootstrap_duca_h65_official_map.py --prediction "off=$DUCA_PJST_OFF_PREDICTION" --prediction "on=$DUCA_PJST_ON_PREDICTION" --baseline off --annotation "${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}" --nonce "DUCA-PJST-D1-OFF-ON-v001" --namespace PAIRED_VIDEO_BOOTSTRAP_V1 --workers 1 --evaluator-thread 16 --chunksize 1 --sample-start "$start" --sample-stop "$stop" --output "$out"; }
running=(); for i in $(seq 0 15); do run_shard "$i" & running+=("$!"); while (( ${#running[@]} >= workers )); do wait "${running[0]}" || fail "bootstrap shard failed"; running=("${running[@]:1}"); done; done; for pid in "${running[@]}"; do wait "$pid" || fail "bootstrap shard failed"; done
args=(); for i in $(seq 0 15); do p="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$i").json"; [[ -r "$p" ]] || fail "missing shard $i"; args+=(--shard "$p"); done
mkdir -p "$DUCA_BOOTSTRAP_OUTPUT_ROOT"; python tools/bata/merge_duca_h65_bootstrap_shards.py "${args[@]}" --output "$DUCA_BOOTSTRAP_OUTPUT_ROOT/paired_bootstrap.json"
