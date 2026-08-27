#!/usr/bin/env bash
set -euo pipefail
fail(){ echo "[PJST_D1_BOOTSTRAP][FAIL] $*" >&2; exit 1; }
: "${DUCA_REPO_ROOT:?}"; : "${DUCA_BOOTSTRAP_COMMIT:?}"; : "${DUCA_BOOTSTRAP_SHARD_ROOT:?}"; : "${DUCA_PJST_OFF_PREDICTION:?}"; : "${DUCA_PJST_ON_PREDICTION:?}"
cd "$(cd "$DUCA_REPO_ROOT" && pwd -P)"; [[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "wrong HEAD"; [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "dirty tree"
for p in "$DUCA_PJST_OFF_PREDICTION" "$DUCA_PJST_ON_PREDICTION"; do [[ -r "$p" ]] || fail "missing prediction: $p"; done
idx="${SLURM_ARRAY_TASK_ID:?}"; [[ "$idx" =~ ^[0-9]+$ && "$idx" -lt 16 ]] || fail "invalid shard"; start=$((idx*10000/16)); stop=$(((idx+1)*10000/16)); mkdir -p "$DUCA_BOOTSTRAP_SHARD_ROOT"; out="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$idx").json"; [[ ! -e "$out" ]] || fail "output collision"
python tools/bata/bootstrap_duca_h65_official_map.py --prediction "off=$DUCA_PJST_OFF_PREDICTION" --prediction "on=$DUCA_PJST_ON_PREDICTION" --baseline off --annotation "${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}" --nonce "DUCA-PJST-D1-OFF-ON-v001" --namespace PAIRED_VIDEO_BOOTSTRAP_V1 --workers "${SLURM_CPUS_PER_TASK:-8}" --evaluator-thread 16 --chunksize 1 --sample-start "$start" --sample-stop "$stop" --output "$out"
