#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-old-bs-shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
set -euo pipefail

fail() { echo "[DUCA_H65_BOOTSTRAP_SHARD][FAIL] $*" >&2; exit 1; }
: "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
: "${DUCA_BOOTSTRAP_COMMIT:?DUCA_BOOTSTRAP_COMMIT is required}"
: "${DUCA_BOOTSTRAP_SHARD_ROOT:?DUCA_BOOTSTRAP_SHARD_ROOT is required}"

ROOT="$(cd -- "$DUCA_REPO_ROOT" && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "checkout must be clean"
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

PAIR_ROOT="${DUCA_OLD_PAIR_ROOT:-/data/run01/sczc063/yuzibo/duca_truetime_curriculum_official60_11126684_20260821T200057Z}"
RANKPACK_PREDICTION="$PAIR_ROOT/rankpack_k384/run/terminal_eval/gpu1_id0/result_detection.json"
TRUETIME_PREDICTION="$PAIR_ROOT/truetime_k384/run/terminal_eval/gpu1_id0/result_detection.json"
ANNOTATION="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
for path in "$RANKPACK_PREDICTION" "$TRUETIME_PREDICTION" "$ANNOTATION"; do
  [[ -r "$path" ]] || fail "immutable input is missing: $path"
done

SHARD_COUNT=16
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_INDEX" -lt "$SHARD_COUNT" ]] || fail "invalid shard index"
START=$(( SHARD_INDEX * 10000 / SHARD_COUNT ))
STOP=$(( (SHARD_INDEX + 1) * 10000 / SHARD_COUNT ))
mkdir -p "$DUCA_BOOTSTRAP_SHARD_ROOT"
OUTPUT="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$SHARD_INDEX").json"
[[ ! -e "$OUTPUT" ]] || fail "shard output already exists: $OUTPUT"

python tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "rankpack=$RANKPACK_PREDICTION" \
  --prediction "truetime=$TRUETIME_PREDICTION" \
  --baseline rankpack --annotation "$ANNOTATION" \
  --nonce "DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823" \
  --namespace "PAIRED_VIDEO_BOOTSTRAP_V1" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --evaluator-thread 16 --chunksize 1 \
  --sample-start "$START" --sample-stop "$STOP" --output "$OUTPUT"
echo "[DUCA_H65_BOOTSTRAP_SHARD] complete index=$SHARD_INDEX range=[$START,$STOP) output=$OUTPUT"
