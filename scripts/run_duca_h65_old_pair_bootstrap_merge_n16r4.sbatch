#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-old-bs-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_BOOTSTRAP_MERGE][FAIL] $*" >&2; exit 1; }
: "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
: "${DUCA_BOOTSTRAP_COMMIT:?DUCA_BOOTSTRAP_COMMIT is required}"
: "${DUCA_BOOTSTRAP_SHARD_ROOT:?DUCA_BOOTSTRAP_SHARD_ROOT is required}"
: "${DUCA_BOOTSTRAP_OUTPUT_ROOT:?DUCA_BOOTSTRAP_OUTPUT_ROOT is required}"
: "${DUCA_INPUT_IDENTITY_PATH:?DUCA_INPUT_IDENTITY_PATH is required}"

ROOT="$(cd -- "$DUCA_REPO_ROOT" && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "checkout must be clean"
[[ -r "$DUCA_INPUT_IDENTITY_PATH" ]] || fail "input identity is not readable"
[[ ! -e "$DUCA_BOOTSTRAP_OUTPUT_ROOT" ]] || fail "merge output root already exists"
mkdir -p "$DUCA_BOOTSTRAP_OUTPUT_ROOT"
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

SHARD_COUNT=16
shard_args=()
for ((index=0; index<SHARD_COUNT; index++)); do
  path="$DUCA_BOOTSTRAP_SHARD_ROOT/shard_$(printf '%03d' "$index").json"
  [[ -r "$path" ]] || fail "missing bootstrap shard: $path"
  shard_args+=(--shard "$path")
done
python tools/bata/merge_duca_h65_bootstrap_shards.py \
  "${shard_args[@]}" --output "$DUCA_BOOTSTRAP_OUTPUT_ROOT/paired_bootstrap.json"

python - "$DUCA_BOOTSTRAP_OUTPUT_ROOT/completion.json" "$DUCA_BOOTSTRAP_COMMIT" \
  "$DUCA_INPUT_IDENTITY_PATH" "$DUCA_BOOTSTRAP_OUTPUT_ROOT/paired_bootstrap.json" \
  "$SHARD_COUNT" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from tools.bata.duca_p0_training import atomic_write_json

output, commit, identity, bootstrap, expected_shards = sys.argv[1:]
identity, bootstrap = Path(identity).resolve(), Path(bootstrap).resolve()
payload = json.loads(bootstrap.read_text(encoding="utf-8"))
identity_payload = json.loads(identity.read_text(encoding="utf-8"))
if payload.get("samples") != 10000 or payload.get("lower_rank") != 250 or payload.get("upper_rank") != 9750:
    raise RuntimeError("merged bootstrap violates the frozen statistical contract")
if payload.get("execution", {}).get("shard_count") != int(expected_shards):
    raise RuntimeError("merged bootstrap shard count mismatch")
if identity_payload.get("schema_version") != "duca_h65_old_pair_input_identity_v1":
    raise RuntimeError("old-pair input identity schema mismatch")
prediction_sha = payload.get("prediction_sha256") or {}
if identity_payload.get("rankpack_prediction_sha256") != prediction_sha.get("rankpack"):
    raise RuntimeError("rankpack prediction identity mismatch")
if identity_payload.get("truetime_prediction_sha256") != prediction_sha.get("truetime"):
    raise RuntimeError("truetime prediction identity mismatch")
evaluation = payload.get("evaluation_config") or {}
annotation_path = Path(str(evaluation.get("ground_truth_filename", "")))
if not annotation_path.is_file() or identity_payload.get("annotation_sha256") != hashlib.sha256(annotation_path.read_bytes()).hexdigest():
    raise RuntimeError("annotation identity mismatch")
if identity_payload.get("evaluation_config_sha256") != payload.get("evaluation_config_sha256"):
    raise RuntimeError("evaluation config identity mismatch")
if identity_payload.get("evaluator") != payload.get("evaluator"):
    raise RuntimeError("official evaluator identity mismatch")
atomic_write_json(output, {
    "schema_version": "duca_h65_old_pair_bootstrap_completion_v1",
    "git_commit": commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_identity_path": str(identity),
    "input_identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
    "bootstrap_path": str(bootstrap),
    "bootstrap_sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
    "samples": 10000,
    "rng": "numpy.random.PCG64",
    "interval_ranks_one_based": [250, 9750],
})
PY
echo "[DUCA_H65_BOOTSTRAP_MERGE] complete: $DUCA_BOOTSTRAP_OUTPUT_ROOT/completion.json"
