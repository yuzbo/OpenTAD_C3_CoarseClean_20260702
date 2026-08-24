#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-old-bs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_OLD_BOOTSTRAP][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_COMMIT="${DUCA_BOOTSTRAP_COMMIT:?DUCA_BOOTSTRAP_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "bootstrap checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "bootstrap checkout must be clean"

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
PYTHON="${PYTHON:-python}"
PAIR_ROOT="${DUCA_OLD_PAIR_ROOT:-/data/run01/sczc063/yuzibo/duca_truetime_curriculum_official60_11126684_20260821T200057Z}"
RANKPACK_PREDICTION="$PAIR_ROOT/rankpack_k384/run/terminal_eval/gpu1_id0/result_detection.json"
TRUETIME_PREDICTION="$PAIR_ROOT/truetime_k384/run/terminal_eval/gpu1_id0/result_detection.json"
RANKPACK_METRICS="$PAIR_ROOT/rankpack_k384/run/terminal_evaluation.json"
TRUETIME_METRICS="$PAIR_ROOT/truetime_k384/run/terminal_evaluation.json"
ANNOTATION="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
CATEGORY="/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
OUTPUT_ROOT="${DUCA_BOOTSTRAP_OUTPUT_ROOT:?DUCA_BOOTSTRAP_OUTPUT_ROOT is required}"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "bootstrap output root already exists: $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

for path in "$RANKPACK_PREDICTION" "$TRUETIME_PREDICTION" "$RANKPACK_METRICS" "$TRUETIME_METRICS" "$ANNOTATION" "$CATEGORY"; do
  [[ -r "$path" ]] || fail "required immutable evidence is missing: $path"
done

"$PYTHON" - "$RANKPACK_PREDICTION" "$TRUETIME_PREDICTION" \
  "$RANKPACK_METRICS" "$TRUETIME_METRICS" "$ANNOTATION" "$CATEGORY" \
  "$OUTPUT_ROOT/input_identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.duca_p0_evaluation import official_evaluator_identity, prediction_counts
from tools.bata.duca_p0_training import atomic_write_json

rank_prediction, true_prediction, rank_metrics, true_metrics, annotation, category, output = map(Path, sys.argv[1:])
rank_payload = json.loads(rank_metrics.read_text(encoding="utf-8"))
true_payload = json.loads(true_metrics.read_text(encoding="utf-8"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for label, prediction, payload in (
    ("rankpack", rank_prediction, rank_payload),
    ("truetime", true_prediction, true_payload),
):
    if payload.get("prediction_sha256") != sha(prediction):
        raise RuntimeError(f"{label} terminal sidecar does not bind its prediction")
    if payload.get("evaluation_annotation_sha256") != sha(annotation):
        raise RuntimeError(f"{label} terminal sidecar annotation mismatch")
    if payload.get("evaluation_class_map_sha256") != sha(category):
        raise RuntimeError(f"{label} terminal sidecar class-map mismatch")
if rank_payload.get("evaluation_config_sha256") != true_payload.get("evaluation_config_sha256"):
    raise RuntimeError("old paired predictions use different evaluator configurations")
if rank_payload.get("evaluator") != true_payload.get("evaluator"):
    raise RuntimeError("old paired predictions use different evaluator identities")
observed = rank_payload.get("evaluator") or {}
current = official_evaluator_identity()
identity_keys = ("module", "class_name", "source_sha256")
if any(observed.get(key) != current.get(key) for key in identity_keys):
    raise RuntimeError("old paired prediction evaluator differs from this checkout")
rank_counts = prediction_counts(rank_prediction)
true_counts = prediction_counts(true_prediction)
if rank_counts != true_counts or rank_counts[1] != 211:
    raise RuntimeError("old paired predictions do not cover the same 211 videos")
atomic_write_json(
    output,
    {
        "schema_version": "duca_h65_old_pair_input_identity_v1",
        "rankpack_prediction": str(rank_prediction.resolve()),
        "rankpack_prediction_sha256": sha(rank_prediction),
        "truetime_prediction": str(true_prediction.resolve()),
        "truetime_prediction_sha256": sha(true_prediction),
        "prediction_count": rank_counts[0],
        "video_count": rank_counts[1],
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha(annotation),
        "category": str(category.resolve()),
        "category_sha256": sha(category),
        "evaluation_config_sha256": rank_payload["evaluation_config_sha256"],
        "evaluator": rank_payload["evaluator"],
    },
)
PY

"$PYTHON" tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "rankpack=$RANKPACK_PREDICTION" \
  --prediction "truetime=$TRUETIME_PREDICTION" \
  --baseline rankpack \
  --annotation "$ANNOTATION" \
  --nonce "DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823" \
  --namespace "PAIRED_VIDEO_BOOTSTRAP_V1" \
  --workers "${DUCA_BOOTSTRAP_WORKERS:-8}" \
  --evaluator-thread 16 \
  --chunksize 1 \
  --output "$OUTPUT_ROOT/paired_bootstrap.json" \
  2>&1 | tee "$OUTPUT_ROOT/bootstrap.out"

"$PYTHON" - "$OUTPUT_ROOT/completion.json" "$EXPECTED_COMMIT" \
  "$OUTPUT_ROOT/input_identity.json" "$OUTPUT_ROOT/paired_bootstrap.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from tools.bata.duca_p0_training import atomic_write_json

output, commit, identity, bootstrap = sys.argv[1:]
identity = Path(identity).resolve()
bootstrap = Path(bootstrap).resolve()
payload = json.loads(bootstrap.read_text(encoding="utf-8"))
if payload.get("samples") != 10000 or payload.get("lower_rank") != 250 or payload.get("upper_rank") != 9750:
    raise RuntimeError("old paired bootstrap did not satisfy the frozen statistical contract")
atomic_write_json(
    output,
    {
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
    },
)
PY
echo "[DUCA_H65_OLD_BOOTSTRAP] complete: $OUTPUT_ROOT/completion.json"
