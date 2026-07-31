#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH -J af-odfcr-g3

set -o pipefail
source /etc/profile >/dev/null 2>&1
set -eu
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"

case "$CANDIDATE_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*) ;;
  *) echo "candidate root outside write boundary" >&2; exit 2 ;;
esac
case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*/runs/*) ;;
  *) echo "run root outside write boundary" >&2; exit 2 ;;
esac
test -n "${SLURM_JOB_ID:-}"
test -d "$RUN_ROOT"
test -x "$ACTIONFORMER_PYTHON_ENV/bin/python"
test -f "$ACTIONFORMER_ENVIRONMENT_RECEIPT"
test "$(sha256sum "$ACTIONFORMER_ENVIRONMENT_RECEIPT" | awk '{print $1}')" = \
  "$EXPECTED_ENVIRONMENT_RECEIPT_SHA256"
export PATH="$ACTIONFORMER_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$ACTIONFORMER_PYTHON_ENV/bin/python")"
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = \
  "$EXPECTED_CANDIDATE_COMMIT"
test "$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')" = \
  "$EXPECTED_CANDIDATE_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"

G3_ROOT="$RUN_ROOT/aggregate_g3"
test ! -e "$G3_ROOT"
mkdir "$G3_ROOT"
RUN_LOG="$G3_ROOT/runtime.log"
STAGE=g3_aggregate
exec > >(tee -a "$RUN_LOG") 2>&1
failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    STATUS="$status" STAGE="$STAGE" G3_ROOT="$G3_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G3_ROOT"])
payload = {
    "schema_version": "actionformer_odfcr_g3_failure_v1",
    "failure_class": "engineering_or_protocol_failure",
    "exit_code": int(os.environ["STATUS"]),
    "stage": os.environ["STAGE"],
    "model_result_claim_allowed": False,
}
temporary = root / "ENGINEERING_FAILURE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "ENGINEERING_FAILURE.json")
PY
  fi
}
trap failure_receipt EXIT
output="$G3_ROOT/ODFCR_G3_AGGREGATE_COMPLETE.json"
(
  cd "$CANDIDATE_ROOT"
  python -m tools.aggregate_odfcr_k384_replays \
    --g2-aggregate "$RUN_ROOT/aggregate_g2/ODFCR_G2_AGGREGATE_COMPLETE.json" \
    --matrix "$RUN_ROOT/seed_2026073101/ODFCR_MATRIX_COMPLETE.json" \
    --matrix "$RUN_ROOT/seed_2026073102/ODFCR_MATRIX_COMPLETE.json" \
    --matrix "$RUN_ROOT/seed_2026073103/ODFCR_MATRIX_COMPLETE.json" \
    --replay "$RUN_ROOT/k384_seed_2026073101/ODFCR_K384_REPLAY_COMPLETE.json" \
    --replay "$RUN_ROOT/k384_seed_2026073102/ODFCR_K384_REPLAY_COMPLETE.json" \
    --replay "$RUN_ROOT/k384_seed_2026073103/ODFCR_K384_REPLAY_COMPLETE.json" \
    --output "$output"
) | tee "$G3_ROOT/g3_aggregate.log"
OUTPUT="$output" python - <<'PY'
import json
import os

record = json.loads(open(os.environ["OUTPUT"], encoding="utf-8").read())
assert record["schema_version"] == "actionformer_odfcr_g3_internal_aggregate_v1"
assert record["validation_pass"] is True
assert isinstance(record["support_gate_pass"], bool)
assert record["paper_performance_row_allowed"] is False
PY
trap - EXIT
rm -f "$G3_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_ODFCR_G3_AGGREGATE_COMPLETE"
