#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH -J af-k384-gate

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${AUDIT_ROOT:?missing AUDIT_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${OFFICIAL_CHECKPOINT:?missing OFFICIAL_CHECKPOINT}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_AUDIT_COMMIT:?missing EXPECTED_AUDIT_COMMIT}"
: "${EXPECTED_AUDIT_TREE:?missing EXPECTED_AUDIT_TREE}"
: "${SOURCE_DIFF_ATTESTATION:?missing SOURCE_DIFF_ATTESTATION}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?missing EXPECTED_SOURCE_DIFF_SHA256}"

case "$CANDIDATE_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*) ;;
  *) echo "candidate root is outside the remote write boundary" >&2; exit 2 ;;
esac
case "$AUDIT_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*) ;;
  *) echo "audit root is outside the remote write boundary" >&2; exit 2 ;;
esac
case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/phystime_tad/runs/*) ;;
  *) echo "run root is outside the remote write boundary" >&2; exit 2 ;;
esac

test -d "$RUN_ROOT"
test ! -e "$RUN_ROOT/GATE_COMPLETE.json"
test ! -e "$RUN_ROOT/CUDA_GATE.json"
test -n "${CUDA_VISIBLE_DEVICES:-}"

RUN_LOG="$RUN_ROOT/gate_runtime.log"
FAILURE_RECEIPT="$RUN_ROOT/ENGINEERING_FAILURE.json"
exec > >(tee -a "$RUN_LOG") 2>&1

failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    RUN_ROOT="$RUN_ROOT" STATUS="$status" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema_version": "actionformer_native_grid_sparse_head_failure_v1",
    "failure_class": "engineering_or_gate_failure",
    "exit_code": int(os.environ["STATUS"]),
    "model_metric_claim_allowed": False,
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

candidate_commit=$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)
candidate_tree=$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')
candidate_branch=$(git -C "$CANDIDATE_ROOT" rev-parse --abbrev-ref HEAD)
audit_commit=$(git -C "$AUDIT_ROOT" rev-parse HEAD)
audit_tree=$(git -C "$AUDIT_ROOT" rev-parse 'HEAD^{tree}')
audit_branch=$(git -C "$AUDIT_ROOT" rev-parse --abbrev-ref HEAD)
test "$candidate_commit" = "$EXPECTED_CANDIDATE_COMMIT"
test "$candidate_tree" = "$EXPECTED_CANDIDATE_TREE"
test "$audit_commit" = "$EXPECTED_AUDIT_COMMIT"
test "$audit_tree" = "$EXPECTED_AUDIT_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$AUDIT_ROOT" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SOURCE_DIFF_ATTESTATION" | awk '{print $1}')" = \
  "$EXPECTED_SOURCE_DIFF_SHA256"

python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(
    {
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices_preserved": True,
    }
)
PY

(
  cd "$CANDIDATE_ROOT"
  python -m pytest \
    tests/test_native_grid_sparse_heads.py \
    tests/test_sparsehead_official_config.py \
    -q
) | tee "$RUN_ROOT/candidate_focused_tests.log"

python \
  "$AUDIT_ROOT/tools/bata/validate_actionformer_native_grid_sparse_head_cuda.py" \
  --source-root "$CANDIDATE_ROOT" \
  --config \
    "$CANDIDATE_ROOT/configs/thumos_i3d_sparsehead_k384_uniform.yaml" \
  --checkpoint "$OFFICIAL_CHECKPOINT" \
  --output "$RUN_ROOT/CUDA_GATE.json" \
  --expected-commit "$EXPECTED_CANDIDATE_COMMIT" \
  --expected-tree "$EXPECTED_CANDIDATE_TREE" \
  --budget 384 \
  --warmup 5 \
  --repeats 30 \
  --timing-rounds 3

gate_sha256=$(sha256sum "$RUN_ROOT/CUDA_GATE.json" | awk '{print $1}')
source_diff_sha256=$(sha256sum "$SOURCE_DIFF_ATTESTATION" | awk '{print $1}')
run_log_sha256=$(sha256sum "$RUN_LOG" | awk '{print $1}')
export RUN_ROOT gate_sha256 source_diff_sha256 run_log_sha256
export candidate_commit candidate_tree candidate_branch
export audit_commit audit_tree audit_branch
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
gate = json.loads((root / "CUDA_GATE.json").read_text(encoding="utf-8"))
if gate.get("gate_pass") is not True:
    raise SystemExit("CUDA gate did not pass")
payload = {
    "schema_version": "actionformer_native_grid_sparse_head_gate_completion_v1",
    "validation_pass": True,
    "gate_pass": True,
    "candidate": {
        "branch": os.environ["candidate_branch"],
        "commit": os.environ["candidate_commit"],
        "tree": os.environ["candidate_tree"],
        "clean": True,
    },
    "audit": {
        "branch": os.environ["audit_branch"],
        "commit": os.environ["audit_commit"],
        "tree": os.environ["audit_tree"],
        "clean": True,
    },
    "receipts": {
        "cuda_gate_sha256": os.environ["gate_sha256"],
        "source_diff_attestation_sha256": os.environ["source_diff_sha256"],
        "runtime_log_sha256_before_completion": os.environ["run_log_sha256"],
    },
    "claim_boundary": "isolated_head_cuda_gate_only",
    "paper_metric_claim_allowed": False,
    "end_to_end_wall_clock_claim_allowed": False,
}
temporary = root / "GATE_COMPLETE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "GATE_COMPLETE.json")
PY

trap - EXIT
rm -f "$FAILURE_RECEIPT"
echo "ACTIONFORMER_NATIVE_GRID_K384_CUDA_GATE_COMPLETE"
