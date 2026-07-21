#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
DECISION="${DUCA_FRONTEND_DECISION_JSON:-}"
DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256:-}"
GATE_ROOT="${DUCA_TWO_STAGE_GATE_ROOT:-}"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${DECISION}" ]] || fail "frontend decision is missing"
[[ "$(sha256sum "${DECISION}" | awk '{print $1}')" == "${DECISION_SHA256}" ]] \
  || fail "frontend decision hash drift"
[[ -n "${GATE_ROOT}" && ! -e "${GATE_ROOT}" ]] || fail "fresh gate root is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

readarray -t winner < <("${PYTHON}" - "${DECISION}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("ok") is not True or payload.get("status") != "GO_TO_MATCHED_OFFICIAL60":
    raise SystemExit("frontend mechanism gate did not authorize official-60")
manifest = json.loads(Path(payload["candidate_manifest_path"]).read_text(encoding="utf-8"))
if manifest.get("git_commit") != sys.argv[2] or manifest.get("test_subset_consumed") is not False:
    raise SystemExit("frontend candidate manifest commit/split mismatch")
winner = payload["winner"]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
)
export DUCA_FRONTEND_CHECKPOINT="${winner[0]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${winner[1]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${winner[2]}"
[[ -f "${DUCA_FRONTEND_CHECKPOINT}" ]] || fail "selected frontend checkpoint is missing"
[[ "$(sha256sum "${DUCA_FRONTEND_CHECKPOINT}" | awk '{print $1}')" == "${DUCA_FRONTEND_CHECKPOINT_SHA256}" ]] \
  || fail "selected frontend checkpoint hash drift"

mkdir -p "${GATE_ROOT}/contracts" "${GATE_ROOT}/tests" "${GATE_ROOT}/full_model"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"
configs=(
  configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py
  configs/adatad/thumos/duca_two_stage_scratch_fixed384_official60.py
  configs/adatad/thumos/duca_two_stage_pretrained_joint_fixed384_official60.py
  configs/adatad/thumos/duca_two_stage_pretrained_frozen_fixed384_official60.py
)
for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  "${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
    --config "${config}" \
    --output-json "${GATE_ROOT}/contracts/${name}.json"
done

"${PYTHON}" -m pytest \
  tests/test_duca_two_stage_curriculum.py \
  tests/test_duca_frontend_checkpoint_selection.py \
  tests/test_duca_online_frame_selector_contracts.py \
  tests/test_duca_selected_axis_optimization_configs.py \
  tests/test_duca_detector_gradient_bridge.py \
  tests/test_duca_hard_soft_alignment.py \
  tests/test_duca_temporal_sampling_contract.py \
  -q 2>&1 | tee "${GATE_ROOT}/tests/focused_pytest.out"

for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-two-stage-${SLURM_JOB_ID}-${name}" \
    tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
    --config "${config}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
    --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
    --output-json "${GATE_ROOT}/full_model/${name}.json"
done

"${PYTHON}" - "${GATE_ROOT}" "${EXPECTED_COMMIT}" "${DECISION}" "${DECISION_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
records = []
for path in sorted(path for path in root.rglob("*") if path.is_file()):
    records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
payload = {
    "schema": "duca_selected_axis_optimization_gate_v1",
    "ok": True,
    "status": "two_stage_four_arm_full_model_gate_passed",
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "formal_training_unlocked": True,
    "frontend_decision_path": str(Path(sys.argv[3]).resolve()),
    "frontend_decision_sha256": sys.argv[4],
    "artifacts": records,
}
(root / "gate_suite.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "[DUCA_TWO_STAGE_GATE] passed ${GATE_ROOT}/gate_suite.json"
