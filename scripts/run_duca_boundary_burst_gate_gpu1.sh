#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_BURST_GATE][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
GATE_ROOT="${DUCA_BOUNDARY_BURST_GATE_ROOT:-${RUN_ROOT}/full_model_gate}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
DECISION="${DUCA_FRONTEND_DECISION_JSON:-${RUN_ROOT}/frontend_decision.json}"
DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256:-}"
FROZEN_PRETRAIN_PATH="${DUCA_ADATAD_PRETRAIN_PATH:-}"
FROZEN_PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
"${PYTHON}" - "${ADATAD_PRETRAIN_PATH}" "${FROZEN_PRETRAIN_PATH}" \
  "${FROZEN_PRETRAIN_SHA256}" <<'PY'
import sys
from tools.bata.duca_selected_axis_training import validate_frozen_pretrain_binding

validate_frozen_pretrain_binding(
    runtime_path=sys.argv[1], expected_path=sys.argv[2], expected_sha256=sys.argv[3]
)
PY
[[ -f "${DECISION}" ]] || fail "frontend decision is missing"
[[ "$(sha256sum "${DECISION}" | awk '{print $1}')" == "${DECISION_SHA256}" ]] || fail "decision drift"
"${PYTHON}" - "${DECISION}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import validate_p0_real_gate
import json
from pathlib import Path

decision = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = decision.get("p0_real_gate")
if not isinstance(gate, dict):
    raise SystemExit("frontend decision lacks P0 real gate binding")
validate_p0_real_gate(
    gate_path=gate.get("path", ""),
    gate_sha256=gate.get("sha256", ""),
    expected_commit=sys.argv[2],
)
if gate.get("schema") != "duca_frontend_p0_real_cuda_gate_v1" or gate.get("ok") is not True:
    raise SystemExit("frontend decision P0 real gate fields drifted")
PY
[[ ! -e "${GATE_ROOT}" ]] || fail "fresh gate root is required"

mkdir -p "${GATE_ROOT}/contracts" "${GATE_ROOT}/full_model" "${GATE_ROOT}/tests"
ADATAD_PRETRAIN_SHA256="${FROZEN_PRETRAIN_SHA256}"
entries=(
  "two_stage_exact_uniform:gaussian_matched:configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
  "gaussian_matched_g0:gaussian_matched:configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
  "boundary_burst_r2q3_g0:burst_r2q3:configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
  "boundary_burst_r4q5_g0:burst_r4q5:configs/adatad/thumos/duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
)
for entry in "${entries[@]}"; do
  IFS=: read -r variant frontend config <<<"${entry}"
  config_stem="$(basename "${config}" .py)"
  readarray -t selected < <("${PYTHON}" - "${DECISION}" "${frontend}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
w = p["winners"][sys.argv[2]]
print(w["checkpoint_path"]); print(w["checkpoint_sha256"]); print(int(w["epoch_one_based"]) - 1)
PY
)
  export DUCA_FRONTEND_CHECKPOINT="${selected[0]}"
  export DUCA_FRONTEND_CHECKPOINT_SHA256="${selected[1]}"
  export DUCA_FRONTEND_CHECKPOINT_EPOCH="${selected[2]}"
  "${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
    --config "${config}" --output-json "${GATE_ROOT}/contracts/${variant}.json"
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-burst-${SLURM_JOB_ID}-${variant}" \
    tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
    --config "${config}" --expected-commit "${EXPECTED_COMMIT}" \
    --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
    --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
    --output-json "${GATE_ROOT}/full_model/${config_stem}.json"
done

"${PYTHON}" -m pytest \
  tests/test_duca_transition_only.py \
  tests/test_duca_boundary_burst_selection.py \
  tests/test_duca_selection_quality_analysis.py \
  tests/test_duca_frontend_p0_contract.py \
  tests/test_duca_temporal_sampling_contract.py -q \
  2>&1 | tee "${GATE_ROOT}/tests/focused_pytest.out"

"${PYTHON}" - "${GATE_ROOT}" "${EXPECTED_COMMIT}" "${DECISION}" "${DECISION_SHA256}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
records = [
    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(root.rglob("*")) if path.is_file()
]
payload = {
    "schema": "duca_boundary_burst_full_model_gate_v1",
    "ok": True,
    "formal_training_unlocked": True,
    "status": "uniform_gaussian_r2q3_r4q5_full_model_gate_passed",
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "frontend_decision_path": str(Path(sys.argv[3]).resolve()),
    "frontend_decision_sha256": sys.argv[4],
    "p0_real_gate": json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["p0_real_gate"],
    "artifacts": records,
}
(root / "gate_suite.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[DUCA_BURST_GATE] passed ${GATE_ROOT}/gate_suite.json"
