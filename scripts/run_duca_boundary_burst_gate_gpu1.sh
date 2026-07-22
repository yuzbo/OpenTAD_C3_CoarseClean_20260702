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
readarray -t selected_route < <("${PYTHON}" - "${DECISION}" \
  "${DECISION_SHA256}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
)

decision = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
)
routing = decision["family_routing"]
print(routing["selected_p0_variant"])
print(routing["selected_official60_variant"])
print(routing["selected_official60_config"])
print(routing["uniform_official60_config"])
PY
)
SELECTED_P0_VARIANT="${selected_route[0]}"
SELECTED_OFFICIAL60_VARIANT="${selected_route[1]}"
SELECTED_OFFICIAL60_CONFIG="${selected_route[2]}"
UNIFORM_OFFICIAL60_CONFIG="${selected_route[3]}"
[[ -n "${SELECTED_P0_VARIANT}" && -n "${SELECTED_OFFICIAL60_VARIANT}" \
  && -f "${SELECTED_OFFICIAL60_CONFIG}" && -f "${UNIFORM_OFFICIAL60_CONFIG}" ]] \
  || fail "selected family route is incomplete"
[[ ! -e "${GATE_ROOT}" ]] || fail "fresh gate root is required"

mkdir -p "${GATE_ROOT}/contracts" "${GATE_ROOT}/full_model" "${GATE_ROOT}/tests"
ADATAD_PRETRAIN_SHA256="${FROZEN_PRETRAIN_SHA256}"
entries=(
  "two_stage_exact_uniform:not_applicable:${UNIFORM_OFFICIAL60_CONFIG}"
  "${SELECTED_OFFICIAL60_VARIANT}:${SELECTED_P0_VARIANT}:${SELECTED_OFFICIAL60_CONFIG}"
)
for entry in "${entries[@]}"; do
  IFS=: read -r variant frontend config <<<"${entry}"
  config_stem="$(basename "${config}" .py)"
  if [[ "${frontend}" == "not_applicable" ]]; then
    unset DUCA_FRONTEND_CHECKPOINT DUCA_FRONTEND_CHECKPOINT_SHA256 \
      DUCA_FRONTEND_CHECKPOINT_EPOCH
  else
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
  fi
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
import hashlib, json, os, sys
from pathlib import Path
from uuid import uuid4

from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
)

root = Path(sys.argv[1]).resolve()
decision_path = Path(sys.argv[3]).resolve()
decision = validate_frontend_decision(
    decision_path=decision_path,
    decision_sha256=sys.argv[4],
    expected_commit=sys.argv[2],
)
routing = decision["family_routing"]
records = [
    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(root.rglob("*")) if path.is_file()
]
payload = {
    "schema": "duca_boundary_burst_full_model_gate_v1",
    "ok": True,
    "fail_closed": True,
    "formal_training_unlocked": True,
    "status": "matched_u_selected_g0_full_model_gate_passed",
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "frontend_decision_path": str(decision_path),
    "frontend_decision_sha256": sys.argv[4],
    "family_manifest": decision["family_manifest"],
    "r0_headroom_gate": decision["r0_headroom_gate"],
    "family_routing": routing,
    "p0_real_gate": decision["p0_real_gate"],
    "p0_training_asformer_consumer": decision["p0_training_asformer_consumer"],
    "gated_variants": routing["required_official60_variants"],
    "required_official60_variants": routing["required_official60_variants"],
    "diagnostic_official60_variants": routing["diagnostic_official60_variants"],
    "artifacts": records,
}
output = root / "gate_suite.json"
temporary = root / f".gate_suite.json.{uuid4().hex}.tmp"
try:
    with temporary.open("xb") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
PY

GATE_SUITE_SHA256="$(sha256sum "${GATE_ROOT}/gate_suite.json" | awk '{print $1}')"
"${PYTHON}" - "${GATE_ROOT}/gate_suite.json" "${GATE_SUITE_SHA256}" \
  "${DECISION}" "${DECISION_SHA256}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import validate_full_model_gate

validate_full_model_gate(
    gate_path=sys.argv[1],
    gate_sha256=sys.argv[2],
    decision_path=sys.argv[3],
    decision_sha256=sys.argv[4],
    expected_commit=sys.argv[5],
)
PY

echo "[DUCA_BURST_GATE] passed ${GATE_ROOT}/gate_suite.json"
