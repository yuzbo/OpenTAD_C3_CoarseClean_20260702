#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_SELECTED_OPT_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
GATE_ROOT="${DUCA_SELECTED_OPT_GATE_ROOT:-}"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${GATE_ROOT}" && ! -e "${GATE_ROOT}" ]] || fail "fresh external gate root required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] || fail "official ASFormer source is missing"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

mkdir -p "${GATE_ROOT}/contracts" "${GATE_ROOT}/tests" "${GATE_ROOT}/full_model"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"

configs=(
  configs/adatad/thumos/duca_exact_uniform_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_direct025_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_homotopy025_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py
)
for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  "${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
    --config "${config}" \
    --output-json "${GATE_ROOT}/contracts/${name}.json"
done

"${PYTHON}" -m pytest \
  tests/test_duca_online_frame_selector_contracts.py \
  tests/test_duca_selected_axis_optimization_configs.py \
  tests/test_duca_detector_gradient_bridge.py \
  tests/test_duca_hard_soft_alignment.py \
  tests/test_duca_temporal_sampling_contract.py \
  -q 2>&1 | tee "${GATE_ROOT}/tests/focused_pytest.out"

full_model_configs=(
  configs/adatad/thumos/duca_exact_uniform_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_direct025_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_homotopy025_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py
)
for config in "${full_model_configs[@]}"; do
  name="$(basename "${config}" .py)"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-selected-opt-${SLURM_JOB_ID}-${name}" \
    tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
    --config "${config}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
    --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
    --output-json "${GATE_ROOT}/full_model/${name}.json"
done

"${PYTHON}" - "${GATE_ROOT}" "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
records = []
for path in sorted(path for path in root.rglob("*") if path.is_file()):
    records.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema": "duca_selected_axis_optimization_gate_v1",
    "ok": True,
    "status": "four_matched_variants_full_model_gate_passed",
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "formal_training_unlocked": True,
    "artifacts": records,
}
(root / "gate_suite.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[DUCA_SELECTED_OPT_GATE] passed: ${GATE_ROOT}/gate_suite.json"
