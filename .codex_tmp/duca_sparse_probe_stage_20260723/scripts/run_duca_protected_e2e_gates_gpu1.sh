#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_E2E_GATES][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
[[ -f "${REPO_ROOT}/tools/bata/validate_duca_protected_e2e_official60.py" ]] \
  || fail "DUCA_REPO_ROOT does not identify the exact repository snapshot"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
GATE_ROOT="${DUCA_PROTECTED_E2E_GATE_ROOT:-}"
CHECKPOINT="${DUCA_PROTECTED_E2E_AUDIT_CHECKPOINT:-}"
CHECKPOINT_SHA256="${DUCA_PROTECTED_E2E_AUDIT_CHECKPOINT_SHA256:-}"
CHECKPOINT_SOURCE_COMMIT="${DUCA_PROTECTED_E2E_AUDIT_CHECKPOINT_SOURCE_COMMIT:-}"
CHECKPOINT_EVIDENCE="${DUCA_PROTECTED_E2E_AUDIT_CHECKPOINT_EVIDENCE:-}"
CHECKPOINT_EVIDENCE_SHA256="${DUCA_PROTECTED_E2E_AUDIT_CHECKPOINT_EVIDENCE_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "gates must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "DUCA_EXPECTED_COMMIT must be exact"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${GATE_ROOT}" && ! -e "${GATE_ROOT}" ]] || fail "fresh external gate root required"
[[ -f "${CHECKPOINT}" ]] || fail "trained audit checkpoint is missing"
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA256 is invalid"
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] \
  || fail "checkpoint hash drift"
[[ "${CHECKPOINT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "checkpoint source commit is invalid"
[[ -f "${CHECKPOINT_EVIDENCE}" ]] || fail "checkpoint evidence is missing"
[[ "${CHECKPOINT_EVIDENCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "checkpoint evidence SHA256 is invalid"
[[ "$(sha256sum "${CHECKPOINT_EVIDENCE}" | awk '{print $1}')" == "${CHECKPOINT_EVIDENCE_SHA256}" ]] \
  || fail "checkpoint evidence hash drift"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] \
  || fail "official ASFormer source is missing"

mkdir -p "${GATE_ROOT}/contracts" "${GATE_ROOT}/tests" "${GATE_ROOT}/gradients" "${GATE_ROOT}/alignment"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"

configs=(
  configs/adatad/thumos/duca_exact_uniform_fixed384_official60.py
  configs/adatad/thumos/duca_transition_no_bridge_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py
  configs/adatad/thumos/duca_protected_e2e_rho_fixed384_official60.py
)
for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  "${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
    --config "${config}" \
    --output-json "${GATE_ROOT}/contracts/${name}.json"
done

"${PYTHON}" -m pytest \
  tests/test_duca_official_asformer_hidden.py \
  tests/test_duca_transition_only.py \
  tests/test_duca_detector_gradient_bridge.py \
  tests/test_duca_hard_soft_alignment.py \
  tests/test_duca_temporal_sampling_contract.py \
  -q 2>&1 | tee "${GATE_ROOT}/tests/focused_pytest.out"

"${PYTHON}" tools/bata/run_duca_protected_e2e_gradient_gate.py \
  --config configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py \
  --device cuda \
  --output-json "${GATE_ROOT}/gradients/protected_main.json"

"${PYTHON}" tools/bata/run_duca_protected_e2e_gradient_gate.py \
  --config configs/adatad/thumos/duca_protected_e2e_rho_fixed384_official60.py \
  --device cuda \
  --output-json "${GATE_ROOT}/gradients/protected_rho.json"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-protected-main-${SLURM_JOB_ID}" \
  tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
  --config configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
  --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
  --output-json "${GATE_ROOT}/gradients/exact_protected_main.json"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-protected-rho-${SLURM_JOB_ID}" \
  tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
  --config configs/adatad/thumos/duca_protected_e2e_rho_fixed384_official60.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
  --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
  --output-json "${GATE_ROOT}/gradients/exact_protected_rho.json"

"${PYTHON}" tools/bata/run_duca_protected_e2e_hard_soft_gate.py \
  --config configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --checkpoint-source-commit "${CHECKPOINT_SOURCE_COMMIT}" \
  --checkpoint-evidence "${CHECKPOINT_EVIDENCE}" \
  --checkpoint-evidence-sha256 "${CHECKPOINT_EVIDENCE_SHA256}" \
  --adatad-pretrain "${ADATAD_PRETRAIN_PATH}" \
  --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
  --real-batches 4 \
  --candidates-per-batch 8 \
  --bootstrap-samples 2000 \
  --output-json "${GATE_ROOT}/alignment/protected_main.json"

"${PYTHON}" - "${GATE_ROOT}" "${EXPECTED_COMMIT}" "${CHECKPOINT_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
checkpoint_sha256 = sys.argv[3]
paths = sorted(path for path in root.rglob("*") if path.is_file())
records = []
for path in paths:
    records.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
alignment = json.loads((root / "alignment" / "protected_main.json").read_text(encoding="utf-8"))
if alignment.get("ok") is not True:
    raise SystemExit("P3 alignment did not pass")
payload = {
    "schema": "duca_protected_e2e_gate_suite_v1",
    "ok": True,
    "status": "p0_p1_p2_p3_passed_official60_submission_unlocked",
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "audit_checkpoint_sha256": checkpoint_sha256,
    "artifacts": records,
}
(root / "gate_suite.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[DUCA_PROTECTED_E2E_GATES] P0-P3 passed: ${GATE_ROOT}/gate_suite.json"
