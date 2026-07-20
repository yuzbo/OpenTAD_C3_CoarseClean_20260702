#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_FULL_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_protected_physical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
ARM="${DUCA_PROTECTED_GATE_ARM:-}"
PROTOCOL_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
PROTOCOL_SHA256="${DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256:-}"
OUTPUT_JSON="${DUCA_PROTECTED_GATE_OUTPUT_JSON:-}"
case "${ARM}" in
  protected_e2e)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_fixed384_official60.py"
    ;;
  protected_e2e_rho001)
    CONFIG="configs/adatad/thumos/duca_protected_physical_e2e_rho001_fixed384_official60.py"
    ;;
  *)
    fail "arm must be protected_e2e or protected_e2e_rho001"
    ;;
esac

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PROTOCOL_JSON}" ]] || fail "P0 protocol manifest is missing"
[[ "${PROTOCOL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "P0 SHA256 is invalid"
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] || fail "P0 hash drift"
[[ -n "${OUTPUT_JSON}" && ! -e "${OUTPUT_JSON}" ]] || fail "fresh gate output is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one logical GPU is required"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

PRETRAIN_SHA256="$(sha256sum "${DUCA_PROTECTED_ADATAD_PRETRAIN}" | awk '{print $1}')"
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-protected-${SLURM_JOB_ID}-${ARM}" \
  tools/bata/run_duca_protected_physical_full_model_gate.py \
  --config "${CONFIG}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --protocol-manifest "${PROTOCOL_JSON}" \
  --protocol-manifest-sha256 "${PROTOCOL_SHA256}" \
  --adatad-pretrain "${DUCA_PROTECTED_ADATAD_PRETRAIN}" \
  --adatad-pretrain-sha256 "${PRETRAIN_SHA256}" \
  --output-json "${OUTPUT_JSON}"
sha256sum "${OUTPUT_JSON}" > "${OUTPUT_JSON}.sha256"
