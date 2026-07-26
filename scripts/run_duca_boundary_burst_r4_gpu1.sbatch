#!/usr/bin/env bash
#SBATCH --job-name=duca_r4
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

set -euo pipefail

fail() { echo "[DUCA_BURST_R4][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TERMINAL_SUITE="${DUCA_BOUNDARY_BURST_TERMINAL_SUITE:-}"
TERMINAL_SUITE_SHA256="${DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256:-}"
R4_ROOT="${DUCA_BOUNDARY_BURST_R4_ROOT:-}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${TERMINAL_SUITE}" ]] || fail "sealed terminal U/G0 suite is required"
[[ "${TERMINAL_SUITE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "terminal suite SHA256 is invalid"
[[ "$(sha256sum "${TERMINAL_SUITE}" | awk '{print $1}')" == "${TERMINAL_SUITE_SHA256}" ]] \
  || fail "terminal suite content drift"
R4_ROOT="$(duca_cellcf_require_external_path \
  DUCA_BOUNDARY_BURST_R4_ROOT "${REPO_ROOT}" "${BASE}" "${R4_ROOT}")" \
  || fail "R4 output root violates the external path contract"
[[ ! -e "${R4_ROOT}" ]] || fail "fresh R4 output root is required"
mkdir -p "${R4_ROOT}"

export DUCA_BOUNDARY_BURST_ALIGNMENT_ROOT="${R4_ROOT}/alignment"
bash scripts/run_duca_boundary_burst_hard_swap_alignment_gpu1.sh

ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_ROOT}/alignment.json"
IFS= read -r ALIGNMENT_SHA256 < "${ALIGNMENT_JSON}.sha256"
"${PYTHON}" -m tools.bata.duca_boundary_burst_hard_swap_alignment validate \
  --expected-commit "${EXPECTED_COMMIT}" \
  --artifact "${ALIGNMENT_JSON}" --artifact-sha256 "${ALIGNMENT_SHA256}"

run_stage() {
  local stage="$1"
  local route variant
  readarray -t route < <("${PYTHON}" - "${ALIGNMENT_JSON}" \
    "${ALIGNMENT_SHA256}" "${EXPECTED_COMMIT}" "${stage}" <<'PY'
import json
import sys
from pathlib import Path

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    load_alignment_context,
    validate_alignment_artifact,
)

artifact_path, artifact_sha256, commit, stage = sys.argv[1:]
validate_alignment_artifact(
    path=artifact_path, digest=artifact_sha256, expected_commit=commit
)
artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
matches = [
    (variant, record)
    for variant, record in artifact["authorized_variants"].items()
    if record["stage"] == stage
]
if len(matches) != 1:
    raise SystemExit(f"alignment artifact does not authorize exactly one {stage}")
variant, record = matches[0]
validate_alignment_artifact(
    path=artifact_path,
    digest=artifact_sha256,
    expected_commit=commit,
    expected_variant=variant,
    source_config_path=record["config_path"],
    source_config_sha256=record["config_sha256"],
)
context, _ = load_alignment_context(
    path=artifact["context"]["path"],
    digest=artifact["context"]["sha256"],
    expected_commit=commit,
)
print(variant)
print(context["frontend_decision"]["path"])
print(context["frontend_decision"]["sha256"])
print(context["gate_suite"]["path"])
print(context["gate_suite"]["sha256"])
print(context["adatad_pretrain"]["path"])
print(context["adatad_pretrain"]["sha256"])
PY
  )
  [[ "${#route[@]}" == 7 ]] || fail "${stage} sealed route is incomplete"
  variant="${route[0]}"
  export DUCA_SELECTED_OPT_VARIANT="${variant}"
  export DUCA_FRONTEND_DECISION_JSON="${route[1]}"
  export DUCA_FRONTEND_DECISION_SHA256="${route[2]}"
  export DUCA_SELECTED_OPT_GATE_SUITE="${route[3]}"
  export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="${route[4]}"
  export DUCA_ADATAD_PRETRAIN_PATH="${route[5]}"
  export DUCA_ADATAD_PRETRAIN_SHA256="${route[6]}"
  export DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${ALIGNMENT_JSON}"
  export DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${ALIGNMENT_SHA256}"
  export RUN_DIR="${R4_ROOT}/official60/${variant}/run"
  export WORK_DIR="${R4_ROOT}/official60/${variant}/work"
  bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
  sha256sum "${RUN_DIR}/completion.json" | awk '{print $1}' \
    > "${RUN_DIR}/completion.json.sha256.tmp"
  mv -f "${RUN_DIR}/completion.json.sha256.tmp" \
    "${RUN_DIR}/completion.json.sha256"
}

run_stage G1
run_stage G2

echo "[DUCA_BURST_R4] alignment=${ALIGNMENT_JSON}"
echo "[DUCA_BURST_R4] G1/G2 completions=${R4_ROOT}/official60/*/run/completion.json"
