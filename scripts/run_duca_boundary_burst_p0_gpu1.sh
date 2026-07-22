#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_BURST_P0][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
R0_PRODUCER_COMMIT="${DUCA_R0_PRODUCER_COMMIT:-${EXPECTED_COMMIT}}"
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"
R0_SUMMARY="${DUCA_R0_SUMMARY_JSON:-${RUN_ROOT}/r0_holdout_map/r0_summary.json}"
R0_SUMMARY_SHA256_FILE="${DUCA_R0_SUMMARY_SHA256_FILE:-${RUN_ROOT}/r0_holdout_map/r0_summary.sha256}"
FAMILY_MANIFEST="${DUCA_BOUNDARY_BURST_FAMILY_MANIFEST:-${RUN_ROOT}/family_routing_manifest.json}"
FAMILY_MANIFEST_SHA256_FILE="${DUCA_BOUNDARY_BURST_FAMILY_MANIFEST_SHA256_FILE:-${RUN_ROOT}/family_routing_manifest.sha256}"
FROZEN_PRETRAIN_PATH="${DUCA_ADATAD_PRETRAIN_PATH:-}"
FROZEN_PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "${R0_PRODUCER_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact R0 producer commit is required"
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
[[ -f "${SPLIT_MANIFEST}" ]] || fail "split manifest is missing"
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == "${SPLIT_SHA256}" ]] || fail "split drift"

export DUCA_FRONTEND_TRAIN_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_train_block_list.txt"
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="${RUN_ROOT}/frontend_split/frontend_holdout_block_list.txt"
"${PYTHON}" -m tools.bata.create_duca_frontend_split \
  --validate-manifest "${SPLIT_MANIFEST}" \
  --expected-manifest-sha256 "${SPLIT_SHA256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --train-block-list "${DUCA_FRONTEND_TRAIN_BLOCK_LIST}" \
  --holdout-block-list "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}" \
  > "${RUN_ROOT}/frontend_split.validation.json"

[[ -f "${R0_SUMMARY}" ]] || fail "R0 summary is missing"
[[ -f "${R0_SUMMARY_SHA256_FILE}" ]] || fail "R0 summary SHA256 seal is missing"
IFS= read -r R0_SUMMARY_SHA256 < "${R0_SUMMARY_SHA256_FILE}"
[[ "${R0_SUMMARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid R0 summary SHA256 seal"
"${PYTHON}" - "${R0_SUMMARY}" "${R0_SUMMARY_SHA256}" "${R0_PRODUCER_COMMIT}" \
  "${RUN_ROOT}/r0_headroom_gate.json" <<'PY'
import sys
from pathlib import Path

from tools.bata.select_duca_boundary_burst_candidates import (
    _atomic_write_json,
    validate_r0_headroom_summary,
)

gate = validate_r0_headroom_summary(
    summary_path=sys.argv[1],
    summary_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
)
_atomic_write_json(
    Path(sys.argv[4]).resolve(), gate, require_absent=True
)
PY
"${PYTHON}" - "${R0_SUMMARY}" "${R0_SUMMARY_SHA256}" "${EXPECTED_COMMIT}" \
  "${R0_PRODUCER_COMMIT}" "${FAMILY_MANIFEST}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    create_family_routing_manifest,
)

create_family_routing_manifest(
    summary_path=sys.argv[1],
    summary_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
    r0_expected_commit=sys.argv[4],
    output_path=sys.argv[5],
)
PY
FAMILY_MANIFEST_SHA256="$(sha256sum "${FAMILY_MANIFEST}" | awk '{print $1}')"
[[ "${FAMILY_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid family manifest SHA256"
printf '%s\n' "${FAMILY_MANIFEST_SHA256}" > "${FAMILY_MANIFEST_SHA256_FILE}.tmp"
mv -f "${FAMILY_MANIFEST_SHA256_FILE}.tmp" "${FAMILY_MANIFEST_SHA256_FILE}"
readarray -t selected_route < <("${PYTHON}" - "${FAMILY_MANIFEST}" \
  "${FAMILY_MANIFEST_SHA256}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_family_routing_manifest,
)

manifest = validate_family_routing_manifest(
    manifest_path=sys.argv[1],
    manifest_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
)
routing = manifest["family_routing"]
print(routing["selected_p0_variant"])
print(routing["selected_p0_config"])
print(routing["selected_official60_variant"])
PY
)
SELECTED_P0_VARIANT="${selected_route[0]}"
SELECTED_P0_CONFIG="${selected_route[1]}"
SELECTED_OFFICIAL60_VARIANT="${selected_route[2]}"
[[ -n "${SELECTED_P0_VARIANT}" && -f "${SELECTED_P0_CONFIG}" \
  && -n "${SELECTED_OFFICIAL60_VARIANT}" ]] \
  || fail "R0-selected family route is incomplete"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  --config "${SELECTED_P0_CONFIG}" \
  --variant-config "${SELECTED_P0_CONFIG}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --checkpoint "${ADATAD_PRETRAIN_PATH}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split-sha256 "${SPLIT_SHA256}" \
  --output-json "${RUN_ROOT}/p0_real_gate.json"
P0_REAL_GATE="${RUN_ROOT}/p0_real_gate.json"
P0_REAL_GATE_SHA256="$(sha256sum "${P0_REAL_GATE}" | awk '{print $1}')"
[[ "${P0_REAL_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid P0 real gate SHA256"
P0_ASFORMER_CONSUMER="${RUN_ROOT}/p0_training_asformer_consumer.json"
"${PYTHON}" - "${P0_REAL_GATE}" "${P0_REAL_GATE_SHA256}" \
  "${EXPECTED_COMMIT}" "${SELECTED_P0_CONFIG}" "${P0_ASFORMER_CONSUMER}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    create_p0_training_asformer_consumer_receipt,
)

create_p0_training_asformer_consumer_receipt(
    gate_path=sys.argv[1],
    gate_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
    selected_config_path=sys.argv[4],
    output_path=sys.argv[5],
)
PY
P0_ASFORMER_CONSUMER_SHA256="$(sha256sum "${P0_ASFORMER_CONSUMER}" | awk '{print $1}')"
[[ "${P0_ASFORMER_CONSUMER_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid P0 training ASFormer consumer SHA256"

export DUCA_FRONTEND_VARIANT="${SELECTED_P0_VARIANT}"
export RUN_DIR="${RUN_ROOT}/p0/${SELECTED_P0_VARIANT}/run"
export WORK_DIR="${RUN_ROOT}/p0/${SELECTED_P0_VARIANT}/work"
bash scripts/run_duca_frontend_pretrain_variant_gpu1.sh

"${PYTHON}" -m tools.bata.select_duca_boundary_burst_candidates \
  --expected-commit "${EXPECTED_COMMIT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${SPLIT_SHA256}" \
  --family-manifest "${FAMILY_MANIFEST}" \
  --family-manifest-sha256 "${FAMILY_MANIFEST_SHA256}" \
  --p0-real-gate "${P0_REAL_GATE}" \
  --p0-real-gate-sha256 "${P0_REAL_GATE_SHA256}" \
  --p0-asformer-consumer "${P0_ASFORMER_CONSUMER}" \
  --p0-asformer-consumer-sha256 "${P0_ASFORMER_CONSUMER_SHA256}" \
  --receipt "${RUN_ROOT}/p0/${SELECTED_P0_VARIANT}/run/completion.json" \
  --output-json "${RUN_ROOT}/frontend_decision.json"
sha256sum "${RUN_ROOT}/frontend_decision.json" | awk '{print $1}' > \
  "${RUN_ROOT}/frontend_decision.sha256.tmp"
mv -f "${RUN_ROOT}/frontend_decision.sha256.tmp" \
  "${RUN_ROOT}/frontend_decision.sha256"

echo "[DUCA_BURST_P0] completed ${RUN_ROOT}/frontend_decision.json"
