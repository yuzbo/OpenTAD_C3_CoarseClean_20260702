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
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"
R0_SUMMARY="${DUCA_R0_SUMMARY_JSON:-${RUN_ROOT}/r0_holdout_map/r0_summary.json}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
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
"${PYTHON}" - "${R0_SUMMARY}" "${EXPECTED_COMMIT}" \
  "${RUN_ROOT}/r0_headroom_gate.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(source.read_text(encoding="utf-8"))
if payload.get("schema") != "duca_r0_selected_axis_boundary_burst_map_v2":
    raise SystemExit("R0 summary schema mismatch")
if payload.get("ok") is not True or payload.get("git_commit") != sys.argv[2]:
    raise SystemExit("R0 summary did not complete on the exact commit")
if payload.get("source_subset") != "training_internal_holdout":
    raise SystemExit("R0 did not use the sealed training holdout")
if payload.get("test_subset_consumed") is not False:
    raise SystemExit("R0 consumed the test subset")
rows = {row.get("family"): row for row in payload.get("rows", [])}
required = {
    "A_exact_uniform",
    "R2Q3_privileged_boundary_burst",
    "R4Q5_privileged_boundary_burst",
}
if set(rows) != required:
    raise SystemExit("R0 family set mismatch")
values = {}
for family, row in rows.items():
    metrics_path = Path(row.get("metrics_path", "")).expanduser().resolve()
    if not metrics_path.is_file():
        raise SystemExit(f"R0 metrics are missing for {family}")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if digest != row.get("metrics_sha256"):
        raise SystemExit(f"R0 metrics hash drift for {family}")
    value = float(row.get("metrics", {}).get("average_mAP", float("nan")))
    if not math.isfinite(value):
        raise SystemExit(f"R0 average_mAP is missing for {family}")
    values[family] = value
uniform = values["A_exact_uniform"]
best_privileged = max(
    values["R2Q3_privileged_boundary_burst"],
    values["R4Q5_privileged_boundary_burst"],
)
headroom = best_privileged - uniform
required_headroom = float(payload.get("required_headroom_average_mAP", float("nan")))
if not math.isfinite(required_headroom) or required_headroom < 0.20:
    raise SystemExit("R0 required headroom contract is missing or too weak")
if not headroom > required_headroom:
    raise SystemExit(
        "R0 constrained burst Oracle headroom does not clear the frozen threshold: "
        f"headroom={headroom}, required>{required_headroom}"
    )
gate = {
    "schema": "duca_r0_headroom_gate_v1",
    "ok": True,
    "git_commit": sys.argv[2],
    "r0_summary_path": str(source),
    "r0_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "average_mAP": values,
    "best_privileged_minus_uniform_average_mAP": headroom,
    "required_strict_headroom_average_mAP": required_headroom,
    "test_subset_consumed": False,
    "paper_claim_allowed": False,
}
Path(sys.argv[3]).write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
variant_configs=(
  configs/adatad/thumos/duca_gaussian_frontend_pretrain_matched_fixed384.py
  configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py
  configs/adatad/thumos/duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py
)
gate_args=()
for config in "${variant_configs[@]}"; do gate_args+=(--variant-config "${config}"); done
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  --config configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py \
  "${gate_args[@]}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --checkpoint "${ADATAD_PRETRAIN_PATH}" \
  --official-repos-root "${C3_OFFICIAL_ACTION_SEG_REPOS}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split-sha256 "${SPLIT_SHA256}" \
  --output-json "${RUN_ROOT}/p0_real_gate.json"

variants=(gaussian_matched burst_r2q3 burst_r4q5)
for variant in "${variants[@]}"; do
  export DUCA_FRONTEND_VARIANT="${variant}"
  export RUN_DIR="${RUN_ROOT}/p0/${variant}/run"
  export WORK_DIR="${RUN_ROOT}/p0/${variant}/work"
  bash scripts/run_duca_frontend_pretrain_variant_gpu1.sh
done

"${PYTHON}" -m tools.bata.select_duca_boundary_burst_candidates \
  --expected-commit "${EXPECTED_COMMIT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${SPLIT_SHA256}" \
  --receipt "${RUN_ROOT}/p0/gaussian_matched/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/burst_r2q3/run/completion.json" \
  --receipt "${RUN_ROOT}/p0/burst_r4q5/run/completion.json" \
  --output-json "${RUN_ROOT}/frontend_decision.json"

echo "[DUCA_BURST_P0] completed ${RUN_ROOT}/frontend_decision.json"
