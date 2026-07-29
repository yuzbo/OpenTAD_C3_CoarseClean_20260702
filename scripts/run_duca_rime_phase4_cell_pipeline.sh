#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_PIPELINE][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

[[ "${DUCA_RIME_ENABLE_PHASE4:-0}" == 1 ]] \
  || fail "Phase-4 and official-final remain sealed pending an explicit authorized release"

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_EXPECTED_BRANCH \
  DUCA_RIME_ACQUISITION_ADMISSION \
  DUCA_RIME_ACQUISITION_ADMISSION_SHA256 \
  DUCA_RIME_PHASE4_BACKEND \
  DUCA_RIME_PHASE4_TARGET \
  DUCA_RIME_PHASE4_SEED \
  DUCA_RIME_PHASE4_CELL_ROOT \
  DUCA_RIME_PHASE4_AUTHORIZATION \
  DUCA_RIME_PHASE4_AUTHORIZATION_SHA256 \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_PHASE2_PROTOCOL_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-4 cell pipeline must run inside Slurm"
[[ "${DUCA_RIME_PHASE4_BACKEND}" == ActionFormer || "${DUCA_RIME_PHASE4_BACKEND}" == TriDet ]] \
  || fail "backend must be ActionFormer or TriDet"
[[ "${DUCA_RIME_PHASE4_TARGET}" == 384 || "${DUCA_RIME_PHASE4_TARGET}" == 192 ]] \
  || fail "formal target must be 384 or 192"
[[ "${DUCA_RIME_PHASE4_SEED}" == 5801 || "${DUCA_RIME_PHASE4_SEED}" == 8123 || "${DUCA_RIME_PHASE4_SEED}" == 12011 ]] \
  || fail "formal seed is not registered"
[[ ! -e "${DUCA_RIME_PHASE4_CELL_ROOT}" ]] \
  || fail "a fresh final cell root is required"
for sibling in \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.training_exposure.json" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.training_exposure.json.receipt.json" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.rime_train" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.fixed_train" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.rime_eval" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.fixed_eval" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.same_k_eval" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.same_k_replay" \
  "${DUCA_RIME_PHASE4_CELL_ROOT}.cost"; do
  [[ ! -e "${sibling}" ]] \
    || fail "stale Phase-4 sibling output is forbidden: ${sibling}"
done

cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_ACQUISITION_ADMISSION}" \
  "${DUCA_RIME_ACQUISITION_ADMISSION_SHA256}" \
  "acquisition admission-v2"
python -m tools.bata.verify_duca_acquisition_admission_v2 \
  --receipt "${DUCA_RIME_ACQUISITION_ADMISSION}" \
  --expected-sha256 "${DUCA_RIME_ACQUISITION_ADMISSION_SHA256}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --repo-root "${DUCA_RIME_REPO_ROOT}" \
  --expected-branch "${DUCA_RIME_EXPECTED_BRANCH}" \
  --expected-artifact "split_assignment=${DUCA_RIME_SPLIT_MANIFEST}"
check_sha256 \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" \
  "Phase-4 authorization"
check_sha256 \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "Phase-2 receipt"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
check_sha256 \
  "${DUCA_RIME_TARGETS_JSONL}" \
  "${DUCA_RIME_TARGETS_SHA256}" \
  "cross-fitted targets"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE pretrain"

protocol="${DUCA_RIME_PHASE2_PROTOCOL_ROOT}/budget_protocol_k${DUCA_RIME_PHASE4_TARGET}.json"
[[ -f "${protocol}" ]] || fail "frozen budget protocol is missing: ${protocol}"
protocol_sha="$(sha256sum "${protocol}" | awk '{print $1}')"
python - \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "${protocol}" \
  "${protocol_sha}" \
  "${DUCA_RIME_PHASE4_TARGET}" <<'PY'
import json
import os
import sys

authorization = json.load(open(sys.argv[1], encoding="utf-8"))
phase2 = json.load(open(sys.argv[2], encoding="utf-8"))
phase2_sha, protocol_path, protocol_sha, target = sys.argv[3:]
target = float(target)
authorized_phase2 = authorization.get("phase2_receipt")
authorized_protocols = authorization.get("formal_budget_protocols")
phase2_protocols = phase2.get("formal_budget_protocols")
matches = (
    [
        row
        for row in authorized_protocols
        if isinstance(row, dict)
        and float(row.get("target_mean_cost", float("nan"))) == target
    ]
    if isinstance(authorized_protocols, list)
    else []
)
if (
    not isinstance(authorized_phase2, dict)
    or os.path.abspath(str(authorized_phase2.get("path", "")))
    != os.path.abspath(sys.argv[2])
    or authorized_phase2.get("sha256") != phase2_sha
    or authorized_protocols != phase2_protocols
    or len(matches) != 1
    or os.path.abspath(str(matches[0].get("path", "")))
    != os.path.abspath(protocol_path)
    or matches[0].get("sha256") != protocol_sha
):
    raise SystemExit(
        "Phase-4 authorization/Phase-2 receipt/budget protocol binding mismatch"
    )
PY
export DUCA_RIME_BUDGET_PROTOCOL_JSON="${protocol}"
export DUCA_RIME_BUDGET_PROTOCOL_SHA256="${protocol_sha}"
export DUCA_RIME_TARGET_MEAN_COST="${DUCA_RIME_PHASE4_TARGET}"
export DUCA_RIME_FIXED_BUDGET="${DUCA_RIME_PHASE4_TARGET}"

readarray -t split_values < <(python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    row = manifest["train_roles"][role]
    print(row["block_list_path"])
    print(row["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] || fail "failed to resolve train/development roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"

if [[ "${DUCA_RIME_PHASE4_BACKEND}" == ActionFormer ]]; then
  rime_arm="RIME-full"
  fixed_arm="U-fixed"
  same_arm="U-same-K"
  rime_train_config="configs/adatad/thumos/duca_rime_full_total60.py"
  fixed_train_config="configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py"
  rime_eval_config="configs/adatad/thumos/duca_rime_full_formal_validation.py"
  fixed_eval_config="configs/adatad/thumos/duca_rime_uniform_fixed_formal_validation.py"
  same_eval_config="configs/adatad/thumos/duca_rime_uniform_same_k_formal_validation.py"
  required DUCA_RIME_DENSE_CONFIG_ACTIONFORMER
  required DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER
  required DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER
  required DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER
  required DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER
  dense_config="${DUCA_RIME_DENSE_CONFIG_ACTIONFORMER}"
  dense_checkpoint="${DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER}"
  dense_evidence="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER}"
  dense_evidence_sha="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER}"
  dense_commit="${DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER}"
else
  rime_arm="RIME-full-TriDet"
  fixed_arm="U-fixed-TriDet"
  same_arm="U-same-K-TriDet"
  rime_train_config="configs/adatad/thumos/duca_rime_full_tridet_total60.py"
  fixed_train_config="configs/adatad/thumos/duca_rime_uniform_fixed_tridet_total60.py"
  rime_eval_config="configs/adatad/thumos/duca_rime_full_tridet_formal_validation.py"
  fixed_eval_config="configs/adatad/thumos/duca_rime_uniform_fixed_tridet_formal_validation.py"
  same_eval_config="configs/adatad/thumos/duca_rime_uniform_same_k_tridet_formal_validation.py"
  required DUCA_RIME_DENSE_CONFIG_TRIDET
  required DUCA_RIME_DENSE_CHECKPOINT_TRIDET
  required DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET
  required DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_TRIDET
  required DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET
  dense_config="${DUCA_RIME_DENSE_CONFIG_TRIDET}"
  dense_checkpoint="${DUCA_RIME_DENSE_CHECKPOINT_TRIDET}"
  dense_evidence="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET}"
  dense_evidence_sha="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_TRIDET}"
  dense_commit="${DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET}"
fi

for path in \
  "${rime_train_config}" \
  "${fixed_train_config}" \
  "${rime_eval_config}" \
  "${fixed_eval_config}" \
  "${same_eval_config}" \
  "${dense_config}" \
  "${dense_checkpoint}"; do
  [[ -f "${path}" ]] || fail "required Phase-4 artifact is missing: ${path}"
done
check_sha256 "${dense_evidence}" "${dense_evidence_sha}" "dense checkpoint evidence"

cell_parent="$(dirname "${DUCA_RIME_PHASE4_CELL_ROOT}")"
mkdir -p "${cell_parent}"
exposure="${DUCA_RIME_PHASE4_CELL_ROOT}.training_exposure.json"
python tools/bata/create_duca_rime_training_exposure.py \
  --repo-root "${DUCA_RIME_REPO_ROOT}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --config "${rime_train_config}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --output "${exposure}" \
  --research-phase 4 \
  --seed "${DUCA_RIME_PHASE4_SEED}" \
  --detector-backend "${DUCA_RIME_PHASE4_BACKEND}" \
  --target-mean-cost "${DUCA_RIME_PHASE4_TARGET}" \
  > "${exposure}.receipt.json"
export DUCA_RIME_TRAINING_EXPOSURE_JSON="${exposure}"
export DUCA_RIME_TRAINING_EXPOSURE_SHA256="$(
  sha256sum "${exposure}" | awk '{print $1}'
)"

rime_train_root="${DUCA_RIME_PHASE4_CELL_ROOT}.rime_train"
fixed_train_root="${DUCA_RIME_PHASE4_CELL_ROOT}.fixed_train"

export DUCA_RIME_PHASE4_ARM="${rime_arm}"
export DUCA_RIME_PHASE4_CONFIG="${rime_train_config}"
export DUCA_RIME_PHASE4_ROOT="${rime_train_root}"
scripts/run_duca_rime_phase4_train_cell.sh

export DUCA_RIME_PHASE4_ARM="${fixed_arm}"
export DUCA_RIME_PHASE4_CONFIG="${fixed_train_config}"
export DUCA_RIME_PHASE4_ROOT="${fixed_train_root}"
scripts/run_duca_rime_phase4_train_cell.sh

rime_receipt="${rime_train_root}/training_receipt.json"
rime_checkpoint="${rime_train_root}/train/gpu1_id0/checkpoint/terminal_ema.pth"
fixed_receipt="${fixed_train_root}/training_receipt.json"
fixed_checkpoint="${fixed_train_root}/train/gpu1_id0/checkpoint/terminal_ema.pth"
for path in "${rime_receipt}" "${rime_checkpoint}" "${fixed_receipt}" "${fixed_checkpoint}"; do
  [[ -f "${path}" ]] || fail "Phase-4 training output is missing: ${path}"
done

run_eval() {
  local arm="$1" config="$2" root="$3" receipt="$4" checkpoint="$5"
  export DUCA_RIME_EVAL_PHASE=4
  export DUCA_RIME_EVAL_ARM="${arm}"
  export DUCA_RIME_EVAL_CONFIG="${config}"
  export DUCA_RIME_EVAL_ROOT="${root}"
  export DUCA_RIME_EVAL_SEED="${DUCA_RIME_PHASE4_SEED}"
  export DUCA_RIME_TRAINING_RECEIPT="${receipt}"
  export DUCA_RIME_TRAINING_RECEIPT_SHA256="$(
    sha256sum "${receipt}" | awk '{print $1}'
  )"
  export DUCA_RIME_CHECKPOINT="${checkpoint}"
  export DUCA_RIME_CHECKPOINT_SHA256="$(
    sha256sum "${checkpoint}" | awk '{print $1}'
  )"
  scripts/run_duca_rime_evaluate_arm.sh
}

rime_eval_root="${DUCA_RIME_PHASE4_CELL_ROOT}.rime_eval"
fixed_eval_root="${DUCA_RIME_PHASE4_CELL_ROOT}.fixed_eval"
same_eval_root="${DUCA_RIME_PHASE4_CELL_ROOT}.same_k_eval"
run_eval "${rime_arm}" "${rime_eval_config}" "${rime_eval_root}" \
  "${rime_receipt}" "${rime_checkpoint}"
run_eval "${fixed_arm}" "${fixed_eval_config}" "${fixed_eval_root}" \
  "${fixed_receipt}" "${fixed_checkpoint}"
same_replay_root="${DUCA_RIME_PHASE4_CELL_ROOT}.same_k_replay"
mkdir -p "${same_replay_root}"
IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
python tools/bata/build_duca_rime_budget_replay.py \
  --mode paired \
  --input-jsonl "${rime_eval_root}/inference_ledger.jsonl" \
  --output-jsonl "${same_replay_root}/paired_replay.jsonl" \
  --candidate-budgets "${candidate_budgets[@]}"
export DUCA_RIME_REPLAY_JSONL="${same_replay_root}/paired_replay.jsonl"
export DUCA_RIME_REPLAY_SHA256="$(
  sha256sum "${DUCA_RIME_REPLAY_JSONL}" | awk '{print $1}'
)"
run_eval "${same_arm}" "${same_eval_config}" "${same_eval_root}" \
  "${rime_receipt}" "${rime_checkpoint}"

cost_root="${DUCA_RIME_PHASE4_CELL_ROOT}.cost"
export DUCA_RIME_COST_PHASE=4
export DUCA_RIME_COST_ARM="${rime_arm}"
export DUCA_RIME_COST_BACKEND="${DUCA_RIME_PHASE4_BACKEND}"
export DUCA_RIME_COST_TARGET="${DUCA_RIME_PHASE4_TARGET}"
export DUCA_RIME_COST_SEED="${DUCA_RIME_PHASE4_SEED}"
export DUCA_RIME_COST_ROOT="${cost_root}"
export DUCA_RIME_CANDIDATE_CONFIG="${rime_eval_config}"
export DUCA_RIME_CANDIDATE_CHECKPOINT="${rime_checkpoint}"
export DUCA_RIME_CANDIDATE_CHECKPOINT_SHA256="$(
  sha256sum "${rime_checkpoint}" | awk '{print $1}'
)"
export DUCA_RIME_CANDIDATE_TRAINING_RECEIPT="${rime_receipt}"
export DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256="$(
  sha256sum "${rime_receipt}" | awk '{print $1}'
)"
export DUCA_RIME_FIXED_CONFIG="${same_eval_config}"
export DUCA_RIME_FIXED_CHECKPOINT="${rime_checkpoint}"
export DUCA_RIME_FIXED_CHECKPOINT_SHA256="$(
  sha256sum "${rime_checkpoint}" | awk '{print $1}'
)"
export DUCA_RIME_FIXED_TRAINING_RECEIPT="${rime_receipt}"
export DUCA_RIME_FIXED_TRAINING_RECEIPT_SHA256="$(
  sha256sum "${rime_receipt}" | awk '{print $1}'
)"
export DUCA_RIME_DENSE_CONFIG="${dense_config}"
export DUCA_RIME_DENSE_CHECKPOINT="${dense_checkpoint}"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE="${dense_evidence}"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256="${dense_evidence_sha}"
export DUCA_RIME_DENSE_TRAINED_COMMIT="${dense_commit}"
scripts/run_duca_rime_cost_cell.sh

export DUCA_RIME_PHASE4_RIME_EVAL_ROOT="${rime_eval_root}"
export DUCA_RIME_PHASE4_FIXED_EVAL_ROOT="${fixed_eval_root}"
export DUCA_RIME_PHASE4_SAME_K_EVAL_ROOT="${same_eval_root}"
export DUCA_RIME_PHASE4_COST_EVIDENCE="${cost_root}/paired_cost.json"
export DUCA_RIME_PHASE4_COST_EVIDENCE_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE4_COST_EVIDENCE}" | awk '{print $1}'
)"
scripts/run_duca_rime_phase4_seal_cell.sh

echo "[DUCA_RIME_PHASE4_PIPELINE] PASS ${DUCA_RIME_PHASE4_CELL_ROOT}/cell_result.json"
