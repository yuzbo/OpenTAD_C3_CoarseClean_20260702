#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_O2_PANEL][FAIL] $*" >&2
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

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_PHASE2_O2_ROOT \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS \
  DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256 \
  DUCA_RIME_CROSSFIT_SUMMARY \
  DUCA_RIME_CROSSFIT_SUMMARY_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "formal O2 production must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "a complete exact Git worktree is required"
[[ ! -e "${DUCA_RIME_PHASE2_O2_ROOT}" ]] \
  || fail "a fresh O2 output root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_TRAINING_RECEIPT}" \
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  "mixed-K training receipt"
check_sha256 \
  "${DUCA_RIME_CHECKPOINT}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" \
  "mixed-K terminal checkpoint"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"
check_sha256 \
  "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" \
  "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256}" \
  "counterfactual measurements"
check_sha256 \
  "${DUCA_RIME_CROSSFIT_SUMMARY}" \
  "${DUCA_RIME_CROSSFIT_SUMMARY_SHA256}" \
  "cross-fit summary"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_TRAINING_RECEIPT}" \
    "${DUCA_RIME_CHECKPOINT_SHA256}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import json
import sys

from tools.bata.create_duca_rime_splits import validate_rime_splits

validation = validate_rime_splits(sys.argv[1])
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
receipt = json.load(open(sys.argv[2], encoding="utf-8"))
if (
    receipt.get("schema_version")
    != "duca_rime_phase2_mixed_k_training_receipt_v1"
    or receipt.get("status") != "passed"
    or receipt.get("checkpoint_sha256") != sys.argv[3]
    or receipt.get("git_commit") != sys.argv[4]
    or receipt.get("detector_training_exposure")
    != "mixed_k_registered_panel"
    or int(receipt.get("successful_detector_updates", -1)) != 6000
    or receipt.get("uses_official_final") is not False
):
    raise SystemExit("invalid O2 mixed-K training receipt")
row = manifest["train_roles"]["certification_development"]
print(row["block_list_path"])
print(row["block_list_sha256"])
print(validation["assignment_sha256"])
PY
)
[[ "${#split_values[@]}" == 3 ]] \
  || fail "failed to resolve certification_development"
check_sha256 "${split_values[0]}" "${split_values[1]}" \
  "certification_development block list"

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
[[ "${candidate_budgets[*]}" == "192 256 384 512" ]] \
  || fail "formal O2 requires K={192,256,384,512}"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[0]}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
    "${DUCA_RIME_CROSSFIT_SUMMARY}" <<'PY'
import json
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
summary = json.load(open(sys.argv[2], encoding="utf-8"))
assert cfg.workflow.formal_protocol == "duca_rime_phase2_mixed_k_baseline_v1"
assert tuple(cfg.duca_rime_variant.candidate_budgets) == (192, 256, 384, 512)
assert cfg.duca_rime_contract.pad_to_kmax is False
assert summary["models"]["o2_decoder"]["runtime_decoder_api"] == "decode_rime_panel"
assert summary["models"]["o2_decoder"]["eval_role"] == "certification_development"
PY
  echo "[DUCA_RIME_PHASE2_O2_PANEL] PRECHECK PASS"
  exit 0
fi

python tools/bata/produce_duca_rime_o2_panel.py \
  --config "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --checkpoint "${DUCA_RIME_CHECKPOINT}" \
  --checkpoint-sha256 "${DUCA_RIME_CHECKPOINT_SHA256}" \
  --training-receipt "${DUCA_RIME_TRAINING_RECEIPT}" \
  --training-receipt-sha256 "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --measurements-jsonl "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" \
  --measurements-sha256 "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256}" \
  --crossfit-summary "${DUCA_RIME_CROSSFIT_SUMMARY}" \
  --crossfit-summary-sha256 "${DUCA_RIME_CROSSFIT_SUMMARY_SHA256}" \
  --output-root "${DUCA_RIME_PHASE2_O2_ROOT}" \
  --candidate-budgets "${candidate_budgets[@]}" \
  --weak-overlap-fraction "${DUCA_RIME_WEAK_OVERLAP_FRACTION:-0.50}" \
  --seed "${DUCA_RIME_PHASE2_SEED:-3407}" \
  --device cuda:0 \
  --num-workers "${DUCA_RIME_O2_WORKERS:-2}" \
  --backbone-pretrain "${DUCA_RIME_PRETRAIN_PATH}"

manifest_args=()
for family in independent strict_nested weak_overlap; do
  for budget in "${candidate_budgets[@]}"; do
    metrics="${DUCA_RIME_PHASE2_O2_ROOT}/${family}/k${budget}.metrics.json"
    ledger="${DUCA_RIME_PHASE2_O2_ROOT}/${family}/k${budget}.ledger.jsonl"
    manifest_args+=(
      --evaluation
      "${family}"
      "${budget}"
      "${metrics}"
      "$(sha256sum "${metrics}" | awk '{print $1}')"
      "${ledger}"
      "$(sha256sum "${ledger}" | awk '{print $1}')"
    )
  done
done

python tools/bata/build_duca_rime_source_manifest.py o2 \
  "${manifest_args[@]}" \
  --mixed-k-detector-identity-sha256 "${DUCA_RIME_CHECKPOINT_SHA256}" \
  --training-receipt "${DUCA_RIME_TRAINING_RECEIPT}" \
  --training-receipt-sha256 "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  --crossfit-summary "${DUCA_RIME_CROSSFIT_SUMMARY}" \
  --crossfit-summary-sha256 "${DUCA_RIME_CROSSFIT_SUMMARY_SHA256}" \
  --output "${DUCA_RIME_PHASE2_O2_ROOT}/source_manifest.json" \
  > "${DUCA_RIME_PHASE2_O2_ROOT}/source_manifest.stdout.json"

python tools/bata/build_duca_rime_gate_records.py o2 \
  --source-manifest "${DUCA_RIME_PHASE2_O2_ROOT}/source_manifest.json" \
  --output "${DUCA_RIME_PHASE2_O2_ROOT}/o2_records.jsonl" \
  --score-metric counterfactual_negative_detector_loss \
  > "${DUCA_RIME_PHASE2_O2_ROOT}/o2_records.summary.json"

if [[ -n "${DUCA_RIME_O2_MAX_REGRET:-}" ]]; then
  python tools/bata/duca_rime_phase2.py o2 \
    --records-jsonl "${DUCA_RIME_PHASE2_O2_ROOT}/o2_records.jsonl" \
    --output "${DUCA_RIME_PHASE2_O2_ROOT}/o2_gate.json" \
    --selected-family "${DUCA_RIME_DECODER_FAMILY:-weak_overlap}" \
    --max-regret "${DUCA_RIME_O2_MAX_REGRET}" \
    --bootstrap-samples "${DUCA_RIME_BOOTSTRAP_SAMPLES:-5000}" \
    --seed "${DUCA_RIME_PHASE2_SEED:-3407}"
fi

python - \
  "${DUCA_RIME_PHASE2_O2_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${split_values[2]}" <<'PY'
import hashlib
import json
import os
import sys

root, commit, job_id, assignment_sha256 = sys.argv[1:]
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
summary = os.path.join(root, "producer_summary.json")
manifest = os.path.join(root, "source_manifest.json")
records = os.path.join(root, "o2_records.jsonl")
payload = {
    "schema_version": "duca_rime_phase2_o2_runtime_receipt_v1",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": job_id,
    "split_assignment_sha256": assignment_sha256,
    "producer_summary_path": os.path.abspath(summary),
    "producer_summary_sha256": sha(summary),
    "source_manifest_path": os.path.abspath(manifest),
    "source_manifest_sha256": sha(manifest),
    "records_path": os.path.abspath(records),
    "records_sha256": sha(records),
    "runtime_decoder_api": "decode_rime_panel",
    "score_metric": "counterfactual_negative_detector_loss",
    "uses_official_final": False,
    "claim_scope": (
        "measured_detector_objective_decoder_family_regret_"
        "not_tad_map_not_localization_quality"
    ),
}
output = os.path.join(root, "production_receipt.json")
temporary = output + f".partial.{os.getpid()}"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

echo "[DUCA_RIME_PHASE2_O2_PANEL] PASS ${DUCA_RIME_PHASE2_O2_ROOT}"
