#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_EVIDENCE][FAIL] $*" >&2
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
  DUCA_RIME_PHASE2_EVIDENCE_ROOT \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_PHASE1_RECEIPT \
  DUCA_RIME_PHASE1_RECEIPT_SHA256 \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_CANDIDATE_COSTS \
  DUCA_RIME_TARGET_MEAN_COST \
  DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST \
  DUCA_RIME_DECODER_FAMILY \
  DUCA_RIME_RISK_WEIGHT \
  DUCA_RIME_RISK_THRESHOLD \
  DUCA_RIME_O4_MAX_BRIER \
  DUCA_RIME_O4_MAX_ECE \
  DUCA_RIME_O4_MIN_COVERAGE \
  DUCA_RIME_O4_MAX_LOW_RISK_FAILURE \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-2 evidence production must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ ! -e "${DUCA_RIME_PHASE2_EVIDENCE_ROOT}" ]] \
  || fail "a fresh Phase-2 evidence root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
for binding in \
  "${DUCA_RIME_PHASE1_RECEIPT}|${DUCA_RIME_PHASE1_RECEIPT_SHA256}|Phase-1 receipt" \
  "${DUCA_RIME_TRAINING_RECEIPT}|${DUCA_RIME_TRAINING_RECEIPT_SHA256}|mixed-K training receipt" \
  "${DUCA_RIME_CHECKPOINT}|${DUCA_RIME_CHECKPOINT_SHA256}|mixed-K checkpoint" \
  "${DUCA_RIME_SPLIT_MANIFEST}|${DUCA_RIME_SPLIT_MANIFEST_SHA256}|split manifest" \
  "${DUCA_RIME_PRETRAIN_PATH}|${DUCA_RIME_PRETRAIN_SHA256}|VideoMAE pretrain"; do
  IFS='|' read -r path expected label <<<"${binding}"
  check_sha256 "${path}" "${expected}" "${label}"
done

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
IFS=',' read -r -a candidate_costs <<<"${DUCA_RIME_CANDIDATE_COSTS}"
[[ "${candidate_budgets[*]}" == "192 256 384 512" ]] \
  || fail "formal Phase-2 evidence requires K={192,256,384,512}"
[[ "${#candidate_costs[@]}" == "${#candidate_budgets[@]}" ]] \
  || fail "candidate budget/cost cardinality drift"

readarray -t split_values < <(
  python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    row = manifest["train_roles"][role]
    print(row["block_list_path"])
    print(row["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] \
  || fail "failed to resolve Phase-2 train/development roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
    "${DUCA_RIME_TARGET_MEAN_COST}" \
    "${DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST}" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
assert cfg.workflow.formal_protocol == "duca_rime_phase2_mixed_k_baseline_v1"
assert tuple(cfg.duca_rime_variant.candidate_budgets) == (192, 256, 384, 512)
assert float(cfg.duca_rime_variant.training_target_mean_cost) == 384.0
assert [float(value) for value in sys.argv[2:]] == [384.0, 192.0]
assert os.environ["DUCA_RIME_CANDIDATE_BUDGETS"] == "192,256,384,512"
PY
  echo "[DUCA_RIME_PHASE2_EVIDENCE] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE2_EVIDENCE_ROOT}"
o1_root="${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/o1"
mkdir -p "${o1_root}"
o1_manifest_args=()
for index in "${!candidate_budgets[@]}"; do
  budget="${candidate_budgets[${index}]}"
  cost="${candidate_costs[${index}]}"
  eval_root="${o1_root}/k${budget}"
  export DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT="${eval_root}"
  export DUCA_RIME_PHASE2_SPLIT_ROLE=certification_development
  export DUCA_RIME_EVAL_FIXED_BUDGET="${budget}"
  export DUCA_RIME_EVAL_SEED="${DUCA_RIME_PHASE2_SEED:-3407}"
  scripts/run_duca_rime_phase2_mixed_k_eval.sh
  metrics="${eval_root}/localization_metrics.json"
  o1_manifest_args+=(
    --evaluation
    "${budget}"
    "${cost}"
    "${metrics}"
    "$(sha256sum "${metrics}" | awk '{print $1}')"
  )
done

python tools/bata/build_duca_rime_source_manifest.py o1 \
  "${o1_manifest_args[@]}" \
  --mixed-k-detector-identity-sha256 "${DUCA_RIME_CHECKPOINT_SHA256}" \
  --detector-training-exposure mixed_k_registered_panel \
  --training-receipt "${DUCA_RIME_TRAINING_RECEIPT}" \
  --training-receipt-sha256 "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  --output "${o1_root}/source_manifest.json" \
  > "${o1_root}/source_manifest.stdout.json"
python tools/bata/build_duca_rime_gate_records.py o1 \
  --source-manifest "${o1_root}/source_manifest.json" \
  --output "${o1_root}/o1_records.jsonl" \
  --score-metric avg_map \
  > "${o1_root}/o1_records.summary.json"

export DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT="${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/counterfactual"
scripts/run_duca_rime_phase2_counterfactual_measurements.sh
export DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS="${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}/counterfactual_measurements.jsonl"
export DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256="$(
  sha256sum "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" | awk '{print $1}'
)"

export DUCA_RIME_PHASE2_CROSSFIT_ROOT="${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/crossfit"
scripts/run_duca_rime_phase2_crossfit_producer.sh
export DUCA_RIME_CROSSFIT_SUMMARY="${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/producer_summary.json"
export DUCA_RIME_CROSSFIT_SUMMARY_SHA256="$(
  sha256sum "${DUCA_RIME_CROSSFIT_SUMMARY}" | awk '{print $1}'
)"

export DUCA_RIME_PHASE2_O2_ROOT="${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/o2"
scripts/run_duca_rime_phase2_o2_panel.sh

export DUCA_RIME_PHASE2_ROOT="${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/seal"
export DUCA_RIME_O1_RECORDS="${o1_root}/o1_records.jsonl"
export DUCA_RIME_O1_RECORDS_SHA256="$(
  sha256sum "${DUCA_RIME_O1_RECORDS}" | awk '{print $1}'
)"
export DUCA_RIME_O2_RECORDS="${DUCA_RIME_PHASE2_O2_ROOT}/o2_records.jsonl"
export DUCA_RIME_O2_RECORDS_SHA256="$(
  sha256sum "${DUCA_RIME_O2_RECORDS}" | awk '{print $1}'
)"
export DUCA_RIME_O3_RECORDS="${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/o3_records.jsonl"
export DUCA_RIME_O3_RECORDS_SHA256="$(
  sha256sum "${DUCA_RIME_O3_RECORDS}" | awk '{print $1}'
)"
export DUCA_RIME_O4_RECORDS="${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/o4_records.jsonl"
export DUCA_RIME_O4_RECORDS_SHA256="$(
  sha256sum "${DUCA_RIME_O4_RECORDS}" | awk '{print $1}'
)"
export DUCA_RIME_PRICE_CALIBRATION="${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/price_records.jsonl"
export DUCA_RIME_PRICE_CALIBRATION_SHA256="$(
  sha256sum "${DUCA_RIME_PRICE_CALIBRATION}" | awk '{print $1}'
)"
scripts/run_duca_rime_phase2_gates.sh

python - \
  "${DUCA_RIME_PHASE2_EVIDENCE_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
required = {
    "o1_records": root / "o1" / "o1_records.jsonl",
    "counterfactual_measurements": root / "counterfactual" / "counterfactual_measurements.jsonl",
    "crossfit_summary": root / "crossfit" / "producer_summary.json",
    "o2_records": root / "o2" / "o2_records.jsonl",
    "phase2_receipt": root / "seal" / "phase2_receipt.json",
}
artifacts = {}
for name, path in required.items():
    if not path.is_file():
        raise SystemExit(f"missing Phase-2 terminal artifact: {path}")
    artifacts[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
payload = {
    "schema_version": "duca_rime_phase2_evidence_pipeline_receipt_v1",
    "status": "passed",
    "git_commit": sys.argv[2],
    "slurm_job_id": sys.argv[3],
    "mixed_k_checkpoint_sha256": sys.argv[4],
    "artifacts": artifacts,
    "uses_official_final": False,
}
target = root / "pipeline_receipt.json"
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
sha256sum "${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/pipeline_receipt.json" \
  > "${DUCA_RIME_PHASE2_EVIDENCE_ROOT}/pipeline_receipt.sha256"
echo "[DUCA_RIME_PHASE2_EVIDENCE] PASS ${DUCA_RIME_PHASE2_EVIDENCE_ROOT}"
