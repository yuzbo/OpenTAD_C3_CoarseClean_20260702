#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_ASSETS][FAIL] $*" >&2
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
  DUCA_RIME_PHASE3_ASSET_ROOT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_PHASE2_CROSSFIT_SUMMARY \
  DUCA_RIME_PHASE2_CROSSFIT_SUMMARY_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_CANDIDATE_BUDGETS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-3 asset production must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE3_ASSET_ROOT}" ]] \
  || fail "a fresh Phase-3 asset root is required"
check_sha256 \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "Phase-2 receipt"
check_sha256 \
  "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
  "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  "K384 budget protocol"
check_sha256 \
  "${DUCA_RIME_PHASE2_CROSSFIT_SUMMARY}" \
  "${DUCA_RIME_PHASE2_CROSSFIT_SUMMARY_SHA256}" \
  "cross-fit summary"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
[[ -f "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" ]] \
  || fail "mixed-K data config is missing"

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
[[ "${#split_values[@]}" == 4 ]] || fail "failed to resolve Phase-3 split roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"
export DUCA_RIME_EVAL_FIXED_BUDGET=384

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
[[ "${candidate_budgets[*]}" == "192 256 384 512" ]] \
  || fail "formal Phase-3 assets require budgets 192,256,384,512"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
assert cfg.workflow.formal_protocol == "duca_rime_phase2_mixed_k_baseline_v1"
assert cfg.duca_rime_variant.arm == "U-mixed-K"
assert cfg.workflow.expected_successful_optimizer_updates == 6000
PY
  echo "[DUCA_RIME_PHASE3_ASSETS] PRECHECK PASS"
  exit 0
fi

python tools/bata/produce_duca_rime_phase3_assets.py \
  --config "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --phase2-receipt "${DUCA_RIME_PHASE2_RECEIPT}" \
  --phase2-receipt-sha256 "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  --budget-protocol "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
  --budget-protocol-sha256 "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  --crossfit-summary "${DUCA_RIME_PHASE2_CROSSFIT_SUMMARY}" \
  --crossfit-summary-sha256 "${DUCA_RIME_PHASE2_CROSSFIT_SUMMARY_SHA256}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --output-root "${DUCA_RIME_PHASE3_ASSET_ROOT}" \
  --candidate-budgets "${candidate_budgets[@]}" \
  --epochs 60 \
  --seed 3407

python tools/bata/create_duca_rime_training_exposure.py \
  --repo-root "${DUCA_RIME_REPO_ROOT}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --config "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --output "${DUCA_RIME_PHASE3_ASSET_ROOT}/training_exposure.json" \
  --research-phase 3 \
  --seed 3407 \
  --detector-backend ActionFormer \
  --target-mean-cost 384 \
  > "${DUCA_RIME_PHASE3_ASSET_ROOT}/training_exposure.receipt.json"

python - "${DUCA_RIME_PHASE3_ASSET_ROOT}" "${DUCA_RIME_EXPECTED_COMMIT}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
summary = json.loads((root / "producer_summary.json").read_text(encoding="utf-8"))
required = {
    "training_targets": "training_targets.jsonl",
    "dshuffle_training_replay": "dshuffle_training_replay.jsonl",
    "adaptok_replay": "adaptok_replay.jsonl",
}
artifacts = {}
for name, filename in required.items():
    path = root / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if summary["artifacts"][name]["sha256"] != digest:
        raise SystemExit(f"Phase-3 asset summary drifted for {name}")
    artifacts[name] = {"path": str(path), "sha256": digest}
exposure = root / "training_exposure.json"
artifacts["training_exposure"] = {
    "path": str(exposure),
    "sha256": hashlib.sha256(exposure.read_bytes()).hexdigest(),
}
payload = {
    "schema_version": "duca_rime_phase3_asset_receipt_v1",
    "status": "passed",
    "git_commit": sys.argv[2],
    "slurm_job_id": sys.argv[3],
    "producer_summary": {
        "path": str(root / "producer_summary.json"),
        "sha256": hashlib.sha256((root / "producer_summary.json").read_bytes()).hexdigest(),
    },
    "artifacts": artifacts,
    "official_final_subset_consumed": False,
    "claim_scope": "training_and_development_control_assets_only_no_model_result",
}
target = root / "production_receipt.json"
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
sha256sum "${DUCA_RIME_PHASE3_ASSET_ROOT}/production_receipt.json" \
  > "${DUCA_RIME_PHASE3_ASSET_ROOT}/production_receipt.sha256"

echo "[DUCA_RIME_PHASE3_ASSETS] PASS ${DUCA_RIME_PHASE3_ASSET_ROOT}"
