#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_VALIDATION_EXPORT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export BASE
source scripts/duca_cellcf_canonical_env.sh
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_ALLOCATION_CHECKPOINT:-}"
PRETRAIN="${ADATAD_PRETRAIN_PATH:-}"
OUTPUT_ROOT="${DUCA_ALLOCATION_VALIDATION_ROOT:-}"
EXPECTED_EPOCH="${DUCA_ALLOCATION_CHECKPOINT_EPOCH:-131}"
CONFIG="configs/adatad/thumos/duca_allocation_ceiling_validation_windows.py"
REPLAY_CONFIG="configs/adatad/thumos/duca_allocation_ceiling_physical_grid_replay.py"
GO_JSON="${DUCA_ALLOCATION_VALIDATION_GO_JSON:-}"
GO_SHA256="${DUCA_ALLOCATION_VALIDATION_GO_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "validation export requires a Slurm GPU"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] \
  || fail "validation export requires cluster n16r4"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "validation export requires exactly one Slurm-visible GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "validation export requires a clean exact-commit checkout"
[[ -f "${CHECKPOINT}" && -f "${PRETRAIN}" ]] || fail "checkpoint or pretrain is missing"
[[ -f "${GO_JSON}" && "${GO_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "validation export requires a hashed GO receipt"
[[ "$(sha256sum "${GO_JSON}" | awk '{print $1}')" == "${GO_SHA256}" ]] \
  || fail "validation GO receipt SHA-256 mismatch"
[[ "${EXPECTED_EPOCH}" =~ ^[0-9]+$ ]] || fail "checkpoint epoch is invalid"
[[ -n "${OUTPUT_ROOT}" && "${OUTPUT_ROOT}" == "${BASE}/"* ]] \
  || fail "validation output must stay under BASE"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite validation output"

readarray -t GO_BINDING < <(
  "${PYTHON}" - "${GO_JSON}" "${EXPECTED_COMMIT}" "${CHECKPOINT}" "${PRETRAIN}" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if receipt.get("schema_version") != "duca_allocation_validation_authorization_v1":
    raise SystemExit("validation GO receipt schema mismatch")
if receipt.get("decision") != "GO" or receipt.get("git_commit") != sys.argv[2]:
    raise SystemExit("validation GO receipt decision/commit mismatch")
contract = receipt.get("contract", {})
if contract != {
    "single_use": True,
    "authorizes_validation_export": True,
    "authorizes_validation_replay": True,
    "authorizes_selector_training": False,
    "authorizes_paper_claim": False,
}:
    raise SystemExit("validation GO receipt contract mismatch")
evidence_path = pathlib.Path(receipt["training_suite_evidence_json"]).resolve()
digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
if digest != receipt.get("training_suite_evidence_json_sha256"):
    raise SystemExit("training-suite evidence hash mismatch")
checkpoint_sha = hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
if checkpoint_sha != receipt.get("checkpoint_sha256"):
    raise SystemExit("validation GO receipt checkpoint mismatch")
pretrain_sha = hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest()
if pretrain_sha != receipt.get("pretrain_sha256"):
    raise SystemExit("validation GO receipt pretrain mismatch")
print(evidence_path)
print(digest)
PY
)
[[ "${#GO_BINDING[@]}" == "2" ]] || fail "validation GO binding is incomplete"
GO_EVIDENCE_PATH="${GO_BINDING[0]}"
GO_EVIDENCE_SHA256="${GO_BINDING[1]}"
mkdir "${GO_JSON}.consumed" \
  || fail "validation GO receipt was already consumed"
mkdir -p "${OUTPUT_ROOT}"

INPUT_JSONL="${OUTPUT_ROOT}/validation_inputs.jsonl"
INPUT_SUMMARY="${OUTPUT_ROOT}/validation_inputs.summary.json"
CEILING_JSONL="${OUTPUT_ROOT}/validation_deploy_families.jsonl"
CEILING_SUMMARY="${OUTPUT_ROOT}/validation_deploy_families.summary.json"

"${PYTHON}" -m tools.bata.export_duca_allocation_ceiling_inputs \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-jsonl "${INPUT_JSONL}" \
  --summary-json "${INPUT_SUMMARY}" \
  --split test \
  --requested-budget 384 \
  --device cuda:0 \
  --use-ema true \
  --batch-size 1 \
  --num-workers 2 \
  --coordinate-tolerance-frames 0 \
  --validation-authorized

"${PYTHON}" - "${INPUT_SUMMARY}" "${EXPECTED_COMMIT}" "${EXPECTED_EPOCH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
source = payload.get("source", {})
if source.get("git_commit") != sys.argv[2] or source.get("git_clean") is not True:
    raise SystemExit("validation export commit binding is invalid")
if source.get("checkpoint_state_key") != "state_dict_ema":
    raise SystemExit("validation export did not use state_dict_ema")
if int(source.get("checkpoint_epoch", -1)) != int(sys.argv[3]):
    raise SystemExit("validation export checkpoint epoch mismatch")
if source.get("split") != "test":
    raise SystemExit("validation export split mismatch")
PY

"${PYTHON}" -m tools.bata.diagnose_duca_allocation_family_ceiling \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${CEILING_JSONL}" \
  --summary-json "${CEILING_SUMMARY}" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --gt-families none \
  --quantization-scale 1000000

"${PYTHON}" -m tools.bata.validate_duca_allocation_ceiling_artifact \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${CEILING_JSONL}" \
  --summary-json "${CEILING_SUMMARY}" \
  --validation-json "${OUTPUT_ROOT}/validation_deploy_families.validation.json"

"${PYTHON}" - \
  "${OUTPUT_ROOT}/manifest.json" \
  "${INPUT_SUMMARY}" \
  "${EXPECTED_COMMIT}" \
  "${SLURM_CLUSTER_NAME}" \
  "${CHECKPOINT}" \
  "${EXPECTED_EPOCH}" \
  "${PRETRAIN}" \
  "${CONFIG}" \
  "${REPLAY_CONFIG}" \
  "${GO_JSON}" \
  "${GO_SHA256}" \
  "${GO_EVIDENCE_PATH}" \
  "${GO_EVIDENCE_SHA256}" \
  "${INPUT_JSONL}" \
  "${CEILING_JSONL}" \
  "${CEILING_SUMMARY}" \
  "${OUTPUT_ROOT}/validation_deploy_families.validation.json" <<'PY'
import json
import pathlib
import sys

from tools.bata.export_duca_allocation_ceiling_inputs import (
    data_directory_provenance,
    sha256,
    write_json_exclusive,
)

(
    output_text,
    input_summary_text,
    commit,
    cluster,
    checkpoint_text,
    epoch_text,
    pretrain_text,
    export_config_text,
    replay_config_text,
    go_text,
    go_sha,
    evidence_text,
    evidence_sha,
    input_text,
    ceiling_text,
    ceiling_summary_text,
    ceiling_validation_text,
) = sys.argv[1:]
input_summary = json.load(open(input_summary_text, encoding="utf-8"))
source = input_summary.get("source")
if not isinstance(source, dict):
    raise SystemExit("validation export summary source is missing")
current_data = data_directory_provenance(source["data_path"])
for key, value in current_data.items():
    if source.get(key) != value:
        raise SystemExit(f"validation dataset bytes changed: {key}")
checkpoint = pathlib.Path(checkpoint_text).resolve()
pretrain = pathlib.Path(pretrain_text).resolve()
export_config = pathlib.Path(export_config_text).resolve()
replay_config = pathlib.Path(replay_config_text).resolve()
input_path = pathlib.Path(input_text).resolve()
ceiling = pathlib.Path(ceiling_text).resolve()
ceiling_summary = pathlib.Path(ceiling_summary_text).resolve()
ceiling_validation = pathlib.Path(ceiling_validation_text).resolve()
if (
    pathlib.Path(source.get("config", "")).resolve() != export_config
    or source.get("config_sha256") != sha256(export_config)
    or pathlib.Path(source.get("checkpoint", "")).resolve() != checkpoint
    or source.get("checkpoint_sha256") != sha256(checkpoint)
    or source.get("split") != "test"
):
    raise SystemExit("validation export source/config/checkpoint binding mismatch")
payload = {
    "schema_version": "duca_allocation_validation_export_manifest_v2",
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "execution_cluster": cluster,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "checkpoint_epoch": int(epoch_text),
    "checkpoint_state_key": "state_dict_ema",
    "pretrain": str(pretrain),
    "pretrain_sha256": sha256(pretrain),
    "export_config": str(export_config),
    "export_config_sha256": sha256(export_config),
    "replay_config": str(replay_config),
    "replay_config_sha256": sha256(replay_config),
    "validation_go_json": str(pathlib.Path(go_text).resolve()),
    "validation_go_json_sha256": go_sha,
    "training_suite_evidence_json": str(pathlib.Path(evidence_text).resolve()),
    "training_suite_evidence_json_sha256": evidence_sha,
    "input_jsonl": str(input_path),
    "input_jsonl_sha256": sha256(input_path),
    "ceiling_jsonl": str(ceiling),
    "ceiling_jsonl_sha256": sha256(ceiling),
    "ceiling_summary_json": str(ceiling_summary),
    "ceiling_summary_json_sha256": sha256(ceiling_summary),
    "ceiling_validation_json": str(ceiling_validation),
    "ceiling_validation_json_sha256": sha256(ceiling_validation),
    "export_source": source,
    "split": "test",
    "runtime_gt_input": False,
    "selected_axis_gt_remap": False,
    "model_training": False,
}
write_json_exclusive(output_text, payload)
PY

echo "[DUCA_ALLOCATION_VALIDATION_EXPORT] PASS ${OUTPUT_ROOT}/manifest.json"
