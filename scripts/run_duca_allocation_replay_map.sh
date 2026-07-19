#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_REPLAY_MAP][FAIL] $*" >&2
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
INPUT_JSONL="${DUCA_ALLOCATION_INPUT_JSONL:-}"
CEILING_JSONL="${DUCA_ALLOCATION_ARTIFACT_PATH:-}"
CEILING_SUMMARY="${DUCA_ALLOCATION_ARTIFACT_SUMMARY:-}"
CEILING_SHA256="${DUCA_ALLOCATION_ARTIFACT_SHA256:-}"
FAMILY_KEY="${DUCA_ALLOCATION_FAMILY_KEY:-}"
ALLOW_PRIVILEGED="${DUCA_ALLOCATION_ALLOW_PRIVILEGED:-0}"
ALLOW_PRIVILEGED_JSON=false
if [[ "${ALLOW_PRIVILEGED}" == "1" ]]; then
  ALLOW_PRIVILEGED_JSON=true
fi
OUTPUT_ROOT="${DUCA_ALLOCATION_REPLAY_OUTPUT_ROOT:-}"
EXPECTED_EPOCH="${DUCA_ALLOCATION_CHECKPOINT_EPOCH:-131}"
CONFIG="configs/adatad/thumos/duca_allocation_ceiling_physical_grid_replay.py"
VALIDATION_MANIFEST="${DUCA_ALLOCATION_VALIDATION_EXPORT_MANIFEST:-}"
VALIDATION_MANIFEST_SHA256="${DUCA_ALLOCATION_VALIDATION_EXPORT_MANIFEST_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "replay mAP must run inside a Slurm GPU allocation"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] \
  || fail "replay mAP requires cluster n16r4"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "replay mAP requires exactly one Slurm-visible GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "replay mAP requires a clean exact-commit checkout"
[[ -f "${CHECKPOINT}" && -f "${PRETRAIN}" ]] || fail "checkpoint or backbone pretrain is missing"
[[ -f "${INPUT_JSONL}" && -f "${CEILING_JSONL}" && -f "${CEILING_SUMMARY}" ]] \
  || fail "allocation input/ceiling artifacts are missing"
[[ -f "${VALIDATION_MANIFEST}" && "${VALIDATION_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "sealed validation replay requires a hashed validation-export manifest"
[[ "$(sha256sum "${VALIDATION_MANIFEST}" | awk '{print $1}')" == "${VALIDATION_MANIFEST_SHA256}" ]] \
  || fail "validation-export manifest SHA-256 mismatch"
[[ "$(sha256sum "${CEILING_JSONL}" | awk '{print $1}')" == "${CEILING_SHA256}" ]] \
  || fail "allocation ceiling SHA-256 mismatch"
[[ -n "${FAMILY_KEY}" ]] || fail "allocation family key is required"
[[ "${ALLOW_PRIVILEGED}" == "0" || "${ALLOW_PRIVILEGED}" == "1" ]] \
  || fail "DUCA_ALLOCATION_ALLOW_PRIVILEGED must be 0 or 1"
[[ -n "${OUTPUT_ROOT}" && "${OUTPUT_ROOT}" == "${BASE}/"* ]] \
  || fail "replay output must stay under BASE"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite replay output"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" - \
  "${VALIDATION_MANIFEST}" \
  "${EXPECTED_COMMIT}" \
  "${CHECKPOINT}" \
  "${INPUT_JSONL}" \
  "${CEILING_JSONL}" \
  "${CEILING_SUMMARY}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("schema_version") != "duca_allocation_validation_export_manifest_v1":
    raise SystemExit("validation-export manifest schema mismatch")
if manifest.get("git_commit") != sys.argv[2] or manifest.get("split") != "test":
    raise SystemExit("validation-export manifest commit/split mismatch")
if manifest.get("runtime_gt_input") is not False:
    raise SystemExit("validation-export manifest permits runtime GT")
bindings = (
    ("checkpoint", "checkpoint_sha256", sys.argv[3]),
    ("input_jsonl", "input_jsonl_sha256", sys.argv[4]),
    ("ceiling_jsonl", "ceiling_jsonl_sha256", sys.argv[5]),
    ("ceiling_summary_json", "ceiling_summary_json_sha256", sys.argv[6]),
)
for path_key, hash_key, expected_path in bindings:
    path = pathlib.Path(expected_path).resolve()
    if pathlib.Path(manifest.get(path_key, "")).resolve() != path:
        raise SystemExit(f"validation-export path mismatch: {path_key}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(hash_key):
        raise SystemExit(f"validation-export hash mismatch: {hash_key}")
validation_path = pathlib.Path(manifest["ceiling_validation_json"]).resolve()
if hashlib.sha256(validation_path.read_bytes()).hexdigest() != manifest.get(
    "ceiling_validation_json_sha256"
):
    raise SystemExit("ceiling validation receipt hash mismatch")
PY

"${PYTHON}" - "${CEILING_JSONL}" "${FAMILY_KEY}" "${ALLOW_PRIVILEGED}" <<'PY'
import json
import sys

path, family_key, allow_privileged = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
splits = set()
count = 0
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        splits.add(row["split"])
        matches = [
            family
            for family in row["families"]
            if family["family_key"] == family_key
        ]
        if len(matches) != 1:
            raise SystemExit(f"{row['sample_id']}: missing or duplicate family {family_key}")
        family = matches[0]
        if family["privileged"] and not allow_privileged:
            raise SystemExit("privileged replay requires explicit opt-in")
        count += 1
if splits.isdisjoint({"val", "test"}) or "train" in splits or len(splits) != 1:
    raise SystemExit(f"replay mAP requires one sealed validation/test split, got {sorted(splits)}")
if count < 1:
    raise SystemExit("replay artifact is empty")
PY

export DUCA_ALLOCATION_ARTIFACT_PATH="${CEILING_JSONL}"
export DUCA_ALLOCATION_ARTIFACT_SHA256="${CEILING_SHA256}"
export DUCA_ALLOCATION_FAMILY_KEY="${FAMILY_KEY}"
export DUCA_ALLOCATION_ALLOW_PRIVILEGED="${ALLOW_PRIVILEGED}"

cat > "${OUTPUT_ROOT}/manifest.json" <<EOF
{
  "schema_version": "duca_allocation_replay_map_manifest_v1",
  "task": "offline_temporal_action_detection",
  "git_commit": "${EXPECTED_COMMIT}",
  "checkpoint": "${CHECKPOINT}",
  "checkpoint_sha256": "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')",
  "input_jsonl": "${INPUT_JSONL}",
  "input_jsonl_sha256": "$(sha256sum "${INPUT_JSONL}" | awk '{print $1}')",
  "ceiling_jsonl": "${CEILING_JSONL}",
  "ceiling_jsonl_sha256": "${CEILING_SHA256}",
  "ceiling_summary": "${CEILING_SUMMARY}",
  "ceiling_summary_sha256": "$(sha256sum "${CEILING_SUMMARY}" | awk '{print $1}')",
  "validation_export_manifest": "${VALIDATION_MANIFEST}",
  "validation_export_manifest_sha256": "${VALIDATION_MANIFEST_SHA256}",
  "family_key": "${FAMILY_KEY}",
  "allow_privileged": ${ALLOW_PRIVILEGED_JSON},
  "model_training": false,
  "selected_axis_gt_remap": false,
  "paper_deployable": false
}
EOF

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-allocation-${SLURM_JOB_ID}-${FAMILY_KEY}" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch "${EXPECTED_EPOCH}" \
  --metrics-json "${OUTPUT_ROOT}/metrics.json" \
  --id 0 \
  --seed 0 \
  --cfg-options "work_dir=${OUTPUT_ROOT}/work" \
    "model.backbone.custom.pretrain=${PRETRAIN}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${OUTPUT_ROOT}/test.out"

[[ -f "${OUTPUT_ROOT}/metrics.json" ]] || fail "structured replay mAP evidence is missing"
echo "[DUCA_ALLOCATION_REPLAY_MAP] PASS ${OUTPUT_ROOT}/metrics.json"
