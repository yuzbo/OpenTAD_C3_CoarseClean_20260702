#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_R0_HOLDOUT_MAP][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=exposure132
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_R0_CHECKPOINT:-}"
EXPECTED_EPOCH="${DUCA_R0_CHECKPOINT_EPOCH:-131}"
OUTPUT_ROOT="${DUCA_R0_OUTPUT_ROOT:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${CHECKPOINT}" && -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "checkpoint/pretrain is missing"
[[ -f "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST:-}" ]] || fail "holdout block list is missing"
[[ -n "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}" ]] || fail "fresh output root is required"
mkdir -p "${OUTPUT_ROOT}"

EVAL_BLOCKED="${OUTPUT_ROOT}/evaluation_blocked_videos.json"
"${PYTHON}" - "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}" "${EVAL_BLOCKED}" <<'PY'
import json, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
videos = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if not videos or len(videos) != len(set(videos)):
    raise SystemExit("R0 evaluator blocked-video list must be nonempty and unique")
target.write_text(json.dumps(videos, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
export DUCA_R0_EVAL_BLOCKED_VIDEOS="${EVAL_BLOCKED}"

INPUT="${OUTPUT_ROOT}/holdout_inputs.jsonl"
INPUT_SUMMARY="${OUTPUT_ROOT}/holdout_inputs.summary.json"
FAMILIES="${OUTPUT_ROOT}/holdout_families.jsonl"
FAMILY_SUMMARY="${OUTPUT_ROOT}/holdout_families.summary.json"
"${PYTHON}" -m tools.bata.export_duca_allocation_ceiling_inputs \
  --config configs/adatad/thumos/duca_boundary_burst_r0_holdout_export.py \
  --checkpoint "${CHECKPOINT}" --output-jsonl "${INPUT}" \
  --summary-json "${INPUT_SUMMARY}" --split train --requested-budget 384 \
  --device cuda:0 --use-ema true --batch-size 1 --num-workers 2 \
  --coordinate-tolerance-frames 0

"${PYTHON}" -m tools.bata.diagnose_duca_allocation_family_ceiling \
  --input-jsonl "${INPUT}" --output-jsonl "${FAMILIES}" \
  --summary-json "${FAMILY_SUMMARY}" --score-key transition_policy_scores \
  --cap-policy explicit_frames --cap-value 12 --gt-families both \
  --boundary-radii 0 1 2 4 --quantization-scale 1000000 \
  --gt-time-limit-seconds 120
"${PYTHON}" -m tools.bata.validate_duca_allocation_ceiling_artifact \
  --input-jsonl "${INPUT}" --output-jsonl "${FAMILIES}" \
  --summary-json "${FAMILY_SUMMARY}" \
  --validation-json "${OUTPUT_ROOT}/holdout_families.validation.json"

export DUCA_ALLOCATION_ARTIFACT_PATH="${FAMILIES}"
export DUCA_ALLOCATION_ARTIFACT_SHA256="$(sha256sum "${FAMILIES}" | awk '{print $1}')"
families=(A_exact_uniform D_privileged_gt_ceiling E_privileged_unrestricted_gt)
for family in "${families[@]}"; do
  export DUCA_ALLOCATION_FAMILY_KEY="${family}"
  if [[ "${family}" == A_exact_uniform ]]; then
    export DUCA_ALLOCATION_ALLOW_PRIVILEGED=0
  else
    export DUCA_ALLOCATION_ALLOW_PRIVILEGED=1
  fi
  root="${OUTPUT_ROOT}/map/${family}"
  mkdir -p "${root}"
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-r0-${SLURM_JOB_ID}-${family}" \
    tools/test.py configs/adatad/thumos/duca_boundary_burst_r0_selected_axis_replay.py \
    --checkpoint "${CHECKPOINT}" --checkpoint-state-key state_dict_ema \
    --expected-checkpoint-epoch "${EXPECTED_EPOCH}" \
    --metrics-json "${root}/metrics.json" --id 0 --seed 3407 \
    --cfg-options "work_dir=${root}/work" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
      "post_processing.save_dict=True" "inference.load_from_raw_predictions=False" \
    2>&1 | tee "${root}/test.out"
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${EXPECTED_COMMIT}" "${CHECKPOINT}" \
  "${INPUT}" "${FAMILIES}" "${FAMILY_SUMMARY}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows = []
for family in ("A_exact_uniform", "D_privileged_gt_ceiling", "E_privileged_unrestricted_gt"):
    path = root / "map" / family / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"family": family, "metrics_path": str(path), "metrics_sha256": digest(path), "metrics": payload.get("metrics", payload)})
summary = {
    "schema": "duca_r0_selected_axis_holdout_map_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "checkpoint": str(Path(sys.argv[3]).resolve()),
    "checkpoint_sha256": digest(sys.argv[3]),
    "input_jsonl": str(Path(sys.argv[4]).resolve()),
    "input_jsonl_sha256": digest(sys.argv[4]),
    "families_jsonl": str(Path(sys.argv[5]).resolve()),
    "families_jsonl_sha256": digest(sys.argv[5]),
    "family_summary": str(Path(sys.argv[6]).resolve()),
    "family_summary_sha256": digest(sys.argv[6]),
    "evaluation_blocked_videos": str(root / "evaluation_blocked_videos.json"),
    "evaluation_blocked_videos_sha256": digest(root / "evaluation_blocked_videos.json"),
    "source_subset": "training_internal_holdout",
    "test_subset_consumed": False,
    "runtime_gt_input_to_selector": False,
    "selected_axis_detector": True,
    "rows": rows,
    "paper_claim_allowed": False,
}
(root / "r0_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[DUCA_R0_HOLDOUT_MAP] completed ${OUTPUT_ROOT}/r0_summary.json"
