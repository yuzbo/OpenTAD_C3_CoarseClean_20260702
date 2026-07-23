#!/usr/bin/env bash
# Train a small real-update DUCA run, then render fixed-window diagnostics.
# This is intentionally a compact diagnostic runner, not an official launcher.
set -eo pipefail

fail() { echo "[DUCA_MINI_VISUAL][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
RUN_ROOT="${DUCA_MINI_VISUAL_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "one Slurm GPU allocation is required"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "a fresh DUCA_MINI_VISUAL_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "an exact commit is required"

# Site profiles may inspect unset desktop variables, so source before -u.
set +u
source /etc/profile
set -u
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"

cd "${REPO_ROOT}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
export BASE
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "VideoMAE pretrain is missing"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == 1 ]] \
  || fail "exactly one Slurm-visible GPU is required"

CONFIG="configs/adatad/thumos/duca_sampling_rate_both_asformer_full_mini_visual.py"
WORK_DIR="${RUN_ROOT}/work"
mkdir -p "${RUN_ROOT}/attribution" "${RUN_ROOT}/selection" "${RUN_ROOT}/figures"

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-mini-visual-${SLURM_JOB_ID}" \
  tools/train.py "${CONFIG}" --id 0 --seed 3407 --cfg-options \
  "work_dir=${WORK_DIR}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
for epoch in 0 4 9; do
  checkpoint="${ACTUAL_WORK_DIR}/checkpoint/epoch_${epoch}.pth"
  [[ -f "${checkpoint}" ]] || fail "missing checkpoint ${checkpoint}"
  label=$((epoch + 1))
  "${PYTHON}" -m tools.bata.export_duca_training_attribution \
    --config "${CONFIG}" --checkpoint "${checkpoint}" \
    --checkpoint-state state_dict_ema --device cuda:0 \
    --batch-index 0 --batch-size 1 --seed 3407 \
    --cfg-options "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    --output-jsonl "${RUN_ROOT}/attribution/epoch_${label}.jsonl" \
    --summary-json "${RUN_ROOT}/attribution/epoch_${label}.summary.json" \
    | tee "${RUN_ROOT}/attribution/epoch_${label}.out"
  "${PYTHON}" -m tools.bata.export_duca_selection_quality \
    --config "${CONFIG}" --selector-config "${CONFIG}" --checkpoint "${checkpoint}" \
    --split test --batch-size 1 --limit-batches 2 --use-ema true --device cuda:0 \
    --cfg-options "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    --output-jsonl "${RUN_ROOT}/selection/epoch_${label}.jsonl" \
    --summary-json "${RUN_ROOT}/selection/epoch_${label}.summary.json" \
    | tee "${RUN_ROOT}/selection/epoch_${label}.out"
  "${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
    --records-jsonl "${RUN_ROOT}/selection/epoch_${label}.jsonl" \
    --output-dir "${RUN_ROOT}/figures/epoch_${label}" --bootstrap-repeats 32 \
    | tee "${RUN_ROOT}/figures/epoch_${label}.out"
done

"${PYTHON}" -m tools.bata.plot_duca_training_attribution \
  --records-jsonl "${RUN_ROOT}/attribution/epoch_1.jsonl" \
  "${RUN_ROOT}/attribution/epoch_5.jsonl" \
  "${RUN_ROOT}/attribution/epoch_10.jsonl" \
  --output-prefix "${RUN_ROOT}/figures/training_attribution" \
  | tee "${RUN_ROOT}/figures/training_attribution.out"

"${PYTHON}" - "${RUN_ROOT}" "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
paths = [
    root / "attribution" / f"epoch_{epoch}.summary.json"
    for epoch in (1, 5, 10)
]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": "duca_mini_visual_training_v1",
    "git_commit": sys.argv[2],
    "purpose": "trained_small_sample_mechanism_diagnostic_not_official_map",
    "train_epochs": 10,
    "optimizer_updates": 40,
    "tracked_training_batch": {"split": "train", "batch_index": 0, "batch_size": 1},
    "tracked_validation": {"split": "test", "batch_size": 1, "limit_batches": 2},
    "checkpoint_epochs": [1, 5, 10],
    "gt_role": "train_loss_or_posthoc_overlay_never_inference_decision",
    "attribution_summary_sha256": {path.name: sha256(path) for path in paths},
    "official_map_reported": False,
}
(root / "mini_visual_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
