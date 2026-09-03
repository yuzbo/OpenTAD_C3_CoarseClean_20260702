#!/usr/bin/env bash
#SBATCH --job-name=duca-evidence-admission
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_EVIDENCE_ADMISSION][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
EXPECTED_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:?DUCA_EVIDENCE_EXPECTED_COMMIT is required}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not the expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "source checkout is not clean"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/raw/Test Data/TH14_test_set_mp4}"
export DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT:-$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800}"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-$LEDGER_ROOT/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-$LEDGER_ROOT/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-$LEDGER_ROOT/test/value_transport_ledger_delta_p_action_384.jsonl}"

for required in \
  "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" \
  "$THUMOS14_TRAIN_DATA_PATH" "$THUMOS14_TEST_DATA_PATH" \
  "$DUCA_VIDEOMAE_PRETRAIN" "$DUCA_H65_TRAIN_LEDGER_PATH" \
  "$DUCA_H65_VAL_LEDGER_PATH" "$DUCA_H65_TEST_LEDGER_PATH"; do
  [[ -r "$required" ]] || fail "required admission input is not readable: $required"
done

python - "$DUCA_H65_TRAIN_LEDGER_PATH" "$DUCA_H65_VAL_LEDGER_PATH" "$DUCA_H65_TEST_LEDGER_PATH" <<'PY'
import json
import sys

expected_policy = "c3_lowres_probe_delta_p_action"
for path in sys.argv[1:]:
    rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            rows += 1
            if row.get("policy") != expected_policy:
                raise SystemExit(f"{path}:{line_number}: unexpected policy={row.get('policy')!r}")
            if row.get("deploy_selection_ledger") is not True:
                raise SystemExit(f"{path}:{line_number}: ledger row is not deployable")
            if int(row.get("selected_count", -1)) != len(row.get("selected_positions", ())):
                raise SystemExit(f"{path}:{line_number}: selected-count mismatch")
    if rows == 0:
        raise SystemExit(f"{path}: empty ledger")
    print("DUCA_EVIDENCE_LEDGER_GATE_OK", path, rows)
PY

python -m py_compile tools/train.py tools/test.py tools/bata/duca_evidence_training.py
python -m pytest -q \
  tests/test_duca_evidence_training_contract.py \
  tests/test_duca_evidence_recovery_matrix.py \
  tests/test_duca_evidence_recovery.py
python - <<'PY'
import torch

from opentad.models.duca.evidence_recovery import DucaEvidenceRecoveryModule

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
module = DucaEvidenceRecoveryModule(budget=4, window_size=8).cuda().eval()
inputs = torch.randn(1, 3, 8, 16, 16, device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    scout = module.acquire(inputs)["scout"]
if scout["utility"].dtype != torch.float32:
    raise SystemExit(f"scout utility drifted from FP32: {scout['utility'].dtype}")
if not bool(torch.isfinite(scout["utility"]).all().item()):
    raise SystemExit("scout utility is non-finite")
print("DUCA_EVIDENCE_CUDA_FP32_GATE_OK", scout["utility"].dtype)
PY
