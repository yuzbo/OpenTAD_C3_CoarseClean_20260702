#!/usr/bin/env bash
#SBATCH --job-name=h65-pro-admission
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

fail() { echo "[H65_PRO_ADMISSION][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
EXPECTED_COMMIT="${H65_PRO_EXPECTED_COMMIT:?H65_PRO_EXPECTED_COMMIT is required}"
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not the expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "source checkout is not clean"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export DUCA_MOBILENETV3_WEIGHTS_PATH="${DUCA_MOBILENETV3_WEIGHTS_PATH:-$YUZIBO_ROOT/pretrained/mobilenet_v3_small-047dcff4.pth}"
[[ -r "$DUCA_MOBILENETV3_WEIGHTS_PATH" ]] || fail "MobileNetV3 weights are not readable"

python tools/bata/validate_h65_pro_fullmatrix.py
python -m pytest -q tests/test_h65_pro_fullmatrix.py
python - <<'PY'
import torch

from tools.bata.train_lowres_action_probe import C3MobileNetV3ActionProbe
import os

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
probe = C3MobileNetV3ActionProbe(
    pretrained=True,
    freeze_backbone=True,
    weights_path=os.environ["DUCA_MOBILENETV3_WEIGHTS_PATH"],
    preserve_pretrained_classifier=True,
)
probe.module.cuda().eval()
frames = torch.rand(1, 2, 3, 64, 64, device="cuda")
valid = torch.ones(1, 2, dtype=torch.bool, device="cuda")
with torch.no_grad():
    output = probe(frames, valid, return_hidden=True)
if not bool(torch.isfinite(output["logits"]).all().item()):
    raise SystemExit("MobileNetV3 reference produced non-finite logits")
print("H65_PRO_CUDA_REFERENCE_GATE_OK", tuple(output["logits"].shape))
PY
