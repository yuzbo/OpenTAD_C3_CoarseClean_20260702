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
