#!/usr/bin/env bash
#SBATCH --job-name=bafdr-receipt-seal
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

source /etc/profile
set -euo pipefail
cd "${SEAL_REPO:?}"
[[ "$(git rev-parse HEAD)" == "${SEAL_EXPECTED_SHA:?}" ]]
[[ -z "$(git status --porcelain)" ]]
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
python -m pytest tests/test_bafdr_receipt_seal.py tests/test_bafdr_k16_static_protocol.py tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
python -c 'import torch; assert torch.cuda.is_available(); assert torch.isfinite(torch.ones(2, device="cuda").sum()); print("CUDA environment PASS; receipt-only change")'

SOURCE_ROOT=/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs
OUTPUT_ROOT="${SEAL_OUTPUT_ROOT:?}"
for MODE in precheck seal; do
  for ARM in g96 u16_uniform_a0 late nokd full; do
    EXTRA=()
    [[ "$MODE" != precheck ]] || EXTRA+=(--precheck-only)
    python tools/bata/seal_bafdr_receipt.py \
      "$SOURCE_ROOT/bafdr_k16_${ARM}_seed4407/eval_receipt.json" \
      "$OUTPUT_ROOT/${ARM}_eval_receipt_sealed.json" \
      --expected-evaluator-sha 29b5a7a2b291203ea7b697cfe416b64f0d365d02 \
      --expected-training-sha 539287fa8a035765afd7e79863ce77278bef83f2 \
      "${EXTRA[@]}"
  done
done
