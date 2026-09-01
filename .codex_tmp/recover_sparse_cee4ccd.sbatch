#!/usr/bin/env bash
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --time=7-00:00:00

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

REPO=/data/run01/sczc063/yuzibo/projects/opentad_duca_sparse_cee4ccd_20260723
EXPECTED=cee4ccd33fb20e11978e4a2a6eaa3f5845b51489
OLD_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329/suite
: "${RUN_ROOT:?set RUN_ROOT}"
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED}"
test -z "$(git status --porcelain)"

declare -A p0_sha=(
  [1]=acb6e30673e811f34ce84d710442581bec8a74ca68e9187eb71e005e01536c9b
  [2]=e479df3cd29ba4cdde55ee6a08a125741333befc8377b161bd39396a3eb3fc82
  [3]=aeec4297bf1229b9e4084cf02c91bdd538226d9e8bf87691e1912d1969940b76
  [4]=c23e370286e3e45bac3f6a48847e3178226cab01cc99921936fbc5c02c8f2c3c
)
pids=()
for stride in 1 2 3 4; do
  p0="${OLD_ROOT}/arms/d${stride}/p0/work/gpu1_id0/checkpoint/epoch_19.pth"
  srun --exact --exclusive --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=4 \
    env BASE=/data/run01/sczc063/yuzibo \
      DUCA_REPO_ROOT="${REPO}" \
      DUCA_EXPECTED_COMMIT="${EXPECTED}" \
      DUCA_ADATAD_PRETRAIN_PATH=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth \
      DUCA_ADATAD_PRETRAIN_SHA256=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251 \
      DUCA_INDEPENDENT_VARIANT="sparse_probe_hidden_linear_d${stride}" \
      DUCA_SPARSE_PROBE_STRIDE="${stride}" \
      DUCA_INDEPENDENT_ARM_ROOT="${RUN_ROOT}/arms/d${stride}" \
      DUCA_REUSE_P0_CHECKPOINT="${p0}" \
      DUCA_REUSE_P0_CHECKPOINT_SHA256="${p0_sha[$stride]}" \
      bash scripts/run_duca_independent_official60_gpu1.sh \
      > "${RUN_ROOT}/logs/d${stride}.out" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
test "${failed}" = 0
python -m tools.bata.aggregate_duca_sparse_probe_tad \
  --run-root "${RUN_ROOT}/arms" \
  --output "${RUN_ROOT}/sparse_probe_tad_summary.json"
