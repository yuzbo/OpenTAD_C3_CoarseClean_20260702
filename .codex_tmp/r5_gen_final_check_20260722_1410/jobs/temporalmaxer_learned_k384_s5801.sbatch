#!/usr/bin/env bash
#SBATCH --job-name=r5-te-l-k384-s5801
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs/r5-te-l-k384-s5801-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs/r5-te-l-k384-s5801-%j.err

set -euo pipefail
: "${DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}"
: "${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}"
: "${ADATAD_PRETRAIN_PATH:?set ADATAD_PRETRAIN_PATH}"
: "${ADATAD_PRETRAIN_SHA256:?set ADATAD_PRETRAIN_SHA256}"
cd "${DUCA_REPO_ROOT}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${DUCA_EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')" == "${ADATAD_PRETRAIN_SHA256}" ]]
mkdir -p 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs'
: "${R5_FRONTEND_DECISION:?set R5_FRONTEND_DECISION}"
: "${R5_ALIGNMENT_JSON:?set R5_ALIGNMENT_JSON}"
if [[ -z "${R5_FRONTEND_DECISION_SHA256:-}" ]]; then
  : "${R5_FRONTEND_DECISION_SHA256_FILE:?set R5_FRONTEND_DECISION_SHA256_FILE}"
  IFS= read -r R5_FRONTEND_DECISION_SHA256 < "${R5_FRONTEND_DECISION_SHA256_FILE}"
fi
if [[ -z "${R5_ALIGNMENT_SHA256:-}" ]]; then
  : "${R5_ALIGNMENT_SHA256_FILE:?set R5_ALIGNMENT_SHA256_FILE}"
  IFS= read -r R5_ALIGNMENT_SHA256 < "${R5_ALIGNMENT_SHA256_FILE}"
fi
[[ "${R5_FRONTEND_DECISION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${R5_ALIGNMENT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
export R5_FRONTEND_DECISION_SHA256 R5_ALIGNMENT_SHA256
readarray -t r5_frontend < <("${PYTHON}" -   "${R5_FRONTEND_DECISION}" "${R5_FRONTEND_DECISION_SHA256}"   "${R5_ALIGNMENT_JSON}" "${R5_ALIGNMENT_SHA256}"   "${DUCA_EXPECTED_COMMIT}" boundary_burst_r2q3_g1   E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py <<'PY'
import hashlib
import sys
from pathlib import Path

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    validate_alignment_artifact,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
)

decision_path, decision_sha, alignment_path, alignment_sha, commit, variant, source = sys.argv[1:]
decision = validate_frontend_decision(
    decision_path=decision_path,
    decision_sha256=decision_sha,
    expected_commit=commit,
)
validate_alignment_artifact(
    path=alignment_path,
    digest=alignment_sha,
    expected_commit=commit,
    expected_variant=variant,
    source_config_path=source,
    source_config_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
)
selected = decision["family_routing"]["selected_p0_variant"]
winner = decision["winners"][selected]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
)
[[ "${#r5_frontend[@]}" == 3 ]]
export DUCA_FRONTEND_CHECKPOINT="${r5_frontend[0]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${r5_frontend[1]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${r5_frontend[2]}"
export DUCA_SELECTED_OPT_VARIANT=boundary_burst_r2q3_g1
export DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${R5_ALIGNMENT_JSON}"
export DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${R5_ALIGNMENT_SHA256}"
[[ -f "${DUCA_FRONTEND_CHECKPOINT}" ]]
[[ "$(sha256sum "${DUCA_FRONTEND_CHECKPOINT}" | awk '{print $1}')" ==    "${DUCA_FRONTEND_CHECKPOINT_SHA256}" ]]
mkdir -p E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/results
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="r5-${SLURM_JOB_ID}-r5-te-l-k384-s5801" \
  tools/train.py E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/configs/temporalmaxer_learned_k384_s5801.py \
  --id 0 --seed 5801 \
  --cfg-options "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}"
[[ -f E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/temporalmaxer_learned_k384_s5801/gpu1_id0/checkpoint/epoch_59.pth ]]
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="r5-${SLURM_JOB_ID}-r5-te-l-k384-s5801-eval" \
  tools/test.py E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/configs/temporalmaxer_learned_k384_s5801.py \
  --checkpoint E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/temporalmaxer_learned_k384_s5801/gpu1_id0/checkpoint/epoch_59.pth \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 \
  --metrics-json E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/results/temporalmaxer_learned_k384_s5801.terminal_evaluation.json \
  --id 0 --seed 5801 \
  --cfg-options \
    "work_dir=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/results/temporalmaxer_learned_k384_s5801_eval" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False"
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/results/temporalmaxer_learned_k384_s5801.terminal_evaluation.json ]]
