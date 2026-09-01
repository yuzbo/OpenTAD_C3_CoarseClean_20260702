#!/usr/bin/env bash
#SBATCH --job-name=r5-aggregate
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-aggregate-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-aggregate-%j.err

set -euo pipefail
: "${DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}"
: "${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}"
cd "${DUCA_REPO_ROOT}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${DUCA_EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
IFS= read -r matrix_sha < 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json.sha256'
[[ "$(sha256sum 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json' | awk '{print $1}')" == "${matrix_sha}" ]]
"${PYTHON}" -m tools.bata.aggregate_duca_r5_paper_matrix   --matrix-summary 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json'   --expected-commit "${DUCA_EXPECTED_COMMIT}"   --output-json 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/final_results.json'
sha256sum 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/final_results.json' | awk '{print $1}' >   'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/final_results.json.sha256'
