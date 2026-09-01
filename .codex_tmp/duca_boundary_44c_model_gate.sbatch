#!/usr/bin/env bash
#SBATCH --job-name=duca44c_model_gate
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_44c7227_model_gate_20260722_1305/gate-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_44c7227_model_gate_20260722_1305/gate-%j.err

source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
ROOT="$BASE/projects/c3_lowres_action_probe/duca_boundary_44c7227_model_gate_20260722_1305"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$ROOT"
  DEST="$ROOT/model_gate.sbatch"
  cp "$0" "$DEST"
  bash -n "$DEST"
  JOB_ID=$(sbatch --parsable --dependency=afterok:1179533 "$DEST")
  printf '%s\n' "$JOB_ID" | tee "$ROOT/job_id.txt"
  printf 'DUCA_44C_MODEL_GATE_SUBMITTED job=%s dependency=afterok:1179533 root=%s\n' "$JOB_ID" "$ROOT"
  exit 0
fi

REPO="$BASE/projects/opentad_duca_boundary_44c7227_20260722"
P0_ROOT="$BASE/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357"
COMMIT=44c7227b575b22c666b2f309c69b1dcfdc4102c8
PRETRAIN="$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAIN_SHA=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251

module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(sha256sum "$PRETRAIN" | awk '{print $1}')" = "$PRETRAIN_SHA"
test -f "$P0_ROOT/frontend_decision.json"
mkdir -p "$ROOT"

readarray -t ROUTE < <(python - "$P0_ROOT/frontend_decision.json" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if p.get('ok') is not True:
    raise SystemExit('P0 decision is not GO')
r = p['family_routing']
w = p['winners'][r['selected_p0_variant']]
print(r['selected_weakest_projected_family'])
print(w['checkpoint_path'])
print(w['checkpoint_sha256'])
print(int(w['epoch_one_based']) - 1)
PY
)
test "${#ROUTE[@]}" = 4
FAMILY="${ROUTE[0]}"
export DUCA_FRONTEND_CHECKPOINT="${ROUTE[1]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${ROUTE[2]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${ROUTE[3]}"
export DUCA_ADATAD_PRETRAIN_PATH="$PRETRAIN"
export DUCA_ADATAD_PRETRAIN_SHA256="$PRETRAIN_SHA"

if [[ "$FAMILY" == R4Q5* ]]; then
  G1=configs/adatad/thumos/duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py
  G2=configs/adatad/thumos/duca_boundary_burst_r4q5_g2_uni_companion_fixed384_official60.py
else
  G1=configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py
  G2=configs/adatad/thumos/duca_boundary_burst_g2_uni_companion_fixed384_official60.py
fi

for CONFIG in "$G1" "$G2"; do
  STEM=$(basename "$CONFIG" .py)
  python tools/bata/validate_duca_protected_e2e_official60.py \
    --config "$CONFIG" --output-json "$ROOT/${STEM}.contract.json"
  python -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca44c-${SLURM_JOB_ID}-${STEM}" \
    tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
    --config "$CONFIG" --expected-commit "$COMMIT" \
    --adatad-pretrain "$PRETRAIN" --adatad-pretrain-sha256 "$PRETRAIN_SHA" \
    --output-json "$ROOT/${STEM}.full_model.json"
done

python -m tools.bata.duca_r5_paper_matrix \
  --repo-root "$REPO" --output-dir "$ROOT/r5" \
  --uniform-config configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py \
  --learned-config "$G1" --cluster n16r4

python -m tools.bata.run_duca_temporalmaxer_one_step \
  --config "$ROOT/r5/configs/temporalmaxer_learned_k384_s3407.py" \
  --pretrain "$PRETRAIN" --pretrain-sha256 "$PRETRAIN_SHA" \
  --seed 3407 --device cuda:0 \
  --output "$ROOT/temporalmaxer_one_step.json"

sha256sum "$ROOT"/*.json "$ROOT/r5/cells.tsv" > "$ROOT/artifacts.sha256"
printf 'DUCA_44C_MODEL_GATE_PASS family=%s root=%s\n' "$FAMILY" "$ROOT"
