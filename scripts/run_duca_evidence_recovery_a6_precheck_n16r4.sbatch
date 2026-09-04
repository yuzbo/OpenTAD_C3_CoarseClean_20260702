#!/usr/bin/env bash
#SBATCH --job-name=duca-ev-a6-precheck
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

fail() { echo "[DUCA_EVIDENCE_A6_PRECHECK][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
EXPECTED_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:?DUCA_EVIDENCE_EXPECTED_COMMIT is required}"
RUN_ROOT="${DUCA_A6_PRECHECK_ROOT:?DUCA_A6_PRECHECK_ROOT is required}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not the expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "source checkout is not clean"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export HOME=/data/run01/sczc063/yuzibo/tmp/home
export XDG_CACHE_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_cache
export XDG_CONFIG_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_config
export HF_HOME=/data/run01/sczc063/yuzibo/hf_cache
export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/raw/Test Data/TH14_test_set_mp4}"
export DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
LEGACY_LEDGER_ROOT="$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800"
FULLGRID_LEDGER_ROOT="$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_fullgrid_ec0bb6a_20260702_065702_+0800"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-$LEGACY_LEDGER_ROOT/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-$LEGACY_LEDGER_ROOT/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-$FULLGRID_LEDGER_ROOT/test/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_LEDGER_SOURCE=c3_lowres_probe_delta_p_action
export DUCA_H65_LEDGER_CONFIG_HASH=""

for required in "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" \
  "$THUMOS14_TRAIN_DATA_PATH" "$DUCA_VIDEOMAE_PRETRAIN" \
  "$DUCA_H65_TRAIN_LEDGER_PATH" "$DUCA_H65_VAL_LEDGER_PATH" "$DUCA_H65_TEST_LEDGER_PATH"; do
  [[ -r "$required" ]] || fail "required input is not readable: $required"
done

mkdir -p "$RUN_ROOT"
torchrun --nproc_per_node=1 --master_port="$((31000 + SLURM_JOB_ID % 10000))" \
  tools/train.py configs/adatad/thumos/duca_evidence_recovery_h65_selection_precheck.py \
  --seed 8261 --cfg-options work_dir="$RUN_ROOT"
