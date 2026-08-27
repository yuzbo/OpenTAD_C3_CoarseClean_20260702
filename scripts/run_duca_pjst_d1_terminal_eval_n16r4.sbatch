#!/usr/bin/env bash
# Evaluation-only PJST-D1 terminal receipt (one frozen arm per invocation).
set -euo pipefail
fail(){ echo "[PJST_D1_EVAL][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"; ROOT="$(cd "$ROOT" && pwd -P)"; cd "$ROOT"
[[ -f tools/test.py ]] || fail "invalid repository root"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_COMMIT="${DUCA_EVAL_COMMIT:?DUCA_EVAL_COMMIT is required}"; [[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "wrong repository HEAD"; [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "repository must be clean"
ARM="${DUCA_PJST_ARM:?DUCA_PJST_ARM must be OFF or ON}"; case "$ARM" in OFF) CFG=configs/adatad/thumos/duca_pjst_d1_stage2_off.py; CKPT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/gpu1_id0/checkpoint/epoch_59.pth;; ON) CFG=configs/adatad/thumos/duca_pjst_d1_stage2_on.py; CKPT=/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/gpu1_id0/checkpoint/epoch_59.pth;; *) fail "arm must be OFF or ON";; esac
[[ "$CKPT" == */epoch_59.pth ]] || fail "checkpoint epoch is not frozen"; [[ -r "$CKPT" ]] || fail "checkpoint missing: $CKPT"
VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"; ANN="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"; CLASS="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"; ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"; for p in "$VIDEO_ROOT" "$ANN" "$CLASS" "$ADATAD_PRETRAIN"; do [[ -r "$p" ]] || fail "canonical input missing: $p"; done
OUT="${DUCA_EVAL_OUTPUT_ROOT:?DUCA_EVAL_OUTPUT_ROOT is required}"; [[ ! -e "$OUT" ]] || fail "output root already exists"; mkdir -p "$OUT"
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate; export DUCA_EXPECTED_COMMIT="$EXPECTED_COMMIT"
python -m torch.distributed.run --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py "$CFG" --checkpoint "$CKPT" --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 --metrics-json "$OUT/terminal_evaluation.json" --id 0 --seed 3407 --cfg-options work_dir="$OUT/work" model.backbone.custom.pretrain="$ADATAD_PRETRAIN" dataset.test.ann_file="$ANN" dataset.test.class_map="$CLASS" dataset.test.data_path="$VIDEO_ROOT" evaluation.ground_truth_filename="$ANN" post_processing.save_dict=True inference.load_from_raw_predictions=False
python - "$OUT/terminal_evaluation.json" "$OUT/terminal_eval_sidecar.json" "$ARM" "$EXPECTED_COMMIT" "$CFG" "$CKPT" <<'PY'
import json,sys
from pathlib import Path
src,dst,arm,commit,cfg,ckpt=sys.argv[1:]; p=json.loads(Path(src).read_text());
if p.get('checkpoint_epoch') != 59 or p.get('checkpoint_state_key') != 'state_dict_ema' or p.get('video_count') != 211 or not p.get('prediction_path'): raise SystemExit('terminal evaluation contract violated')
p.update({'arm':arm,'git_commit':commit,'config_path':str(Path(cfg).resolve()),'checkpoint_path':ckpt,'state_key':'state_dict_ema','epoch':59,'prediction_path':str(Path(p['prediction_path']).resolve())})
Path(dst).write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
