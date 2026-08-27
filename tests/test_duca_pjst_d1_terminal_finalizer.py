from pathlib import Path

ROOT = Path(__file__).parents[1]
EVAL = (ROOT / "scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch").read_text()
SHARD = (ROOT / "scripts/run_duca_pjst_d1_terminal_bootstrap_shard_n16r4.sbatch").read_text()
MERGE = (ROOT / "scripts/run_duca_pjst_d1_terminal_bootstrap_merge_n16r4.sbatch").read_text()
BUNDLE = (ROOT / "scripts/run_duca_pjst_d1_terminal_bootstrap_bundle_n16r4.sbatch").read_text()


def test_eval_freezes_arm_checkpoint_and_serialization_contract():
    assert 'case "$ARM" in OFF)' in EVAL and 'ON)' in EVAL
    assert "epoch_59.pth" in EVAL
    assert "--checkpoint-state-key state_dict_ema" in EVAL
    assert "--expected-checkpoint-epoch 59" in EVAL
    assert "post_processing.save_dict=True" in EVAL
    assert "video_count') != 211" in EVAL
    assert "output root already exists" in EVAL
    assert 'ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"' in EVAL
    assert '"$ADATAD_PRETRAIN"; do [[ -r "$p" ]]' in EVAL
    assert 'model.backbone.custom.pretrain="$ADATAD_PRETRAIN"' in EVAL


def test_bootstrap_is_exactly_sharded_and_merged_after_all_inputs():
    assert "idx*10000/16" in SHARD and "(idx+1)*10000/16" in SHARD
    assert "--sample-start \"$start\" --sample-stop \"$stop\"" in SHARD
    assert "--nonce \"DUCA-PJST-D1-OFF-ON-v001\"" in SHARD
    assert "--namespace PAIRED_VIDEO_BOOTSTRAP_V1" in SHARD
    assert "for i in $(seq 0 15)" in MERGE
    assert "missing shard $i" in MERGE
    assert "merge_duca_h65_bootstrap_shards.py" in MERGE


def test_no_training_or_raw_prediction_shortcut():
    assert "tools/train.py" not in EVAL
    assert "inference.load_from_raw_predictions=False" in EVAL


def test_bundle_has_bounded_parallelism_wait_propagation_and_ordered_merge():
    assert "for i in $(seq 0 15)" in BUNDLE
    assert "DUCA_BOOTSTRAP_WORKERS" in BUNDLE and "workers <= 16" in BUNDLE
    assert "run_shard \"$i\" &" in BUNDLE
    assert "wait \"${running[0]}\" || fail" in BUNDLE
    assert "for pid in \"${running[@]}\"; do wait \"$pid\" || fail" in BUNDLE
    assert "missing shard $i" in BUNDLE
    assert BUNDLE.index("for pid in") < BUNDLE.index("merge_duca_h65_bootstrap_shards.py")
