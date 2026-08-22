import pytest
import torch
from opentad.models.utils.temporal_grid import global_rank_clip_coordinates, clip_relative_physical_time_mask
from opentad.models.detectors.two_stage import TwoStageDetector
from opentad.models.duca.structured_selection import exact_uniform_positions

def meta(k=384, valid=768):
    return [{"irregular_selected_positions": exact_uniform_positions(valid, k), "irregular_dense_valid_len": valid}]

def test_singleclock_admission_requires_metadata_and_validates_shape():
    x = torch.zeros(1, 384, 1)
    with pytest.raises(ValueError):
        TwoStageDetector._validate_single_clock_metadata([{}], x, batch_size=1)
    with pytest.raises(ValueError):
        TwoStageDetector._validate_single_clock_metadata(meta(3), x, batch_size=1)
    bad = meta(); bad[0]["irregular_selected_positions"][1] = bad[0]["irregular_selected_positions"][0]
    with pytest.raises(ValueError):
        TwoStageDetector._validate_single_clock_metadata(bad, x, batch_size=1)

def test_singleclock_global_helper_batch2_and_clip_geometry():
    positions = exact_uniform_positions(768, 384)
    assert positions.numel() == 384 and bool((positions[1:] > positions[:-1]).all())
    assert positions[0].item() == 0 and positions[-1].item() == 767
    batched = positions.repeat(2, 1)
    assert batched.shape == (2, 384)

def test_off_contract_keeps_legacy_default():
    detector = TwoStageDetector()
    assert detector.single_clock_admission is False

def test_global_helper_unique_lengths_and_short_window_fail():
    p = torch.arange(384, dtype=torch.long).repeat(2, 1)
    out = global_rank_clip_coordinates(p, torch.tensor([768, 800]))
    assert out["actual"].shape == (2, 24, 8)
    assert p.dtype == torch.long
    assert out["actual"].dtype == torch.float32
    assert out["canonical"].dtype == torch.float32
    assert out["actual"][0, 0, 0].item() == 0.5
    assert out["actual"][1, 0, 0].item() == 0.5
    assert torch.equal(out["irregular_selected_positions"], p)
    uniform_positions = torch.stack([exact_uniform_positions(768, 384), exact_uniform_positions(800, 384)])
    uniform = global_rank_clip_coordinates(uniform_positions, torch.tensor([768, 800]))
    assert torch.equal(uniform["actual"], uniform["canonical"])
    assert clip_relative_physical_time_mask(
        uniform["actual"][:, 0, :], uniform["canonical"][:, 0, :], spatial_tokens=24
    ) is None
    with pytest.raises(ValueError):
        global_rank_clip_coordinates(exact_uniform_positions(383, 384).view(1, -1), torch.tensor([383]))

def test_physical_mask_uniform_none_and_nonuniform_block_shape():
    canonical = torch.arange(8.).repeat(2, 1)
    assert clip_relative_physical_time_mask(canonical, canonical, spatial_tokens=24) is None
    actual = canonical.clone(); actual[0, 1] += 0.5
    mask = clip_relative_physical_time_mask(actual, canonical, spatial_tokens=24)
    assert mask.shape == (2, 1, 192, 192)
    assert torch.equal(mask[1], torch.zeros_like(mask[1]))
    assert mask[0, 0, 0, 24] == -0.5

def test_cycle4_config_checkpoint_contract_is_explicit():
    from pathlib import Path
    text = Path("configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py").read_text()
    assert "checkpoint_interval_epochs = 5" in text
    assert "keep_latest=3" in text and "final_ema=True" in text
    assert "tubelet_packed_runtime_route=dict(enabled=False)" in text

def test_cycle4_launcher_canonical_modes_and_stage1_no_checkpoint_gate():
    from pathlib import Path
    text = Path("scripts/run_duca_h65_matched_cycle4_n16r4.sbatch").read_text()
    assert "/data/run01/sczc063/yuzibo/thumos14/raw_data/video" in text
    assert "thumos_14_anno.json" in text and "category_idx.txt" in text
    assert "vit-small-p16-videomae" in text or "vit-small-p16_videomae" in text
    assert "PRECHECK_TARGET" in text and "STAGE1|STAGE2_OFF|STAGE2_ON" in text
    assert 'if [[ "$PRECHECK_TARGET" != STAGE1 ]]' in text
    assert "model.backbone.custom.pretrain" in text

def test_cycle4_launcher_bounded_pre_run_is_explicit_and_does_not_change_formal_modes():
    from pathlib import Path
    text = Path("scripts/run_duca_h65_matched_cycle4_n16r4.sbatch").read_text()
    assert 'if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then' in text
    assert "workflow.end_epoch=1" in text and "workflow.max_train_iters=1" in text
    assert "workflow.val_eval_interval=-1" in text and "workflow.val_loss_interval=-1" in text
    assert "workflow.val_start_epoch=9999" in text and "workflow.checkpoint_interval=1" in text
    assert "total_epochs=1" not in text and "max_updates=1" not in text
    assert 'if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then' in text
    formal = text[text.index('if [[ "$MODE" == STAGE1 ]]; then'):]
    assert "workflow.end_epoch=1" not in formal
