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
    p = torch.stack([exact_uniform_positions(768, 384), exact_uniform_positions(800, 384)])
    out = global_rank_clip_coordinates(p, torch.tensor([768, 800]))
    assert out["actual"].shape == (2, 24, 8)
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

def test_cycle3_config_checkpoint_contract_is_explicit():
    from pathlib import Path
    text = Path("configs/adatad/thumos/duca_h65_first_singleclock_cycle3.py").read_text()
    assert "checkpoint_interval_epochs = 5" in text
    assert "keep_latest=3" in text and "final_ema=True" in text
    assert "tubelet_packed_runtime_route=dict(enabled=False)" in text
