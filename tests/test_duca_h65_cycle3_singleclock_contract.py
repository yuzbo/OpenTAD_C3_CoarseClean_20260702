import pytest
import torch
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
