from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from opentad.models.builder import build_head, build_selector
from opentad.models.utils.post_processing.utils import convert_to_seconds, selected_axis_to_dense_axis
from opentad.models.utils.truetime_geometry import truetime_map_from_metadata


ROOT = Path(__file__).resolve().parents[1]


def select(arm, valid_lengths=(768, 519)):
    cfg = Config.fromfile(str(ROOT / f"configs/adatad/thumos/duca_ctdp_geometry_{arm}.py"))
    torch.manual_seed(42)
    frames = torch.randn(len(valid_lengths), 3, 768, 2, 2)
    masks = torch.arange(768)[None] < torch.tensor(valid_lengths)[:, None]
    metas = [dict(fps=25, snippet_stride=4, offset_frames=2,
                  window_start_frame=1000 * i, duration=1000)
             for i in range(len(valid_lengths))]
    segments = [torch.tensor([[min(500., float(length - 2)), min(600., float(length))], [0., float(length)]])
                for length in valid_lengths]
    labels = [torch.tensor([0, 1]) for _ in valid_lengths]
    output = build_selector(cfg.model.frame_selector).forward_train(
        frames, masks, metas, segments, labels)
    return cfg, output, segments, labels


@pytest.mark.parametrize("arm", ["g0", "g1"])
def test_ordinary_head_maps_gt_and_predictions_on_same_feature_axis(arm):
    cfg, output, original, labels = select(arm)
    assert not cfg.model.rpn_head.physical_grid_actionformer.enabled
    head = build_head(cfg.model.rpn_head)
    features = [torch.zeros(2, 512, 384 // int(s)) for s in head.prior_generator.strides]
    points = head.prior_generator(features)
    targets, _ = head.prepare_targets(points, [x[:1] for x in output["gt_segments"]],
                                     [x[:1] for x in labels])
    assert all(bool(t.sum() > 0) for t in targets)
    old_targets, _ = head.prepare_targets(points, [x[:1] for x in original],
                                         [x[:1] for x in labels])
    assert all(t.sum() == 0 for t in old_targets)
    for i, (meta, gt) in enumerate(zip(output["metas"], output["gt_segments"])):
        assert not meta["irregular_native_axis"]
        assert meta["detector_prediction_inverse_map_required"]
        time_map = truetime_map_from_metadata(meta)
        assert torch.equal(time_map.selected_positions,
                           output["temporal_positions"][i, :time_map.selected_len])
        restored = selected_axis_to_dense_axis(gt, meta, strict=True)
        torch.testing.assert_close(restored, original[i], atol=1e-4, rtol=0)
        expected = (original[i] * 4 + 1000 * i + 2) / 25
        torch.testing.assert_close(convert_to_seconds(gt.clone(), meta), expected)
        torch.testing.assert_close(convert_to_seconds(gt.clone(), meta, source_axis="selected"), expected)
        torch.testing.assert_close(convert_to_seconds(restored.clone(), meta, source_axis="native"), expected)


@pytest.mark.parametrize("arm", ["g2", "g3"])
def test_physical_head_keeps_dense_gt_and_does_not_inverse_map_twice(arm):
    cfg, output, original, _ = select(arm)
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled
    for i, meta in enumerate(output["metas"]):
        assert meta["irregular_native_axis"]
        assert not meta.get("detector_prediction_inverse_map_required", False)
        torch.testing.assert_close(output["gt_segments"][i], original[i])
        expected = (original[i] * 4 + 1000 * i + 2) / 25
        torch.testing.assert_close(convert_to_seconds(original[i].clone(), meta), expected)


def test_padded_selected_slots_are_not_interpolation_knots():
    _, output, _, _ = select("g0", valid_lengths=(97,))
    meta = output["metas"][0]
    time_map = truetime_map_from_metadata(meta)
    assert time_map.selected_len == 97
    assert meta["irregular_dense_valid_len"] == 97
    endpoints = torch.tensor([[0., 97.]])
    torch.testing.assert_close(time_map.selected_to_true(time_map.true_to_selected(endpoints)), endpoints)
