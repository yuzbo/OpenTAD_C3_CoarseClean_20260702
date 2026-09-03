from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from opentad.models.backbones.d2s_videomae_wrapper import (
    D2STemporalZoomBackboneWrapper,
)
from opentad.models.detectors.actionformer import ActionFormer
from tools.bata.zoomtoken_batch_device import prepare_zoomtoken_batch


def _bare_wrapper() -> D2STemporalZoomBackboneWrapper:
    wrapper = D2STemporalZoomBackboneWrapper.__new__(D2STemporalZoomBackboneWrapper)
    nn.Module.__init__(wrapper)
    wrapper.total_chunks = 48
    wrapper.burst_chunks = 16
    wrapper.tubelets_per_chunk = 8
    wrapper.local_size = 128
    wrapper.source_height = 180
    wrapper.source_width = 320
    wrapper.crop_box = [96, 26, 224, 154]
    return wrapper


def test_route_selects_exactly_16_distinct_chunks():
    wrapper = _bare_wrapper()
    features = torch.randn(2, 48, 32)
    selected, outputs = wrapper._route_chunks(features)
    assert selected.shape == (2, 16)
    assert outputs["repr_shift"].shape == (2, 48)
    assert torch.equal(selected, selected.sort(dim=-1).values)
    assert all(row.unique().numel() == 16 for row in selected)


def test_physical_skip_executes_only_selected_local_chunks(monkeypatch):
    wrapper = _bare_wrapper()

    class IdentityPreprocessor:
        def preprocess(self, values, data_samples=None, training=False):
            return torch.stack(values), None

    wrapper.model = SimpleNamespace(data_preprocessor=IdentityPreprocessor())
    seen = {}

    def fake_native_ragged(selected_native_tubelets, physical_indices):
        seen["shape"] = tuple(selected_native_tubelets.shape)
        seen["physical_indices"] = physical_indices.clone()
        return torch.ones(
            selected_native_tubelets.shape[0],
            selected_native_tubelets.shape[1],
            12,
        )

    monkeypatch.setattr(
        wrapper,
        "_run_shared_backbone_native_ragged",
        fake_native_ragged,
    )
    source = torch.zeros(1, 1, 3, 768, 180, 320, dtype=torch.uint8)
    selected = torch.arange(0, 32, 2).reshape(1, 16)
    features = wrapper._encode_selected_local_chunks(source, selected)
    assert seen["shape"] == (1, 8192, 3, 2, 16, 16)
    physical_indices = seen["physical_indices"]
    assert physical_indices.shape == (1, 8192)
    assert torch.equal(
        physical_indices[:, 1:],
        physical_indices[:, 1:].sort(dim=-1).values,
    )
    assert physical_indices.unique().numel() == 8192
    assert physical_indices[0, 511].item() == 511
    assert physical_indices[0, 512].item() == 1024
    assert features.shape == (1, 12, 128)


def test_token_budget_is_exactly_22016_of_38400():
    global_tokens = 48 * 8 * 36
    local_tokens = 16 * 8 * 64
    dense_tokens = 48 * 8 * 100
    assert global_tokens + local_tokens == 22016
    assert (global_tokens + local_tokens) / dense_tokens == pytest.approx(
        0.5733333333333334
    )


def test_batch_device_move_keeps_only_source_native_video_on_cpu():
    batch = {
        "inputs": {
            "global": torch.zeros(1, 1, 3, 16, 96, 96, dtype=torch.uint8),
            "source": torch.zeros(1, 1, 3, 16, 180, 320, dtype=torch.uint8),
        },
        "masks": torch.ones(1, 16, dtype=torch.bool),
        "gt_segments": [torch.zeros(1, 2)],
    }
    moved = prepare_zoomtoken_batch(batch, torch.device("meta"))
    assert moved["inputs"]["source"].device.type == "cpu"
    assert moved["inputs"]["global"].device.type == "meta"
    assert moved["masks"].device.type == "meta"
    assert moved["gt_segments"][0].device.type == "meta"


def test_actionformer_pads_all_bundle_features_on_one_axis():
    detector = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(detector)
    detector.max_seq_len = 16
    detector.max_div_factor = 4
    bundle = {
        "global_features": torch.ones(2, 4, 12),
        "residual_features": torch.ones(2, 4, 12) * 2,
        "feats": torch.ones(2, 4, 12) * 3,
    }
    masks = torch.ones(2, 12, dtype=torch.bool)
    padded, padded_masks = detector.pad_data(bundle, masks)
    assert padded_masks.shape == (2, 16)
    assert all(value.shape == (2, 4, 16) for value in padded.values())
    assert torch.count_nonzero(padded["residual_features"][..., 12:]) == 0
