from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn

import pytest


def _install_mmaction_registry_shim() -> None:
    if "mmaction.registry" in sys.modules:
        return

    class _RegistryShim:
        def register_module(self, *args, **kwargs):
            def _decorator(cls):
                return cls

            if args and isinstance(args[0], type):
                return args[0]
            return _decorator

    mmaction = types.ModuleType("mmaction")
    mmaction.__version__ = "test-shim"
    mmaction.__path__ = []
    registry = types.ModuleType("mmaction.registry")
    registry.MODELS = _RegistryShim()
    utils = types.ModuleType("mmaction.utils")
    utils.ConfigType = dict
    utils.OptConfigType = object
    models = types.ModuleType("mmaction.models")
    models.__path__ = []
    backbones = types.ModuleType("mmaction.models.backbones")
    backbones.__path__ = []
    swin = types.ModuleType("mmaction.models.backbones.swin")
    vit_mae = types.ModuleType("mmaction.models.backbones.vit_mae")

    class _UnusedBackboneShim(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def _unused(*args, **kwargs):
        raise RuntimeError("mmaction swin shim should not be executed in TrueTime focused tests")

    swin.PatchEmbed3D = _UnusedBackboneShim
    swin.PatchMerging = _UnusedBackboneShim
    swin.WindowAttention3D = _UnusedBackboneShim
    swin.Mlp = _UnusedBackboneShim
    swin.get_window_size = _unused
    swin.compute_mask = _unused
    swin.window_partition = _unused
    swin.window_reverse = _unused
    vit_mae.get_sinusoid_encoding = _unused
    nms_1d_cpu = types.ModuleType("nms_1d_cpu")
    nms_1d_cpu.nms = _unused
    nms_1d_cpu.softnms = _unused
    align_1d = types.ModuleType("Align1D")
    align_1d.forward = _unused
    align_1d.backward = _unused
    boundary_max_pooling_cuda = types.ModuleType("boundary_max_pooling_cuda")
    boundary_max_pooling_cuda.forward = _unused
    boundary_max_pooling_cuda.backward = _unused

    sys.modules["mmaction"] = mmaction
    sys.modules["mmaction.registry"] = registry
    sys.modules["mmaction.utils"] = utils
    sys.modules["mmaction.models"] = models
    sys.modules["mmaction.models.backbones"] = backbones
    sys.modules["mmaction.models.backbones.swin"] = swin
    sys.modules["mmaction.models.backbones.vit_mae"] = vit_mae
    sys.modules.setdefault("nms_1d_cpu", nms_1d_cpu)
    sys.modules.setdefault("Align1D", align_1d)
    sys.modules.setdefault("boundary_max_pooling_cuda", boundary_max_pooling_cuda)


_install_mmaction_registry_shim()

from opentad.models.utils.truetime_geometry import (
    TrueTimeMap,
    inverse_map_prediction_segments,
    remap_selected_axis_segments_to_true_time,
    truetime_map_from_metadata,
)


def test_truetime_selected_dense_roundtrip_preserves_fractional_positions() -> None:
    time_map = TrueTimeMap(selected_positions=[0, 2, 5, 9], dense_len=10, valid_len=10)

    selected_axis = torch.tensor([0.0, 0.5, 1.5, 3.0])
    true_time = time_map.selected_to_true(selected_axis)
    roundtrip = time_map.true_to_selected(true_time)

    assert torch.allclose(true_time, torch.tensor([0.0, 1.0, 3.5, 9.0]))
    assert torch.allclose(roundtrip, selected_axis, atol=1e-5)
    assert time_map.selected_axis_name == "selected_axis_index"
    assert time_map.true_time_axis_name == "true_time_dense_index"


def test_truetime_segment_remap_is_explicit_about_coordinate_spaces() -> None:
    time_map = TrueTimeMap(selected_positions=[0, 2, 5, 9], dense_len=10, valid_len=10)
    selected_segments = torch.tensor([[0.0, 1.0], [1.5, 3.0]])

    true_segments = time_map.remap_segments(
        selected_segments,
        source_coordinate_space="selected_axis_index",
        target_coordinate_space="true_time_dense_index",
    )
    selected_roundtrip = time_map.remap_segments(
        true_segments,
        source_coordinate_space="true_time_dense_index",
        target_coordinate_space="selected_axis_index",
    )

    assert torch.allclose(true_segments, torch.tensor([[0.0, 2.0], [3.5, 9.0]]))
    assert torch.allclose(selected_roundtrip, selected_segments, atol=1e-5)


def test_prediction_inverse_map_records_selected_axis_source() -> None:
    time_map = TrueTimeMap(selected_positions=[1, 3, 6, 7], dense_len=8, valid_len=8)
    predictions = {
        "segments": torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        "scores": torch.tensor([0.9, 0.2]),
        "coordinate_space": "selected_axis_index",
    }

    mapped = inverse_map_prediction_segments(predictions, time_map)

    assert mapped["coordinate_space"] == "true_time_dense_index"
    assert mapped["source_coordinate_space"] == "selected_axis_index"
    assert torch.allclose(mapped["segments"], torch.tensor([[1.0, 6.0], [3.0, 7.0]]))
    assert torch.equal(mapped["scores"], predictions["scores"])


def test_metadata_selected_axis_remap_preserves_ordering() -> None:
    meta = {
        "detector_prediction_inverse_map_required": True,
        "selected_axis_to_true_time_dense_index": [1, 4, 8, 9],
        "truetime_dense_len": 10,
        "irregular_dense_valid_len": torch.tensor([10.0]),
        "irregular_selected_valid_len": [4.0],
        "irregular_selected_count": 4,
    }
    selected_segments = torch.tensor([[0.0, 1.0], [1.25, 2.5], [2.5, 3.0]])

    time_map = truetime_map_from_metadata(meta)
    true_segments = remap_selected_axis_segments_to_true_time(selected_segments, meta)

    assert torch.allclose(true_segments, torch.tensor([[1.0, 4.0], [5.0, 8.5], [8.5, 9.0]]))
    assert torch.all(true_segments[1:, 0] >= true_segments[:-1, 0])
    assert time_map.selected_len == 4


def test_metadata_selected_axis_remap_fails_closed_when_required_mapping_is_missing() -> None:
    meta = {
        "detector_prediction_inverse_map_required": True,
        "truetime_dense_len": 10,
        "irregular_dense_valid_len": 10,
        "irregular_selected_valid_len": 4,
    }

    with pytest.raises(ValueError, match="selected_axis_to_true_time_dense_index"):
        remap_selected_axis_segments_to_true_time(torch.tensor([[0.0, 1.0]]), meta)
