import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIX = "_duca_hidden_path_test_runtime"


class _DummyRegistry:
    def register_module(self, *args, **kwargs):
        def _decorator(cls):
            return cls

        return _decorator


class _DummyTrueTimeMap:
    def __init__(self, *args, **kwargs):
        pass

    def remap_segments(self, segments, *args, **kwargs):
        return segments


def _package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module
    return module


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_duca_runtime():
    if f"{RUNTIME_PREFIX}.models.duca.acquisition" in sys.modules:
        return (
            sys.modules[f"{RUNTIME_PREFIX}.models.duca.acquisition"],
            sys.modules[f"{RUNTIME_PREFIX}.models.selectors.duca_online_frame_selector"],
        )

    _package(RUNTIME_PREFIX, ROOT)
    _package(f"{RUNTIME_PREFIX}.models", ROOT / "opentad" / "models")
    duca_pkg = _package(f"{RUNTIME_PREFIX}.models.duca", ROOT / "opentad" / "models" / "duca")
    _package(f"{RUNTIME_PREFIX}.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _package(f"{RUNTIME_PREFIX}.models.utils", ROOT / "opentad" / "models" / "utils")

    builder = types.ModuleType(f"{RUNTIME_PREFIX}.models.builder")
    builder.SELECTORS = _DummyRegistry()
    sys.modules[builder.__name__] = builder

    truetime = types.ModuleType(f"{RUNTIME_PREFIX}.models.utils.truetime_geometry")
    truetime.SELECTED_AXIS = "selected_axis"
    truetime.TRUE_TIME_AXIS = "true_time_dense_index"
    truetime.TrueTimeMap = _DummyTrueTimeMap
    sys.modules[truetime.__name__] = truetime

    _load_module(
        f"{RUNTIME_PREFIX}.models.duca.dynamic_budget",
        ROOT / "opentad" / "models" / "duca" / "dynamic_budget.py",
    )
    acquisition = _load_module(
        f"{RUNTIME_PREFIX}.models.duca.acquisition",
        ROOT / "opentad" / "models" / "duca" / "acquisition.py",
    )
    for name in (
        "C3CoarseProbeActionnessSource",
        "DucaAcquisitionAdapter",
        "ZeroShotActionnessSource",
        "duca_losses",
    ):
        setattr(duca_pkg, name, getattr(acquisition, name))

    selector = _load_module(
        f"{RUNTIME_PREFIX}.models.selectors.duca_online_frame_selector",
        ROOT / "opentad" / "models" / "selectors" / "duca_online_frame_selector.py",
    )
    return acquisition, selector


def _safe_provenance(source_name="fake_c3_probe"):
    return {
        "source_name": source_name,
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "calibration_split": "none",
    }


def test_c3_coarse_probe_actionness_source_exports_hidden_features(monkeypatch):
    acquisition, _selector = _load_duca_runtime()

    class FakeTemporalProbe(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.hidden_dim = int(kwargs["hidden_dim"])

        def forward(self, inputs, valid_mask):
            batch = int(inputs.shape[0])
            temporal = int(inputs.shape[2])
            logits = torch.arange(batch * temporal, dtype=torch.float32, device=inputs.device).reshape(batch, temporal)
            hidden = torch.arange(
                batch * temporal * self.hidden_dim,
                dtype=torch.float32,
                device=inputs.device,
            ).reshape(batch, temporal, self.hidden_dim)
            return {"logits": logits, "coarse_hidden_features": hidden}

    fake_probe_module = SimpleNamespace(
        C3TemporalTCNActionProbe=FakeTemporalProbe,
        prepare_probe_inputs=lambda inputs, probe_model, spatial_size: inputs,
    )
    monkeypatch.setattr(
        acquisition.C3CoarseProbeActionnessSource,
        "_probe_module",
        staticmethod(lambda: fake_probe_module),
    )

    source = acquisition.C3CoarseProbeActionnessSource(
        probe_model="temporal-tcn",
        tcn_hidden_dim=5,
        spatial_size=4,
        frozen=False,
        thumos_trained=False,
        uses_labels=False,
        uses_teacher=False,
        uses_gt=False,
        uses_prediction_cache=False,
        calibration_split="none",
    )
    inputs = torch.randn(2, 3, 4, 4, 4)
    valid_mask = torch.tensor([[True, True, False, True], [True, False, False, True]])

    output = source(inputs, valid_mask=valid_mask)

    assert output["coarse_hidden_features"].shape == (2, 4, 5)
    assert output["coarse_hidden_dim"] == 5
    assert torch.equal(output["coarse_hidden_valid_mask"], valid_mask)
    assert torch.all(output["coarse_hidden_features"][~valid_mask] == 0)


def test_acquisition_adapter_requires_and_projects_coarse_hidden_features():
    acquisition, _selector = _load_duca_runtime()
    DucaAcquisitionAdapter = acquisition.DucaAcquisitionAdapter

    adapter = DucaAcquisitionAdapter(
        feature_dim=2,
        hidden_dim=8,
        budget=2,
        max_radius=0,
        use_coarse_hidden_features=True,
        require_coarse_hidden_features=True,
        coarse_hidden_dim=3,
        coarse_hidden_proj_dim=4,
        coarse_hidden_dropout=0.0,
    )
    dense = torch.zeros(1, 4, 2)
    valid = torch.ones(1, 4, dtype=torch.bool)
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]])

    with pytest.raises(ValueError, match="coarse_hidden_features"):
        adapter.forward_scores(dense, valid_mask=valid, actionness_logits=logits)

    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    scores = adapter.forward_scores(
        dense,
        valid_mask=valid,
        actionness_logits=logits,
        coarse_hidden_features=hidden,
    )

    assert adapter.encoder[1].in_features == 2 + 7 + 4
    assert scores["coarse_hidden_features"].shape == (1, 4, 3)
    assert scores["coarse_hidden_projected"].shape == (1, 4, 4)
    assert scores["selection_features"].shape == (1, 4, 8)


def test_online_selector_passes_probe_hidden_features_to_adapter():
    _acquisition, selector_module = _load_duca_runtime()
    DucaOnlineFrameSelector = selector_module.DucaOnlineFrameSelector

    class FakeOnlineActionness(nn.Module):
        def forward(self, inputs, valid_mask=None, **kwargs):
            batch = int(inputs.shape[0])
            temporal = int(inputs.shape[2])
            valid = valid_mask.to(device=inputs.device, dtype=torch.bool)
            logits = torch.linspace(-1.0, 1.0, steps=temporal, device=inputs.device)[None, :].expand(batch, -1)
            hidden = torch.arange(batch * temporal * 3, dtype=torch.float32, device=inputs.device).reshape(
                batch,
                temporal,
                3,
            )
            return {
                "p_action": torch.sigmoid(logits).masked_fill(~valid, 0.0),
                "logits": logits.masked_fill(~valid, torch.finfo(torch.float32).min / 4.0),
                "actionness_logits": logits.masked_fill(~valid, torch.finfo(torch.float32).min / 4.0),
                "valid_mask": valid,
                "coarse_hidden_features": hidden.masked_fill(~valid[:, :, None], 0.0),
                "coarse_hidden_dim": 3,
                "provenance": _safe_provenance(),
                "source_name": "fake_c3_probe",
                "compute_profile": {"estimated_macs": 0, "estimated_flops": 0, "parameters": {"total": 0, "trainable": 0}},
            }

    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=2,
        dense_window_size=4,
        selector_hidden_channels=8,
        max_radius=0,
        use_coarse_hidden_features=True,
        require_coarse_hidden_features=True,
        coarse_hidden_dim=3,
        coarse_hidden_proj_dim=4,
        coarse_hidden_dropout=0.0,
    )
    selector.raw_actionness_source = FakeOnlineActionness()
    selector.actionness_source_name = "fake_c3_probe"
    inputs = torch.randn(1, 3, 4)
    masks = torch.ones(1, 4, dtype=torch.bool)

    output = selector._forward_select(inputs, masks, metas=None)
    scores = output["selector_outputs"]

    assert scores["coarse_hidden_features"].shape == (1, 4, 3)
    assert scores["coarse_hidden_projected"].shape == (1, 4, 4)
    assert scores["online_actionness_source"] == "fake_c3_probe"
