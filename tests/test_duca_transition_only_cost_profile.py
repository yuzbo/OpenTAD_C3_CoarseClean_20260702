from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)


MODULE_NAME = "tools.bata.profile_duca_transition_only_cost"


def _profile_module():
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing transition-only cost profiler: {exc}")


class _FakeGrid:
    def __init__(self) -> None:
        self.selected_positions = torch.tensor([[0, 2, 5, 7]], dtype=torch.long)


class _FakeAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transition_scorer = nn.Linear(3, 1)


class _FakeTransitionSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_actionness_source = nn.Linear(2, 3)
        self.adapter = _FakeAdapter()
        self.calls = 0

    def forward_test(self, *, inputs, masks, metas):
        self.calls += 1
        assert inputs.device.type == "cpu"
        assert masks.dtype == torch.bool
        return {
            "inputs": inputs[:, :, [0, 2, 5, 7]],
            "masks": masks[:, [0, 2, 5, 7]],
            "metas": metas,
            "selector_outputs": {
                "grid": _FakeGrid(),
                "compute_profile": {
                    "estimated_macs": 123,
                    "estimated_flops": 456,
                    "estimated_flops_are_lower_bound": True,
                    "complete_memory_accounting": False,
                    "components": {
                        "actionness": {"estimated_macs": 100, "estimated_flops": 200},
                        "selector": {"estimated_macs": 23, "estimated_flops": 46},
                    },
                },
                "provenance": {
                    "source_name": "shared_official_asformer_binary_actionness",
                    "hidden_kind": "official_asformer_encoder_hidden",
                    "official_source_file": "/external/ASFormer/model.py",
                    "official_source_sha256": "source-sha256",
                    "official_source_normalized_lf_sha256": "normalized-source-sha256",
                },
                "selector_variant": "transition_only",
                "selection_path": "learned_global_structured",
            },
        }


def test_profile_selector_reports_auditable_selector_only_cost() -> None:
    profiler = _profile_module()
    selector = _FakeTransitionSelector()
    inputs = torch.rand(1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)

    report = profiler.profile_selector(
        selector,
        inputs=inputs,
        masks=masks,
        warmup=1,
        repeats=3,
        device=torch.device("cpu"),
        config_provenance={"path": "/repo/config.py", "sha256": "config-sha256"},
    )

    assert selector.calls == 4
    assert selector.training is False
    assert report["schema_version"] == "duca-transition-only-cost-v1"
    assert report["accounting_scope"]["selector_only"] is True
    assert report["accounting_scope"]["pre_backbone_only"] is True
    assert report["accounting_scope"]["detector_backbone_included"] is False
    assert report["accounting_scope"]["detector_head_included"] is False
    assert report["accounting_scope"]["full_stack_cost_claim_allowed"] is False
    assert "official ASFormer coarse probe" in report["accounting_scope"]["includes"]
    assert report["input_shape"] == [1, 3, 8, 4, 4]
    assert report["selected_count"] == 4
    assert report["selected_count_per_sample"] == [4]
    assert report["max_gap"] == 2
    assert report["max_gap_per_sample"] == [2]
    assert report["max_gap_definition"] == "maximum_unselected_hole_including_prefix_and_suffix"
    assert report["static_estimate"]["estimated_macs"] == 123
    assert report["static_estimate"]["estimated_flops"] == 456
    assert report["static_estimate"]["is_lower_bound"] is True
    assert report["static_estimate"]["source"] == "selector_forward_compute_profile"
    assert report["latency_ms"]["sample_count"] == 3
    assert report["latency_ms"]["median"] >= 0.0
    assert report["latency_ms"]["p90"] >= report["latency_ms"]["median"]
    assert report["cuda_peak_memory"]["available"] is False
    assert report["parameters"]["selector_total"]["total"] == 13
    assert report["parameters"]["coarse_probe"]["total"] == 9
    assert report["parameters"]["transition_scorer"]["total"] == 4
    assert report["provenance"]["config"]["sha256"] == "config-sha256"
    assert report["provenance"]["coarse_probe"]["hidden_kind"] == "official_asformer_encoder_hidden"
    json.dumps(report, allow_nan=False)


def test_build_selector_from_config_applies_only_explicit_smoke_overrides(monkeypatch) -> None:
    profiler = _profile_module()
    captured = {}
    selector_cfg = {
        "type": "DucaOnlineFrameSelector",
        "selector_variant": "transition_only",
        "dense_window_size": 768,
        "budget": 384,
        "max_unselected_hole": 15,
        "profile_runtime": True,
        "profile_sync_cuda": True,
        "actionness_source_cfg": {
            "type": "C3CoarseProbeActionnessSource",
            "probe_model": "official-action-seg",
            "official_action_seg_backend": "official_asformer",
            "spatial_size": 64,
            "tcn_hidden_dim": 96,
        },
    }
    fake_cfg = SimpleNamespace(
        duca_transition_only_contract=SimpleNamespace(
            selector_variant="transition_only",
            coarse_hidden_kind="official_asformer_encoder_hidden",
        ),
        model=SimpleNamespace(frame_selector=selector_cfg),
    )
    monkeypatch.setattr(profiler.Config, "fromfile", lambda _path: fake_cfg)

    sentinel = object()

    def fake_build_selector(cfg):
        captured.update(cfg)
        return sentinel

    monkeypatch.setattr(profiler, "build_selector", fake_build_selector)

    selector, metadata = profiler.build_transition_only_selector(
        "/repo/config.py",
        temporal_len=16,
        budget=8,
        probe_spatial_size=12,
    )

    assert selector is sentinel
    assert captured["selector_variant"] == "transition_only"
    assert captured["dense_window_size"] == 16
    assert captured["budget"] == 8
    assert captured["profile_runtime"] is False
    assert captured["profile_sync_cuda"] is False
    assert captured["actionness_source_cfg"]["spatial_size"] == 12
    assert captured["actionness_source_cfg"]["tcn_hidden_dim"] == 96
    assert metadata["configured_temporal_len"] == 768
    assert metadata["configured_budget"] == 384
    assert metadata["profile_temporal_len"] == 16
    assert metadata["profile_budget"] == 8
    assert metadata["scaled_smoke"] is True


@pytest.mark.parametrize(
    ("temporal_len", "budget", "message"),
    [(0, 1, "temporal_len"), (8, 0, "budget"), (8, 9, "cannot exceed")],
)
def test_build_selector_rejects_invalid_smoke_geometry(
    monkeypatch,
    temporal_len: int,
    budget: int,
    message: str,
) -> None:
    profiler = _profile_module()
    fake_cfg = SimpleNamespace(
        duca_transition_only_contract=SimpleNamespace(
            selector_variant="transition_only",
            coarse_hidden_kind="official_asformer_encoder_hidden",
        ),
        model=SimpleNamespace(
            frame_selector={
                "type": "DucaOnlineFrameSelector",
                "selector_variant": "transition_only",
                "dense_window_size": 768,
                "budget": 384,
                "max_unselected_hole": 15,
                "actionness_source_cfg": {
                    "type": "C3CoarseProbeActionnessSource",
                    "probe_model": "official-action-seg",
                    "official_action_seg_backend": "official_asformer",
                    "spatial_size": 64,
                },
            }
        ),
    )
    monkeypatch.setattr(profiler.Config, "fromfile", lambda _path: fake_cfg)

    with pytest.raises(ValueError, match=message):
        profiler.build_transition_only_selector(
            "/repo/config.py",
            temporal_len=temporal_len,
            budget=budget,
            probe_spatial_size=16,
        )
