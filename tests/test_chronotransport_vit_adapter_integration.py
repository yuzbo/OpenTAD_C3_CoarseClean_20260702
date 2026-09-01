from __future__ import annotations

import copy

import pytest
import torch


pytest.importorskip("mmcv")
pytest.importorskip("mmengine")
pytest.importorskip("mmaction")

from opentad.models.backbones.vit_adapter import VisionTransformerAdapter


def _tiny_kwargs() -> dict:
    return dict(
        img_size=16,
        patch_size=16,
        in_channels=3,
        embed_dims=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_frames=4,
        tubelet_size=2,
        use_mean_pooling=True,
        return_feat_map=True,
        with_cp=False,
        adapter_mlp_ratio=0.5,
        total_frames=8,
        adapter_index=[0, 1],
    )


def _runtime_cfg(*, forced_schedule: str, allow_legacy_checkpoint: bool = True) -> dict:
    return dict(
        enabled=True,
        layer_groups=[(0, 1), (1, 2)],
        signal_dims=4,
        risk_hidden_dims=8,
        transport_bottleneck_dims=4,
        risk_quantile=0.9,
        risk_epsilon=1.0,
        max_cache_age=2,
        forced_schedule=forced_schedule,
        cache_detach=True,
        profile_sync_cuda=False,
        measured_cost=None,
        allow_unmeasured_cost_for_debug=False,
        risk_ready=False,
        require_checkpoint_for_dynamic=True,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )


def test_legacy_dense_checkpoint_loads_only_for_forced_stage_a_baseline() -> None:
    dense = VisionTransformerAdapter(**_tiny_kwargs())
    stage_a = VisionTransformerAdapter(
        **_tiny_kwargs(),
        chronotransport=_runtime_cfg(forced_schedule="dense", allow_legacy_checkpoint=True),
    )
    incompatible = stage_a.load_state_dict(dense.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert stage_a.chronotransport_checkpoint_loaded is False
    assert stage_a.chronotransport.checkpoint_loaded is False


def test_learned_checkpoint_guard_rejects_missing_chronotransport_parameters() -> None:
    dense = VisionTransformerAdapter(**_tiny_kwargs())
    learned = VisionTransformerAdapter(
        **_tiny_kwargs(),
        chronotransport=_runtime_cfg(forced_schedule="dense", allow_legacy_checkpoint=False),
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        learned.load_state_dict(dense.state_dict(), strict=True)


def test_forced_dense_runtime_is_exactly_equal_to_original_vit_loop() -> None:
    torch.manual_seed(17)
    dense = VisionTransformerAdapter(**_tiny_kwargs()).eval()
    routed = VisionTransformerAdapter(
        **_tiny_kwargs(),
        chronotransport=_runtime_cfg(forced_schedule="dense"),
    ).eval()
    routed.load_state_dict(dense.state_dict(), strict=True)
    frames = torch.randn(2, 3, 4, 16, 16)
    with torch.no_grad():
        expected = dense(frames.clone())
        actual = routed(frames.clone())
    assert torch.equal(actual, expected)
    assert routed.latest_chronotransport_summary["forced_dense_exact_path"] is True
    assert routed.latest_chronotransport_summary["internal_tubelet_points"] == 4


def test_mixed_runtime_preserves_actual_adapter_shape_and_reduces_heavy_rows() -> None:
    torch.manual_seed(19)
    model = VisionTransformerAdapter(
        **_tiny_kwargs(),
        chronotransport=_runtime_cfg(forced_schedule="periodic2_transport"),
    ).eval()
    frames = torch.randn(2, 3, 4, 16, 16)
    with torch.no_grad():
        output = model(frames)
    assert output.shape == (2, 8, 2, 1, 1)
    summary = model.latest_chronotransport_summary
    assert summary["forced_dense_exact_path"] is False
    assert summary["adapter_path_dense"] is True
    assert summary["heavy_attention_mlp_gathered"] is True
    assert summary["recompute_rows"] < 2 * 2
    assert summary["internal_tubelet_points"] == 4


def test_packed_tubelet_route_and_chronotransport_are_mutually_exclusive() -> None:
    packed = dict(
        enabled=True,
        mode="deterministic_tubelet_cap",
        keep_ratio=0.5,
        min_keep_tubelets=1,
        allow_training_mode=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        VisionTransformerAdapter(
            **_tiny_kwargs(),
            tubelet_packed_runtime_route=packed,
            chronotransport=_runtime_cfg(forced_schedule="dense"),
        )
