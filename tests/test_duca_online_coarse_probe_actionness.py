from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


def _selector() -> DucaOnlineFrameSelector:
    return DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        max_radius=2,
        dense_window_size=8,
        selector_hidden_channels=8,
        detector_gradient_mode="st_sparse_gather_soft_context",
        profile_runtime=True,
        actionness_source_cfg={
            "type": "C3CoarseProbeActionnessSource",
            "source_name": "online_c3_official_asformer_coarse_actionness",
            "probe_model": "official-action-seg",
            "official_action_seg_backend": "official_asformer",
            "spatial_size": 16,
            "tcn_hidden_dim": 16,
            "official_num_layers": 1,
            "dropout": 0.0,
            "frozen": False,
            "trainable": True,
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
        },
        loss_weights={
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
    )


def test_online_c3_official_asformer_probe_produces_actionness_profile() -> None:
    selector = _selector()
    inputs = torch.randn(1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])

    profile = out["selector_outputs"]["compute_profile"]
    assert out["metas"][0]["duca_online_actionness_source"] == "online_c3_official_asformer_coarse_actionness"
    assert profile["actionness"]["source_kind"] == "task_adapted_coarse_classifier"
    assert profile["actionness"]["probe_model"] == "official-action-seg"
    assert profile["actionness"]["official_action_seg_backend"] == "official_asformer"
    assert profile["actionness"]["model_family"] == "OfficialActionSeg/official_asformer"
    assert profile["actionness"]["online_backbone_flops_included"] is True
    assert profile["estimated_flops"] >= profile["actionness"]["estimated_flops"]
    assert profile["latency_ms"]["coarse_probe_ms"] >= 0.0


def test_detector_path_can_backprop_into_online_coarse_probe() -> None:
    selector = _selector()
    inputs = torch.randn(1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])
    loss = out["inputs"].float().pow(2).mean()
    loss.backward()

    grads = [
        param.grad.detach().abs().sum().item()
        for param in selector.raw_actionness_source.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert grads
    assert sum(grads) > 0.0


def test_online_selector_accepts_uint8_window_tensor_from_full_train_loader() -> None:
    selector = _selector()
    inputs = torch.randint(0, 255, (1, 1, 3, 8, 16, 16), dtype=torch.uint8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    out = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert out["inputs"].shape[3] == 4
    assert out["inputs"].is_floating_point()
