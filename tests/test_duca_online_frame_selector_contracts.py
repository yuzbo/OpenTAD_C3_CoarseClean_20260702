from __future__ import annotations

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


def _manual_actionness_cfg(p_action: torch.Tensor, *, no_target: bool = True) -> dict:
    cfg = {
        "type": "ZeroShotActionnessSource",
        "mode": "manual",
        "p_action": p_action,
        "uncertainty": 1.0 - torch.abs(2.0 * p_action - 1.0),
        "source_name": "unit_test_manual_actionness",
    }
    if no_target:
        cfg.update(
            {
                "thumos_trained": False,
                "uses_labels": False,
                "uses_teacher": False,
                "uses_gt": False,
                "uses_prediction_cache": False,
                "calibration_split": "none",
                "checkpoint_hash": "unit-test-no-target",
            }
        )
    return cfg


def _selector(p_action: torch.Tensor, *, budget: int = 4, no_target: bool = True, **kwargs) -> DucaOnlineFrameSelector:
    max_radius = int(kwargs.pop("max_radius", 0))
    return DucaOnlineFrameSelector(
        in_channels=2,
        budget=budget,
        max_radius=max_radius,
        dense_window_size=p_action.shape[1],
        actionness_source_cfg=_manual_actionness_cfg(p_action, no_target=no_target),
        loss_weights={
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
        **kwargs,
    )


def test_duca_selector_rejects_runtime_budget_above_hard_cap() -> None:
    p_action = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]])
    selector = _selector(p_action, budget=4)
    inputs = torch.randn(1, 2, 6)
    masks = torch.ones(1, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="budget override exceeds hard cap"):
        selector.forward_train(
            inputs=inputs,
            masks=masks,
            metas=[{"video_name": "v"}],
            gt_segments=[torch.tensor([[1.0, 5.0]])],
            gt_labels=[torch.tensor([0])],
            budget=5,
        )


def test_duca_selector_remaps_train_gt_to_selected_axis() -> None:
    p_action = torch.tensor([[0.1, 0.95, 0.2, 0.85, 0.3, 0.75, 0.4, 0.65]])
    selector = _selector(p_action, budget=4)
    inputs = torch.arange(1 * 2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=[torch.tensor([[1.0, 7.0], [2.0, 6.0]])],
        gt_labels=[torch.tensor([0, 1])],
    )

    positions = out["metas"][0]["duca_online_selected_positions"]
    assert positions == [1, 2, 3, 4]
    assert torch.allclose(out["gt_segments"][0], torch.tensor([[0.0, 3.0], [1.0, 3.0]]), atol=1e-4)
    assert out["metas"][0]["gt_segments_original_time"][0] == [1.0, 7.0]
    assert out["metas"][0]["gt_segments_selected_axis"][0] == pytest.approx([0.0, 3.0])


def test_duca_selector_metadata_matches_actual_gather_and_overwrites_reserved_keys() -> None:
    p_action = torch.tensor([[0.1, 0.95, 0.2, 0.85, 0.3, 0.75]])
    selector = _selector(
        p_action,
        budget=3,
        metadata_keys={
            "selected_positions": "custom_selected_positions",
            "selected_positions_unit": "custom_selected_positions_unit",
            "selected_mask": "custom_selected_mask",
            "selected_count": "custom_selected_count",
            "remap": "custom_remap",
            "source": "custom_source",
        },
    )
    inputs = torch.arange(1 * 2 * 6, dtype=torch.float32).reshape(1, 2, 6)
    masks = torch.ones(1, 6, dtype=torch.bool)
    stale_meta = {
        "video_name": "v",
        "custom_selected_positions": [999],
        "custom_selected_count": 999,
        "custom_remap": {"stale": True},
    }

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[stale_meta])

    positions = out["metas"][0]["custom_selected_positions"]
    assert positions == [1, 2, 4]
    assert out["metas"][0]["custom_selected_count"] == 3
    assert "selected_to_original" in out["metas"][0]["custom_remap"]
    assert torch.equal(out["inputs"], inputs[:, :, positions])
    assert out["masks"].tolist() == [[True, True, True]]


def test_duca_selector_rejects_unknown_manual_actionness_provenance_for_deploy_path() -> None:
    p_action = torch.tensor([[0.1, 0.95, 0.2, 0.85, 0.3, 0.75]])
    selector = _selector(p_action, budget=3, no_target=False)
    inputs = torch.randn(1, 2, 6)
    masks = torch.ones(1, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="actionness provenance"):
        selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])


def _x3d_provenance() -> dict:
    return {
        "source_name": "frozen_kinetics_x3d_xs_actionness",
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "calibration_split": "none",
        "checkpoint_hash": "pytorch_provider:x3d_xs:pretrained=True",
    }


def test_duca_selector_uses_external_x3d_p_action_from_metas() -> None:
    internal = torch.tensor([[0.05, 0.95, 0.10, 0.90, 0.15, 0.85]])
    external = [0.99, 0.01, 0.98, 0.02, 0.97, 0.03]
    selector = _selector(
        internal,
        budget=3,
        external_actionness_meta_key="duca_external_p_action",
        external_actionness_logits_meta_key="duca_external_actionness_logits",
        external_actionness_provenance_meta_key="duca_external_actionness_provenance",
        require_external_actionness=True,
    )
    inputs = torch.arange(1 * 2 * 6, dtype=torch.float32).reshape(1, 2, 6)
    masks = torch.ones(1, 6, dtype=torch.bool)

    out = selector.forward_test(
        inputs=inputs,
        masks=masks,
        metas=[
            {
                "video_name": "v",
                "duca_external_p_action": external,
                "duca_external_actionness_logits": torch.logit(torch.tensor(external).clamp(1e-6, 1 - 1e-6)).tolist(),
                "duca_external_actionness_provenance": _x3d_provenance(),
            }
        ],
    )

    assert out["metas"][0]["duca_online_selected_positions"] == [1, 2, 4]
    assert out["metas"][0]["duca_online_actionness_source"] == "frozen_kinetics_x3d_xs_actionness"
    assert out["selector_outputs"]["p_action"].detach().cpu().tolist()[0] == pytest.approx(external)


def test_duca_selector_requires_external_x3d_p_action_when_configured() -> None:
    internal = torch.tensor([[0.05, 0.95, 0.10, 0.90]])
    selector = _selector(
        internal,
        budget=2,
        external_actionness_meta_key="duca_external_p_action",
        external_actionness_provenance_meta_key="duca_external_actionness_provenance",
        require_external_actionness=True,
    )

    with pytest.raises(ValueError, match="external actionness"):
        selector.forward_test(
            inputs=torch.randn(1, 2, 4),
            masks=torch.ones(1, 4, dtype=torch.bool),
            metas=[{"video_name": "missing"}],
        )


def test_main_duca_selector_forbids_cached_external_actionness_payloads_even_when_unconfigured() -> None:
    internal = torch.tensor([[0.05, 0.95, 0.10, 0.90]])
    external = [0.99, 0.01, 0.98, 0.02]
    selector = _selector(
        internal,
        budget=2,
        forbid_external_actionness=True,
    )

    with pytest.raises(ValueError, match="forbids external actionness"):
        selector.forward_test(
            inputs=torch.randn(1, 2, 4),
            masks=torch.ones(1, 4, dtype=torch.bool),
            metas=[
                {
                    "video_name": "v",
                    "duca_external_p_action": external,
                    "duca_external_actionness_logits": torch.logit(
                        torch.tensor(external).clamp(1e-6, 1 - 1e-6)
                    ).tolist(),
                    "duca_external_actionness_provenance": _x3d_provenance(),
                }
            ],
        )


def test_soft_to_hard_resample_keeps_hard_forward_but_gives_neighbor_frames_gradient() -> None:
    p_action = torch.tensor([[0.01, 0.02, 0.80, 0.99, 0.75, 0.02]], dtype=torch.float32)
    selector = _selector(
        p_action,
        budget=1,
        max_radius=2,
        detector_gradient_mode="soft_to_hard_resample",
    )
    inputs = torch.arange(1 * 2 * 6, dtype=torch.float32).reshape(1, 2, 6).requires_grad_(True)
    masks = torch.ones(1, 6, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])
    selected = out["metas"][0]["duca_online_selected_positions"]
    loss = out["inputs"].sum()
    loss.backward()

    assert selected == [2]
    assert torch.equal(out["inputs"].detach(), inputs.detach()[:, :, selected])
    weights = out["selector_outputs"]["soft_resample_weights"]
    assert weights.shape == (1, 1, 6)
    assert weights[0, 0].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert weights[0, 0, 1].item() > 0.0
    assert weights[0, 0, 2].item() > 0.0
    assert weights[0, 0, 3].item() > 0.0
    assert inputs.grad[0, :, 1].abs().sum().item() > 0.0
    assert inputs.grad[0, :, 2].abs().sum().item() > 0.0
    assert inputs.grad[0, :, 3].abs().sum().item() > 0.0
