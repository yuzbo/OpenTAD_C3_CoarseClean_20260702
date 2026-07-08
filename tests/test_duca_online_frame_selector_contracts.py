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
    return DucaOnlineFrameSelector(
        in_channels=2,
        budget=budget,
        max_radius=0,
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
    assert positions == [1, 3, 5, 7]
    assert torch.allclose(out["gt_segments"][0], torch.tensor([[0.0, 3.0], [0.5, 2.5]]), atol=1e-4)
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
    assert positions == [1, 3, 5]
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
