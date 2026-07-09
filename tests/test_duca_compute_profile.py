from __future__ import annotations

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


def _motion_selector(
    *,
    profile_runtime: bool = True,
    detector_gradient_mode: str = "st_sparse_gather",
) -> DucaOnlineFrameSelector:
    return DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        max_radius=2,
        selector_hidden_channels=8,
        dense_window_size=8,
        detector_gradient_mode=detector_gradient_mode,
        profile_runtime=profile_runtime,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
    )


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


def test_duca_selector_reports_compute_and_latency_profile_for_motion_source() -> None:
    selector = _motion_selector(profile_runtime=True)
    inputs = torch.randn(1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])

    profile = out["selector_outputs"]["compute_profile"]
    assert profile["pre_backbone_model"] == "ZeroShotActionnessSource(mode=motion)+DUCASelectorMLP"
    assert profile["actionness"]["source_name"] == "zero_shot_motion_actionness"
    assert profile["input_shape"] == [1, 8, 3]
    assert profile["parameters"]["trainable"] > 0
    assert profile["estimated_macs"] > 0
    assert profile["estimated_flops"] >= profile["estimated_macs"]
    assert profile["latency_ms"]["total_selector_ms"] >= 0.0
    assert profile["latency_ms"]["descriptor_ms"] >= 0.0
    assert out["metas"][0]["duca_online_compute_profile"]["estimated_flops"] == profile["estimated_flops"]
    assert selector.last_forward_summary["compute_profile"]["estimated_flops"] == profile["estimated_flops"]


def test_soft_to_hard_resample_profile_accounts_slotwise_temporal_bridge() -> None:
    selector = _motion_selector(profile_runtime=True, detector_gradient_mode="soft_to_hard_resample")
    inputs = torch.randn(1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])

    profile = out["selector_outputs"]["compute_profile"]
    bridge = profile["components"]["soft_to_hard_resample"]
    assert profile["detector_gradient_bridge"]["mode"] == "soft_to_hard_resample"
    assert bridge["enabled"] is True
    assert bridge["slot_count"] == 4
    assert bridge["dense_temporal_len"] == 8
    assert bridge["feature_dim"] == 3
    assert bridge["estimated_macs"] == 1 * 4 * 8 * 3
    assert bridge["estimated_flops"] >= bridge["estimated_macs"]
    assert profile["estimated_flops"] >= bridge["estimated_flops"]
    assert out["metas"][0]["duca_online_compute_profile"]["components"]["soft_to_hard_resample"]["enabled"] is True


def test_duca_selector_marks_external_x3d_actionness_as_cached_prior_in_profile() -> None:
    selector = _motion_selector(profile_runtime=True)
    selector.external_actionness_meta_key = "duca_external_p_action"
    selector.external_actionness_logits_meta_key = "duca_external_actionness_logits"
    selector.external_actionness_provenance_meta_key = "duca_external_actionness_provenance"
    selector.external_actionness_source_meta_key = "duca_external_actionness_source"
    selector.require_external_actionness = True
    p_action = [0.99, 0.01, 0.98, 0.02, 0.97, 0.03, 0.96, 0.04]
    inputs = torch.randn(1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)

    out = selector.forward_test(
        inputs=inputs,
        masks=masks,
        metas=[
            {
                "video_name": "v",
                "duca_external_p_action": p_action,
                "duca_external_actionness_logits": torch.logit(torch.tensor(p_action).clamp(1e-6, 1 - 1e-6)).tolist(),
                "duca_external_actionness_source": "frozen_kinetics_x3d_xs_actionness",
                "duca_external_actionness_provenance": _x3d_provenance(),
            }
        ],
    )

    profile = out["selector_outputs"]["compute_profile"]
    assert profile["actionness"]["source_name"] == "frozen_kinetics_x3d_xs_actionness"
    assert profile["actionness"]["source_kind"] == "external_cached_prior"
    assert profile["actionness"]["online_backbone_flops_included"] is False
    assert profile["actionness"]["cache_lookup_or_interpolation"] is True
    assert out["metas"][0]["duca_online_compute_profile"]["actionness"]["source_kind"] == "external_cached_prior"
