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
from opentad.models.duca.acquisition import C3CoarseProbeActionnessSource


def _selector(
    *,
    frozen: bool = False,
    calibration_temperature: float = 1.0,
    calibration_bias: float = 0.0,
    selector_variant: str = "actionness",
) -> DucaOnlineFrameSelector:
    return DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        max_radius=2,
        dense_window_size=8,
        selector_hidden_channels=8,
        selector_variant=selector_variant,
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
            "frozen": frozen,
            "trainable": not frozen,
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "calibration_temperature": calibration_temperature,
            "calibration_bias": calibration_bias,
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


def test_train_only_calibration_without_fit_artifact_fails_closed() -> None:
    with pytest.raises(ValueError, match="real calibration_artifact"):
        C3CoarseProbeActionnessSource(
            probe_model="official-action-seg",
            spatial_size=16,
            tcn_hidden_dim=16,
            official_num_layers=1,
            calibration_split="train_only",
        )


def test_frozen_temperature_bias_is_the_unique_probability_and_transition_source() -> None:
    selector = _selector(calibration_temperature=2.0, calibration_bias=0.7)
    inputs = torch.randn(1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)

    scores = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "v"}])["selector_outputs"]
    expected = torch.sigmoid((scores["raw_actionness_logits"] + 0.7) / 2.0)
    expected_delta = torch.zeros_like(expected)
    expected_delta[:, 1:] = expected[:, 1:] - expected[:, :-1]

    assert torch.allclose(scores["p_action"], expected)
    assert torch.allclose(scores["delta_p_action"], expected_delta)
    assert torch.allclose(scores["abs_delta_p_action"], expected_delta.abs())
    expected_entropy = -(expected * expected.log() + (1.0 - expected) * (1.0 - expected).log())
    expected_uncertainty = 1.0 - (2.0 * expected - 1.0).abs()
    assert torch.allclose(scores["entropy"], expected_entropy)
    assert torch.allclose(scores["uncertainty"], expected_uncertainty)
    assert torch.allclose(scores["calibrated_actionness_logits"], torch.logit(expected))
    assert torch.allclose(scores["actionness_logits"], scores["raw_actionness_logits"])
    assert scores["provenance"]["calibration_temperature"] == 2.0
    assert scores["provenance"]["calibration_bias"] == 0.7
    assert scores["online_actionness_provenance"] == scores["provenance"]


def test_adapter_rejects_probability_inconsistent_with_declared_calibration() -> None:
    selector = _selector(calibration_temperature=2.0, calibration_bias=0.7)
    provenance = selector.raw_actionness_source._provenance()
    logits = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="does not match the calibration"):
        selector.adapter.forward_scores(
            torch.zeros(1, 8, 3),
            valid_mask=torch.ones(1, 8, dtype=torch.bool),
            actionness_logits=logits,
            p_action=torch.full_like(logits, 0.5),
            actionness_provenance=provenance,
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
    assert profile["components"]["selector"]["uses_coarse_hidden_features"] is True
    assert out["selector_outputs"]["uses_coarse_hidden_features"] is True
    assert out["selector_outputs"]["coarse_hidden_features"].shape == (1, 8, 16)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_official_asformer_probe_stays_fp32_under_outer_autocast() -> None:
    selector = _selector().cuda()
    inputs = torch.randn(1, 3, 8, 16, 16, device="cuda")
    masks = torch.ones(1, 8, dtype=torch.bool, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = selector.raw_actionness_source(inputs, valid_mask=masks)
        loss = outputs["actionness_logits"].float().square().mean()

    assert outputs["actionness_logits"].dtype == torch.float32
    assert outputs["coarse_hidden_features"].dtype == torch.float32
    (loss * 65536.0).backward()
    gradients = [
        parameter.grad
        for parameter in selector.raw_actionness_source.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_frozen_coarse_probe_stays_in_eval_when_parent_enters_train_mode() -> None:
    selector = _selector(frozen=True)

    selector.train()

    assert selector.training is True
    assert selector.raw_actionness_source.training is False
    assert selector.raw_actionness_source.probe.module.training is False
    assert all(not param.requires_grad for param in selector.raw_actionness_source.parameters())


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


def test_all_short_counterfactual_batch_keeps_static_loss_graph() -> None:
    selector = _selector(selector_variant="transition_only")
    selector.counterfactual_utility_distillation_weight = 0.25
    inputs = torch.randn(1, 1, 3, 8, 16, 16)
    masks = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    out = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "short"}],
        gt_segments=[torch.tensor([[0.0, 2.0]])],
        gt_labels=[torch.tensor([1])],
    )
    loss = out["losses"]["counterfactual_utility_distillation_loss"]
    assert out["counterfactual_request"] is None
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert any(param.grad is not None for param in selector.adapter.transition_scorer.parameters())


def test_online_selector_pads_physical_slots_for_short_valid_window() -> None:
    selector = _selector()
    inputs = torch.randn(1, 1, 3, 8, 16, 16)
    masks = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "short"}])

    assert out["inputs"].shape[3] == 4
    assert out["masks"].shape == (1, 4)
    assert out["masks"].sum().item() == 3
    assert out["metas"][0]["duca_online_selected_count"] == 3
