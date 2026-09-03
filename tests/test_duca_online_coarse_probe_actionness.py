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
from opentad.models.duca.structured_selection import global_structured_topk


def _selector(
    *,
    frozen: bool = False,
    calibration_temperature: float = 1.0,
    calibration_bias: float = 0.0,
    selector_variant: str = "direct_boundary",
    acquisition_policy: str | None = None,
    detector_gradient_mode: str = "st_sparse_gather_soft_context",
    budget: int = 4,
    transition_objective: str = "gaussian_mass",
    require_bilateral_burst: bool = False,
    require_global_mandatory_burst: bool = False,
) -> DucaOnlineFrameSelector:
    uses_structured_policy = (
        selector_variant == "transition_only" or acquisition_policy == "global_structured_topk"
    )
    return DucaOnlineFrameSelector(
        in_channels=3,
        budget=budget,
        max_radius=2,
        dense_window_size=8,
        selector_hidden_channels=8,
        selector_variant=selector_variant,
        acquisition_policy=(
            acquisition_policy
            if acquisition_policy is not None
            else "global_structured_topk" if selector_variant == "transition_only" else "legacy_center_radius"
        ),
        max_unselected_hole=(3 if uses_structured_policy else None),
        hard_max_gap_repair=not uses_structured_policy,
        forbid_external_actionness=(selector_variant == "transition_only"),
        detector_gradient_mode=detector_gradient_mode,
        transition_objective=transition_objective,
        transition_boundary_radius=2,
        boundary_burst_quota=3.0,
        boundary_burst_budget_fraction=0.5,
        boundary_burst_context_weight=0.05,
        boundary_burst_require_bilateral_offsets=require_bilateral_burst,
        boundary_burst_require_global_mandatory_groups=(
            require_global_mandatory_burst
        ),
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


@pytest.mark.parametrize("probability_dtype", [torch.float16, torch.bfloat16])
def test_adapter_accepts_quantized_probability_from_declared_calibration(
    probability_dtype,
) -> None:
    selector = _selector(calibration_temperature=2.0, calibration_bias=0.7)
    provenance = selector.raw_actionness_source._provenance()
    logits = torch.tensor([[0.1234, -0.9876, 1.2345, -2.3456]])
    probability = torch.sigmoid((logits + 0.7) / 2.0).to(probability_dtype)
    probability[0, -1] = 0.5

    scores = selector.adapter.forward_scores(
        torch.zeros(1, 4, 3),
        valid_mask=torch.tensor([[True, True, True, False]]),
        actionness_logits=logits,
        p_action=probability,
        actionness_provenance=provenance,
    )

    assert torch.isfinite(scores["p_action"]).all()


def test_adapter_rejects_nonfinite_calibrated_probability() -> None:
    selector = _selector()
    provenance = selector.raw_actionness_source._provenance()
    logits = torch.zeros(1, 4)
    probability = torch.full_like(logits, 0.5)
    probability[0, 1] = float("nan")

    with pytest.raises(FloatingPointError, match="non-finite"):
        selector.adapter.forward_scores(
            torch.zeros(1, 4, 3),
            valid_mask=torch.ones(1, 4, dtype=torch.bool),
            actionness_logits=logits,
            p_action=probability,
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
    request = out["counterfactual_request"]
    assert request is not None
    assert not request["candidate_valid"].any()
    loss = selector.counterfactual_distillation_loss(
        out["selector_outputs"],
        request["candidate_positions"],
        request["replaced_slots"],
        torch.zeros_like(request["candidate_valid"], dtype=torch.float32),
        request["candidate_valid"],
        baseline_detector_loss=torch.full((1,), float("nan")),
        candidate_detector_loss=torch.full_like(
            request["candidate_valid"],
            float("nan"),
            dtype=torch.float32,
        ),
    )
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.0)
    summary = selector.last_counterfactual_summary
    assert summary["utility_consistency_max_abs_error"] == 0.0
    assert summary["candidate_count"] == 0
    assert summary["spearman"] == 0.0
    assert summary["sign_agreement"] == 0.0
    assert summary["finite"] is True
    loss.backward()
    assert any(param.grad is not None for param in selector.adapter.transition_scorer.parameters())


@pytest.mark.parametrize("selector_variant", ["direct_boundary", "transition_only"])
def test_structured_surrogate_matches_each_mixed_length_hard_feasible_family(selector_variant: str) -> None:
    selector = _selector(
        selector_variant=selector_variant,
        acquisition_policy="global_structured_topk",
        detector_gradient_mode="structured_zero_forward",
    )
    selector.train()
    out = selector.forward_train(
        inputs=torch.randn(2, 1, 3, 8, 16, 16),
        masks=torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        metas=[{"video_name": "full"}, {"video_name": "short"}],
        gt_segments=[torch.tensor([[1.0, 6.0]]), torch.tensor([[0.0, 2.0]])],
        gt_labels=[torch.tensor([1]), torch.tensor([1])],
    )
    state = out["selector_outputs"]
    slots = state["structured_soft_slot_assignment"]
    mass = slots.sum(dim=-1)

    assert torch.allclose(mass[0], torch.ones_like(mass[0]), atol=1.0e-5)
    assert torch.allclose(mass[1, :3], torch.ones_like(mass[1, :3]), atol=1.0e-5)
    assert torch.equal(mass[1, 3:], torch.zeros_like(mass[1, 3:]))
    assert torch.equal(slots[1, :, 3:], torch.zeros_like(slots[1, :, 3:]))

    expected = global_structured_topk(
        state["decode_policy_logits"][1:2, :3],
        k=3,
        max_unselected_hole=3,
        temperature=0.7,
        training=True,
    )
    assert torch.allclose(slots[1, :3, :3], expected.soft_slot_assignment[0], atol=1.0e-6)
    assert torch.allclose(state["soft_coverage"][1, :3], expected.soft_occupancy[0], atol=1.0e-6)
    assert torch.equal(state["soft_coverage"][1, 3:], torch.zeros_like(state["soft_coverage"][1, 3:]))


def test_boundary_burst_soft_and_hard_arms_share_local_utility_only() -> None:
    torch.manual_seed(37)
    common = dict(
        selector_variant="transition_only",
        acquisition_policy="global_structured_topk",
        detector_gradient_mode="structured_zero_forward",
        budget=6,
        transition_objective="boundary_burst",
        require_bilateral_burst=True,
    )
    soft = _selector(**common, require_global_mandatory_burst=False)
    hard = _selector(**common, require_global_mandatory_burst=True)
    hard.load_state_dict(soft.state_dict())
    soft.train()
    hard.train()
    inputs = torch.randn(1, 1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{"video_name": "burst"}]
    gt_segments = [torch.tensor([[2.0, 6.0]])]
    gt_labels = [torch.tensor([1])]

    soft_state = soft.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )["selector_outputs"]
    hard_state = hard.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )["selector_outputs"]

    local_fields = (
        "center_scores",
        "decode_policy_logits",
        "boundary_burst_mass",
        "boundary_burst_utility",
        "boundary_burst_center_probabilities",
        "boundary_burst_offset_probabilities",
        "boundary_burst_offset_inclusion",
        "boundary_burst_effective_offset_quota",
        "boundary_burst_context_reference",
        "boundary_burst_bilateral_offset_feasible",
        "boundary_burst_bilateral_offset_satisfied",
    )
    for field in local_fields:
        assert torch.equal(soft_state[field], hard_state[field]), field

    assert soft.adapter.boundary_burst_require_bilateral_offsets is True
    assert hard.adapter.boundary_burst_require_bilateral_offsets is True
    assert soft.adapter.boundary_burst_require_global_mandatory_groups is False
    assert hard.adapter.boundary_burst_require_global_mandatory_groups is True
    assert soft_state["boundary_burst_local_bilateral_utility_enabled"] is True
    assert hard_state["boundary_burst_local_bilateral_utility_enabled"] is True
    assert soft_state["boundary_burst_global_mandatory_groups_enabled"] is False
    assert hard_state["boundary_burst_global_mandatory_groups_enabled"] is True
    assert int(soft_state["mandatory_boundary_mask"].sum().item()) == 0
    assert int(hard_state["mandatory_boundary_mask"].sum().item()) == 3
    assert soft_state["mandatory_boundary_group_count"] == [0]
    assert hard_state["mandatory_boundary_group_count"] == [1]


def test_boundary_burst_adapter_preserves_mandatory_group_in_final_hard_positions() -> None:
    selector = _selector(
        selector_variant="transition_only",
        acquisition_policy="global_structured_topk",
        detector_gradient_mode="structured_zero_forward",
        budget=6,
        transition_objective="boundary_burst",
        require_bilateral_burst=True,
        require_global_mandatory_burst=True,
    )
    selector.train()
    out = selector.forward_train(
        inputs=torch.randn(1, 1, 3, 8, 16, 16),
        masks=torch.ones(1, 8, dtype=torch.bool),
        metas=[{"video_name": "burst"}],
        gt_segments=[torch.tensor([[2.0, 6.0]])],
        gt_labels=[torch.tensor([1])],
    )
    state = out["selector_outputs"]
    mandatory = state["mandatory_boundary_mask"]

    assert int(mandatory.sum().item()) == 3
    assert torch.all(state["grid"].selected_mask | ~mandatory)
    assert torch.all(state["soft_coverage"][mandatory] > 0.999)
    assert state["mandatory_boundary_group_count"] == [1]


def test_direct_all_short_structured_slots_keep_active_mass_contract() -> None:
    selector = _selector(
        selector_variant="direct_boundary",
        acquisition_policy="global_structured_topk",
        detector_gradient_mode="structured_zero_forward",
    )
    selector.train()
    out = selector.forward_train(
        inputs=torch.randn(1, 1, 3, 8, 16, 16),
        masks=torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool),
        metas=[{"video_name": "direct-short"}],
        gt_segments=[torch.tensor([[0.0, 2.0]])],
        gt_labels=[torch.tensor([1])],
    )
    slots = out["selector_outputs"]["structured_soft_slot_assignment"]
    mass = slots.sum(dim=-1)
    assert torch.allclose(mass[:, :3], torch.ones_like(mass[:, :3]), atol=1.0e-5)
    assert torch.equal(mass[:, 3:], torch.zeros_like(mass[:, 3:]))


def test_online_selector_pads_physical_slots_for_short_valid_window() -> None:
    selector = _selector()
    inputs = torch.randn(1, 1, 3, 8, 16, 16)
    masks = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_name": "short"}])

    assert out["inputs"].shape[3] == 4
    assert out["masks"].shape == (1, 4)
    assert out["masks"].sum().item() == 3
    assert out["metas"][0]["duca_online_selected_count"] == 3
