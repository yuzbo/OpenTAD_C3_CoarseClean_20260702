from __future__ import annotations

from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models.duca import DucaAcquisitionAdapter, TrueTimeFeatureResidual, ZeroShotActionnessSource
from opentad.models.duca.acquisition import C3CoarseProbeActionnessSource
from tools.bata import duca_selected_axis_training


ROOT = Path(__file__).resolve().parents[1]


def test_t1_variants_are_registered_for_selected_axis_runtime_binding() -> None:
    assert duca_selected_axis_training.VARIANT_CONFIGS[
        "t1_true_time_residual_g0"
    ] == "duca_t1_true_time_residual_g0_fixed384_official60.py"
    assert duca_selected_axis_training.VARIANT_CONFIGS[
        "t1_reversed_time_residual_g0"
    ] == "duca_t1_reversed_time_residual_g0_fixed384_official60.py"


def test_true_time_residual_is_exact_identity_at_initialization_and_trainable() -> None:
    module = TrueTimeFeatureResidual(feature_dim=8, hidden_dim=6, descriptor_mode="actual")
    features = torch.randn(2, 8, 4, requires_grad=True)
    masks = torch.ones(2, 4, dtype=torch.bool)
    metas = [
        {
            "selected_axis_to_true_time_dense_index": [0, 2, 5, 7],
            "truetime_dense_valid_len": 8,
        },
        {
            "selected_axis_to_true_time_dense_index": [1, 3, 4, 7],
            "truetime_dense_valid_len": 9,
        },
    ]
    output = module(features, masks, metas)
    torch.testing.assert_close(output, features, atol=0.0, rtol=0.0)
    output.square().sum().backward()
    assert module.projector[-1].weight.grad is not None
    assert float(module.projector[-1].weight.grad.abs().sum()) > 0.0


def test_reversed_true_time_control_changes_only_descriptor_alignment() -> None:
    features = torch.zeros(1, 4, 3)
    masks = torch.ones(1, 3, dtype=torch.bool)
    metas = [{"selected_axis_to_true_time_dense_index": [0, 3, 7], "truetime_dense_valid_len": 8}]
    actual = TrueTimeFeatureResidual(4, descriptor_mode="actual")._descriptors(features, masks, metas)
    reversed_code = TrueTimeFeatureResidual(4, descriptor_mode="reversed")._descriptors(features, masks, metas)
    torch.testing.assert_close(reversed_code[:, :3], torch.flip(actual[:, :3], dims=(1,)))


def test_parameter_free_feature_change_peaks_at_hidden_state_transition() -> None:
    hidden = torch.zeros(1, 8, 4)
    hidden[:, :4, 0] = 1.0
    hidden[:, 4:, 1] = 1.0
    valid = torch.ones(1, 8, dtype=torch.bool)
    payload = C3CoarseProbeActionnessSource._parameter_free_evidence(
        hidden,
        valid,
        class_logits=None,
        mode="frozen_feature_change",
    )
    peak = int(payload["feature_change"].argmax(dim=1).item())
    assert peak in {3, 4}


def test_parameter_free_r2q3_reuses_exact_k_max_hole_decoder() -> None:
    source = ZeroShotActionnessSource(
        mode="motion",
        source_name="test_parameter_free_prior",
        thumos_trained=False,
        uses_labels=False,
        uses_teacher=False,
        uses_gt=False,
        uses_prediction_cache=False,
        calibration_split="none",
    )
    adapter = DucaAcquisitionAdapter(
        feature_dim=None,
        actionness_source=source,
        budget=4,
        acquisition_policy="global_structured_topk",
        selector_variant="direct_boundary",
        parameter_free_selector=True,
        transition_objective="boundary_burst",
        boundary_burst_radius=1,
        boundary_burst_quota=3,
        boundary_burst_budget_fraction=0.75,
        boundary_burst_require_bilateral_offsets=True,
        boundary_burst_require_global_mandatory_groups=True,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )
    evidence = torch.tensor([[0.01, 0.02, 0.10, 0.99, 0.10, 0.02, 0.01, 0.01]])
    dense = torch.zeros(1, 8, 3)
    grid, scores = adapter.acquire(dense, p_action=evidence)
    assert int(grid.selected_count.item()) == 4
    assert scores["parameter_free_selector"] is True
    selected = set(grid.selected_positions[0].tolist())
    assert {2, 3, 4}.issubset(selected)


def test_parameter_free_r2q3_is_feasible_at_formal_k384_g2_scale() -> None:
    source = ZeroShotActionnessSource(
        mode="motion",
        source_name="test_parameter_free_formal_scale",
        thumos_trained=False,
        uses_labels=False,
        uses_teacher=False,
        uses_gt=False,
        uses_prediction_cache=False,
        calibration_split="none",
    )
    adapter = DucaAcquisitionAdapter(
        feature_dim=None,
        actionness_source=source,
        budget=384,
        acquisition_policy="global_structured_topk",
        selector_variant="direct_boundary",
        parameter_free_selector=True,
        transition_objective="boundary_burst",
        boundary_burst_radius=2,
        boundary_burst_quota=3,
        boundary_burst_budget_fraction=0.25,
        boundary_burst_require_bilateral_offsets=True,
        boundary_burst_require_global_mandatory_groups=True,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )
    time = torch.arange(768, dtype=torch.float32)
    evidence = (
        0.5
        + 0.2 * torch.sin(time / 17.0)
        + 0.15 * torch.sin(time / 41.0)
    ).clamp(1.0e-4, 1.0 - 1.0e-4)[None]
    dense = torch.zeros(1, 768, 3)
    grid, _scores = adapter.acquire(dense, p_action=evidence)
    assert int(grid.selected_count.item()) == 384
    positions = grid.selected_positions[0]
    assert int((positions[1:] - positions[:-1] - 1).max().item()) <= 2


def test_train_free_configs_are_frozen_and_target_label_free() -> None:
    configs = {
        "trainfree_mobilenet_feature_change": "duca_trainfree_fixed384_official60_base.py",
        "trainfree_mobilenet_semantic": "duca_trainfree_mobilenet_semantic_fixed384_official60.py",
        "trainfree_mobilenet_fusion_r2q3": "duca_trainfree_mobilenet_fusion_r2q3_fixed384_official60.py",
        "trainfree_slowfast_fast_fusion_r2q3": "duca_trainfree_slowfast_fast_fusion_r2q3_fixed384_official60.py",
    }
    for variant, name in configs.items():
        cfg = Config.fromfile(str(ROOT / "configs" / "adatad" / "thumos" / name))
        selector = cfg.model.frame_selector
        source = selector.actionness_source_cfg
        assert selector.parameter_free_selector is True
        assert selector.detector_gradient_mode == "none"
        assert source.frozen is True and source.trainable is False
        assert source.train_split_supervised is False
        assert source.uses_labels is False and source.uses_gt is False
        assert source.calibration_split == "none"
        assert cfg.workflow.formal_protocol == duca_selected_axis_training.FORMAL_PROTOCOL
        contract = duca_selected_axis_training.formal_training_contract(cfg)
        assert contract is not None
        assert contract["end_epoch"] == 60
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert contract["checkpoint_criterion"] == "terminal_epoch_59_state_dict_ema"
        assert duca_selected_axis_training.VARIANT_CONFIGS[variant] == name
