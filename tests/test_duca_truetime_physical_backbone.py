from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from types import SimpleNamespace

from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.models.backbones.physical_time import (
    PhysicalTimeTubeletEmbedding,
    physical_gap_scaled_depthwise_conv1d,
)
from opentad.models.backbones.vit_adapter import Adapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"


def _conv3d() -> torch.nn.Conv3d:
    torch.manual_seed(7)
    return torch.nn.Conv3d(
        2,
        3,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        bias=True,
    )


def test_physical_tubelet_nominal_path_matches_original_conv3d() -> None:
    projection = _conv3d()
    module = PhysicalTimeTubeletEmbedding(
        embed_dims=3,
        nominal_pair_gap=2.0,
        physical_extent=8.0,
    )
    x = torch.randn(2, 2, 4, 4, 4)
    positions = torch.tensor([[0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0]])
    tokens, midpoint, support, valid = module(x, positions, None, projection)

    expected = projection(x).permute(0, 2, 3, 4, 1).reshape(2, -1, 3)
    torch.testing.assert_close(tokens, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(midpoint, torch.tensor([[1.0, 5.0], [1.0, 5.0]]))
    assert tuple(support.shape) == (2, 2, 2)
    assert bool(valid.all())


def test_physical_tubelet_nonuniform_gap_changes_tokens_when_residual_is_active() -> None:
    projection = _conv3d()
    module = PhysicalTimeTubeletEmbedding(
        embed_dims=3,
        nominal_pair_gap=2.0,
        physical_extent=16.0,
    )
    with torch.no_grad():
        module.gap_residual_scale.fill_(0.5)
    x = torch.randn(1, 2, 4, 4, 4)
    nominal = torch.tensor([[0.0, 2.0, 4.0, 6.0]])
    irregular = torch.tensor([[0.0, 6.0, 8.0, 10.0]])
    nominal_tokens = module(x, nominal, None, projection)[0]
    irregular_tokens = module(x, irregular, None, projection)[0]
    assert not torch.allclose(nominal_tokens, irregular_tokens)


def test_physical_tubelet_rejects_nonincreasing_valid_positions() -> None:
    projection = _conv3d()
    module = PhysicalTimeTubeletEmbedding(embed_dims=3, nominal_pair_gap=2.0, physical_extent=8.0)
    x = torch.randn(1, 2, 4, 4, 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        module(x, torch.tensor([[0.0, 0.0, 4.0, 6.0]]), None, projection)


def test_physical_depthwise_conv_equals_standard_conv_at_nominal_gap() -> None:
    torch.manual_seed(11)
    conv = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=4)
    x = torch.randn(2, 4, 5)
    positions = torch.tensor([[0.0, 4.0, 8.0, 12.0, 16.0]]).repeat(2, 1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    actual = physical_gap_scaled_depthwise_conv1d(
        x,
        conv,
        positions,
        valid,
        nominal_gap=4.0,
    )
    torch.testing.assert_close(actual, conv(x), rtol=1e-6, atol=1e-6)


def test_physical_depthwise_conv_attenuates_cross_gap_neighbor() -> None:
    conv = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    x = torch.tensor([[[1.0, 2.0, 4.0]]])
    nominal = torch.tensor([[0.0, 4.0, 8.0]])
    large_gap = torch.tensor([[0.0, 40.0, 44.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    nominal_out = physical_gap_scaled_depthwise_conv1d(x, conv, nominal, valid, nominal_gap=4.0)
    gap_out = physical_gap_scaled_depthwise_conv1d(x, conv, large_gap, valid, nominal_gap=4.0)
    assert gap_out[0, 0, 0] < nominal_out[0, 0, 0]
    assert gap_out[0, 0, 1] < nominal_out[0, 0, 1]


def test_adapter_merges_clip_batches_in_the_same_order_as_physical_positions() -> None:
    torch.manual_seed(13)
    adapter = Adapter(embed_dims=4, mlp_ratio=0.5, temporal_size=16)
    with torch.no_grad():
        adapter.up_proj.weight.fill_(0.25)
    x = torch.randn(4, 4, 4)
    positions = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    valid = torch.ones_like(positions, dtype=torch.bool)
    legacy = adapter(x, 1, 1)
    physical = adapter(
        x,
        1,
        1,
        temporal_positions=positions,
        temporal_valid_mask=valid,
        nominal_temporal_gap=1.0,
    )
    torch.testing.assert_close(physical, legacy, rtol=1e-6, atol=1e-6)


def test_wrapper_maps_two_k384_rows_to_48_ordered_sixteen_frame_clips() -> None:
    wrapper = BackboneWrapper.__new__(BackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(backbone=SimpleNamespace(physical_time=True))
    frames = torch.zeros(48, 3, 16, 2, 2)
    masks = torch.ones(2, 384, dtype=torch.bool)
    metas = [
        {"irregular_selected_positions": torch.arange(384)},
        {"irregular_selected_positions": torch.arange(384) + 1000},
    ]
    physical = wrapper._physical_backbone_inputs(frames, masks=masks, metas=metas)
    assert tuple(physical["source_positions"].shape) == (48, 16)
    torch.testing.assert_close(physical["source_positions"][0], torch.arange(16).float())
    torch.testing.assert_close(physical["source_positions"][23], torch.arange(368, 384).float())
    torch.testing.assert_close(
        physical["source_positions"][24], torch.arange(1000, 1016).float()
    )
    assert metas[0]["duca_heavy_executed_k"] == 384
    assert metas[1]["duca_heavy_call_boundary"] == "VisionTransformerAdapter.pre_patch_embed"


def test_wrapper_rejects_nonprefix_selected_mask() -> None:
    wrapper = BackboneWrapper.__new__(BackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(backbone=SimpleNamespace(physical_time=True))
    frames = torch.zeros(24, 3, 16, 2, 2)
    masks = torch.ones(1, 384, dtype=torch.bool)
    masks[0, 7] = False
    metas = [{"irregular_selected_positions": torch.arange(384)}]
    with pytest.raises(ValueError, match="valid-prefix"):
        wrapper._physical_backbone_inputs(frames, masks=masks, metas=metas)


@pytest.mark.parametrize(
    "filename,arm,physical_time",
    [
        ("duca_rankpack_k384_curriculum.py", "RANKPACK_K384", False),
        ("duca_truetime_k384_curriculum.py", "TRUETIME_K384", True),
    ],
)
def test_paired_curriculum_configs_are_online_and_20_20_20(
    filename: str,
    arm: str,
    physical_time: bool,
) -> None:
    cfg = Config.fromfile(CONFIG_DIR / filename)
    assert cfg.arm == arm
    assert cfg.physical_time is physical_time
    assert cfg.model.backbone.backbone.physical_time is physical_time
    assert cfg.model.frame_selector.type == "DucaProtectedE2EFrameSelector"
    assert cfg.model.frame_selector.arm == "protected_e2e_homotopy025"
    assert cfg.model.frame_selector.homotopy_warmup_steps == 2000
    assert cfg.model.frame_selector.homotopy_transition_steps == 2000
    assert cfg.model.frame_selector.homotopy_total_steps == 6000
    assert tuple(cfg.duca_curriculum.phase_boundaries) == (20, 40, 60)
    assert tuple(cfg.duca_curriculum.phase_successful_update_boundaries) == (2000, 4000, 6000)
    assert tuple(cfg.duca_curriculum.phase_names) == (
        "semantic_warmup",
        "cosine_homotopy",
        "joint_training",
    )
    assert cfg.duca_curriculum.warmup_detector_sampling == "exact_uniform_k384"
    assert cfg.duca_curriculum.warmup_selector_controls_acquisition is False
    assert cfg.duca_curriculum.warmup_selector_detector_bridge is False
    assert cfg.duca_curriculum.joint_selector_supervision is True
    assert cfg.duca_curriculum.joint_bounded_detector_bridge is True
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.checkpoint_interval == 5
    assert cfg.workflow.primary_checkpoint_state_key == "state_dict_ema"
    assert cfg.workflow.primary_checkpoint_epoch == 59
    assert cfg.workflow.checkpoint_criterion == "terminal_epoch_59_state_dict_ema"
    assert tuple(cfg.duca_curriculum.resume_state) == (
        "model",
        "optimizer",
        "scheduler",
        "amp_scaler",
        "epoch",
        "successful_update",
        "rng",
        "selector_schedule_step",
    )
    assert cfg.dataset.train.type == "DucaStatelessThumosPaddingDataset"
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.experiment_scope.repeats_dense_uniform_random is False
    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val is None
    assert cfg.dataset.test.subset_name == "validation"
    assert cfg.evaluation.subset == "validation"
    if physical_time:
        assert cfg.experiment_scope.decode_before_nms is True
        assert cfg.experiment_scope.pre_nms_coordinate_space == "true_time_dense_index"


def test_true_time_optimizer_trains_only_adapters_and_physical_time_module() -> None:
    cfg = Config.fromfile(CONFIG_DIR / "duca_truetime_k384_curriculum.py")
    custom_names = [entry.name for entry in cfg.optimizer.backbone.custom]
    assert custom_names == ["adapter", "physical_time_embedding"]
    assert cfg.optimizer.backbone.lr == 0
    assert cfg.optimizer.backbone.exclude == ["backbone"]
    assert cfg.optimizer.backbone.custom[0].lr == pytest.approx(2e-4)
    assert cfg.optimizer.backbone.custom[1].lr == pytest.approx(2e-4)
