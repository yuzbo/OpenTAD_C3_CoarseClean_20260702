import os

import torch

from mmengine.config import Config

from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.models.backbones.vit_adapter import Adapter
from opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector import (
    PCOTMRASPreBackboneFrameSelector,
)


def _selector(*, strategy, target_len, dynamic_budget=None, variable=False):
    return PCOTMRASPreBackboneFrameSelector(
        target_len=target_len,
        dense_window_size=64,
        descriptor_dim=3 * 32 * 32,
        selection_strategy=strategy,
        scout_feature_source="compressed_pixels",
        scout_spatial_size=32,
        remap_gt_to_selected_axis=False,
        physical_dense_reconstruction=True,
        variable_length_output=variable,
        variable_compute_multiple=16,
        dynamic_budget=dynamic_budget,
        frame_score_st_surrogate=(
            "global_rank_topk" if strategy == "frame_score_global_rank_st" else "local_softmax"
        ),
        reader=dict(
            type="PCOTMRASBoundaryDifficultyTemporalFrameScout",
            in_dim=3 * 32 * 32,
            hidden_dim=16,
            num_slots=target_len,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
    )


def test_dynamic_selector_emits_real_variable_length_without_padding():
    selector = _selector(
        strategy="dynamic_B",
        target_len=48,
        variable=True,
        dynamic_budget=dict(
            enabled=True,
            min_budget=16,
            target_budget=32,
            max_budget=48,
            average_budget=32,
            budget_step=16,
        ),
    )
    inputs = torch.randn(1, 1, 3, 64, 8, 8)
    masks = torch.ones(1, 64, dtype=torch.bool)
    out = selector.forward_test(inputs, masks, metas=[{"fps": 25.0}])
    selected_len = int(out["inputs"].shape[3])
    assert selected_len in (16, 32, 48)
    assert selected_len == out["metas"][0]["irregular_selected_count"]
    assert out["masks"].shape == (1, 64)
    assert out["metas"][0]["duca_sparse_variable_compute"] is True
    positions = out["metas"][0]["duca_sparse_physical_positions"]
    assert positions == sorted(set(positions))
    assert len(positions) == selected_len


def test_dynamic_budget_stays_clip_aligned_after_short_window_clamp():
    selector = _selector(
        strategy="dynamic_B",
        target_len=64,
        variable=True,
        dynamic_budget=dict(
            enabled=True,
            min_budget=16,
            target_budget=32,
            max_budget=64,
            average_budget=32,
            budget_step=16,
            actionness_weight=1.0,
            boundary_weight=0.0,
            uncertainty_weight=0.0,
            redundancy_weight=0.0,
        ),
    )
    candidate_valid = torch.ones(1, 63, dtype=torch.bool)
    reader_outputs = {"frame_selection_logits": torch.full((1, 63), 20.0)}
    plan = selector._dynamic_budget_plan(
        reader_outputs=reader_outputs,
        frame_scores=reader_outputs["frame_selection_logits"],
        candidate_valid=candidate_valid,
    )
    assert int(plan["budgets"][0]) == 48
    assert plan["metadata"][0]["valid_len"] == 63


def test_uniform_control_is_exact_k_and_uses_same_dense_reconstruction_contract():
    selector = _selector(strategy="uniform_exact_k", target_len=32)
    inputs = torch.randn(1, 1, 3, 64, 8, 8)
    masks = torch.ones(1, 64, dtype=torch.bool)
    out = selector.forward_test(inputs, masks, metas=[{}])
    positions = out["metas"][0]["duca_sparse_physical_positions"]
    assert out["inputs"].shape[3] == 32
    assert len(positions) == len(set(positions)) == 32
    assert positions[0] == 0 and positions[-1] == 63
    assert out["masks"].shape == (1, 64)


def test_fixed_controls_keep_exact_k_on_short_official_windows():
    inputs = torch.randn(1, 1, 3, 64, 8, 8)
    masks = torch.zeros(1, 64, dtype=torch.bool)
    masks[:, :31] = True

    for strategy in ("uniform_exact_k", "frame_score_global_rank_st"):
        selector = _selector(strategy=strategy, target_len=32)
        out = selector.forward_test(inputs, masks, metas=[{}])
        positions = out["metas"][0]["duca_sparse_physical_positions"]
        assert out["inputs"].shape[3] == 32
        assert len(positions) == len(set(positions)) == 32
        assert positions == sorted(positions)
        assert out["metas"][0]["irregular_dense_valid_len"] == 31
        assert out["masks"].shape == (1, 64)


def test_irregular_physical_time_interpolation_preserves_anchor_values():
    features = torch.tensor([[[1.0, 3.0, 7.0]]])
    positions = torch.tensor([0.0, 2.0, 6.0])
    dense = BackboneWrapper._interpolate_irregular_time(features, positions, 7)
    assert dense.shape == (1, 1, 7)
    assert torch.allclose(dense[0, 0, positions.long()], features[0, 0])
    assert torch.allclose(dense[0, 0, 1], torch.tensor(2.0))
    assert torch.allclose(dense[0, 0, 4], torch.tensor(5.0))


def test_adapter_accepts_shorter_packed_temporal_group():
    adapter = Adapter(embed_dims=8, mlp_ratio=0.5, temporal_size=32, allow_variable_temporal_size=True)
    x = torch.randn(4, 8, 8)
    out = adapter(x, h=2, w=2)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_all_matrix_arms_parse_from_one_official_config(monkeypatch):
    path = "configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py"
    for arm in ("official_dense", "uniform_k384", "learned_k384", "dynamic_k", "dynamic_k_no_risk"):
        monkeypatch.setenv("DUCA_ARM", arm)
        monkeypatch.setenv("DUCA_SEED", "3407")
        cfg = Config.fromfile(path)
        assert cfg.experiment_scope.arm == arm
        assert cfg.workflow.end_epoch == 60
        assert cfg.dataset.train.subset_name == "training"
        assert cfg.dataset.val.subset_name == "validation"
        if arm == "official_dense":
            assert "frame_selector" not in cfg.model
        else:
            assert cfg.model.frame_selector.physical_dense_reconstruction is True
