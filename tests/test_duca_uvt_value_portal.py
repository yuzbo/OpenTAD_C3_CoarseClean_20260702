"""Focused DUCA-UVT value-portal tests with an isolated OpenTAD import runtime."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class _Registry:
    def __init__(self):
        self._module_dict = {}

    def register_module(self, name=None, module=None):
        if module is not None:
            self._module_dict[name] = module
            return module

        def _decorator(cls):
            self._module_dict[name or cls.__name__] = cls
            return cls

        return _decorator

    def build(self, cfg):
        cfg = dict(cfg)
        cls = self._module_dict[cfg.pop("type")]
        return cls(**cfg)


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_duca_uvt_runtime():
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.losses", ROOT / "opentad" / "models" / "losses")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.NECKS = _Registry()
    builder.build_selector = builder.SELECTORS.build
    sys.modules["opentad.models.builder"] = builder

    _load_module(
        "opentad.models.selectors.duca_utility_geometry_targets",
        ROOT / "opentad" / "models" / "selectors" / "duca_utility_geometry_targets.py",
    )
    _load_module(
        "opentad.models.selectors.duca_value_head_group",
        ROOT / "opentad" / "models" / "selectors" / "duca_value_head_group.py",
    )
    _load_module(
        "opentad.models.selectors.duca_value_ema",
        ROOT / "opentad" / "models" / "selectors" / "duca_value_ema.py",
    )
    _load_module(
        "opentad.models.losses.duca_value_learning_losses",
        ROOT / "opentad" / "models" / "losses" / "duca_value_learning_losses.py",
    )
    _load_module(
        "opentad.models.selectors.duca_dynamic_physical",
        ROOT / "opentad" / "models" / "selectors" / "duca_dynamic_physical.py",
    )
    selector_module = _load_module(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py",
    )
    return (
        selector_module,
        sys.modules["opentad.models.selectors.duca_utility_geometry_targets"],
        sys.modules["opentad.models.selectors.duca_value_head_group"],
        sys.modules["opentad.models.selectors.duca_value_ema"],
        sys.modules["opentad.models.losses.duca_value_learning_losses"],
    )


(
    selector_module,
    geometry_module,
    head_module,
    ema_module,
    losses_module,
) = _load_duca_uvt_runtime()

PCOTMRASPreBackboneFrameSelector = selector_module.PCOTMRASPreBackboneFrameSelector
build_geometry_value_target = geometry_module.build_geometry_value_target
DucaValueHeadGroup = head_module.DucaValueHeadGroup
DucaValueEMA = ema_module.DucaValueEMA
build_value_learning_losses = losses_module.build_value_learning_losses
geometry_value_loss = losses_module.geometry_value_loss
self_ema_value_distill_loss = losses_module.self_ema_value_distill_loss


def _selector(mode="geo", decoder=True):
    return PCOTMRASPreBackboneFrameSelector(
        target_len=32,
        dense_window_size=64,
        descriptor_dim=3 * 32 * 32,
        selection_strategy="dynamic_B",
        scout_feature_source="compressed_pixels",
        scout_spatial_size=32,
        remap_gt_to_selected_axis=False,
        physical_dense_reconstruction=True,
        variable_length_output=True,
        variable_compute_multiple=16,
        dynamic_budget=dict(
            enabled=True,
            min_budget=16,
            target_budget=32,
            max_budget=32,
            average_budget=32,
            budget_step=16,
            score_midpoint=0.5,
            actionness_weight=1.0,
            boundary_weight=0.0,
            uncertainty_weight=0.0,
            redundancy_weight=0.0,
        ),
        reader=dict(
            type="PCOTMRASBoundaryDifficultyTemporalFrameScout",
            in_dim=3 * 32 * 32,
            hidden_dim=16,
            num_slots=32,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        value_mode=mode,
        value_hidden_dim=16,
        value_alpha=0.1 if mode != "off" else 0.0,
        value_ema_enabled=mode in ("ema", "geo_ema", "geo_ema_portal"),
        value_geometry_weight=1.0 if mode in ("geo", "geo_ema", "geo_ema_portal") else 0.0,
        value_ema_loss_weight=0.1 if mode in ("ema", "geo_ema", "geo_ema_portal") else 0.0,
        boundary_quota=4 if decoder else 0,
        boundary_center_top_m=2 if decoder else 0,
        boundary_radius_decode=1,
        boundary_pair_max_gap=8,
        mmr_lambda=0.1 if decoder else 0.0,
    )


def test_geometry_target_has_signed_order_and_zero_mean():
    gt = torch.tensor([[[2.0, 8.0], [20.0, 25.0]]])
    valid = torch.ones(1, 32, dtype=torch.bool)
    target = build_geometry_value_target(
        gt_segments=gt,
        gt_labels=None,
        valid_mask=valid,
        boundary_radius=1,
        short_action_duration_sec=20.0,
        short_action_weight=0.5,
    )
    endpoint_mean = target.frame_target[0, 2].item()
    interior_mean = target.frame_target[0, 5].item()
    background_mean = target.frame_target[0, 30].item()
    assert endpoint_mean > interior_mean > background_mean
    assert abs(target.frame_target[0, valid[0]].mean().item()) < 1e-6
    assert target.pair_weight.shape == (1, 32)


def test_value_head_group_output_shape_and_ema_detach():
    head = DucaValueHeadGroup(hidden_dim=8)
    ema = DucaValueEMA(head, decay=0.9)
    x = torch.randn(2, 6, 8)
    valid = torch.tensor([[True, True, True, True, False, False], [True] * 6])
    out = head(x, valid)
    assert out.value.shape == (2, 6)
    target = ema.detach_targets(x, valid)
    assert not target.requires_grad
    ema.update(head)
    assert all(not p.requires_grad for p in ema.ema_head.parameters())


def test_value_losses_are_finite_and_audited():
    head = DucaValueHeadGroup(hidden_dim=8)
    x = torch.randn(1, 8, 8)
    valid = torch.ones(1, 8, dtype=torch.bool)
    out = head(x, valid)
    gt = torch.tensor([[[1.0, 4.0]]])
    target = build_geometry_value_target(
        gt_segments=gt, gt_labels=None, valid_mask=valid,
        boundary_radius=1, short_action_duration_sec=10.0, short_action_weight=0.5,
    )
    ema = DucaValueEMA(head, decay=0.9)
    ema_target = ema.detach_targets(x, valid)
    geo = geometry_value_loss(out.value, target, valid)
    distill = self_ema_value_distill_loss(out.value, ema_target, valid, lambda_scale=0.5)
    assert torch.isfinite(geo) and torch.isfinite(distill)
    bundle = build_value_learning_losses(
        value=out.value, valid=valid, geometry_target=target, ema_target=ema_target,
        geometry_weight=1.0, ema_weight=0.2, portal_enabled=False,
    )
    assert torch.isfinite(bundle.geometry_value_loss)
    assert torch.isfinite(bundle.self_ema_value_distill_loss)
    assert bundle.metadata["uses_dense_detector_teacher"] is False


def test_foveated_decoder_returns_exact_k_unique_sorted():
    selector = _selector(mode="off", decoder=True)
    scores = torch.randn(1, 64)
    boundary = torch.zeros(1, 64)
    boundary[0, [10, 20, 50]] = 3.0
    valid = torch.ones(64, dtype=torch.bool)
    positions = selector._boundary_foveated_positions(
        scores=scores, boundary_logits=boundary, valid=valid, k=32
    )
    assert len(positions) == 32
    assert positions == sorted(set(positions))


def test_off_mode_is_score_level_legacy():
    selector = _selector(mode="off", decoder=False)
    frame_scores = torch.randn(1, 64)
    valid = torch.ones(1, 64, dtype=torch.bool)
    combined = selector._combined_selection_score({}, frame_scores, valid)
    assert torch.equal(combined, frame_scores)
