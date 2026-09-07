"""Regression for time parameters re-enabled outside GradScaler's optimizer."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_config_registers_trainable_time_parameters():
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/duca_evidence_recovery_base.py"))
    groups = {group.name: group for group in cfg.optimizer.backbone.custom}
    assert groups["continuous_timestamp_conditioner"].lr == 2e-4
    assert groups["relative_physical_time_scale"].lr == 2e-4
    assert groups["relative_physical_time_scale"].weight_decay == 0.0
    assert cfg.solver.static_graph is False
    assert cfg.solver.find_unused_parameters is True


def _torch():
    if os.name == "nt":
        pytest.skip("Windows torch/c10.dll unavailable; run runtime tests on N16R4")
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"torch runtime unavailable: {exc}")
    return torch


def _model_and_optimizer(device, legacy=False):
    torch = _torch()
    from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
    from opentad.models.detectors.actionformer import ActionFormer
    from opentad.cores.optimizer import build_optimizer, prepare_optimizer_parameter_freezing

    vit = VisionTransformerAdapter(
        img_size=8, patch_size=4, embed_dims=16, depth=1, num_heads=4,
        num_frames=4, total_frames=4, adapter_index=[0], return_feat_map=True, with_cp=True,
        bounded_interval_adapter=dict(enabled=True),
        continuous_timestamp_conditioner=dict(enabled=True),
    )
    class TinyDetector(ActionFormer):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, frames, timestamps):
            output = self.backbone.model.backbone(frames, tubelet_timestamps=timestamps)
            return output.square().mean() + self.scout(output.mean((2, 3, 4))).square().mean()

    model = TinyDetector()
    model.backbone = torch.nn.Module()
    model.backbone.freeze_backbone = False
    model.backbone.model = torch.nn.Module()
    model.backbone.model.backbone = vit
    model.scout = torch.nn.Linear(16, 1)
    model.to(device)
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/duca_evidence_recovery_base.py"))
    optimizer_cfg = copy.deepcopy(cfg.optimizer)
    if legacy:
        optimizer_cfg.backbone.custom = [optimizer_cfg.backbone.custom[0]]
    logger = logging.getLogger(__name__)
    prepare_optimizer_parameter_freezing(optimizer_cfg, model, logger)
    optimizer = build_optimizer(optimizer_cfg, SimpleNamespace(module=model), logger)
    return model, vit, optimizer


def _forward(model, vit, device):
    torch = _torch()
    return model(
        torch.randn(1, 3, 4, 8, 8, device=device, requires_grad=True),
        torch.tensor([[0.0, 3.0]], device=device),
    )


def test_time_parameter_coverage_survives_real_backbone_forward():
    _torch()
    from opentad.cores.optimizer import assert_optimizer_exact_coverage

    model, vit, optimizer = _model_and_optimizer("cpu")
    _forward(model, vit, "cpu").backward()
    assert_optimizer_exact_coverage(model, optimizer)
    scale = vit.blocks[0].relative_physical_time_scale
    assert scale.grad is not None
    assert any(param.grad is not None for param in vit.continuous_timestamp_conditioner.parameters())


def test_legacy_groups_reproduce_missing_time_parameters_after_forward():
    _torch()
    from opentad.cores.optimizer import assert_optimizer_exact_coverage

    model, vit, optimizer = _model_and_optimizer("cpu", legacy=True)
    _forward(model, vit, "cpu")
    with pytest.raises(AssertionError, match="missing=.*relative_physical_time_scale"):
        assert_optimizer_exact_coverage(model, optimizer)


def test_cuda_time_overflow_skips_update_without_poisoning_scout():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA AMP")
    model, vit, optimizer = _model_and_optimizer("cuda")
    scaler = torch.cuda.amp.GradScaler()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    with torch.cuda.amp.autocast():
        loss = _forward(model, vit, "cuda")
    scaler.scale(loss).backward()
    vit.blocks[0].relative_physical_time_scale.grad.fill_(float("inf"))
    scale_before = scaler.get_scale()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() < scale_before
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = _forward(model, vit, "cuda")
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())


def test_ddp_checkpoint_supports_changing_time_parameter_usage(tmp_path):
    torch = _torch()
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    model, vit, optimizer = _model_and_optimizer("cpu")
    dist.init_process_group(
        "gloo", init_method=f"file://{(tmp_path / 'ddp_store').as_posix()}",
        rank=0, world_size=1,
    )
    try:
        wrapped = DistributedDataParallel(model, static_graph=False, find_unused_parameters=True)
        for enabled in (False, True, False):
            optimizer.zero_grad()
            timestamps = torch.tensor([[0.0, 0.8]]) if enabled else None
            loss = wrapped(torch.randn(1, 3, 4, 8, 8, requires_grad=True), timestamps)
            loss.backward()
            assert (vit.blocks[0].relative_physical_time_scale.grad is not None) == enabled
            assert torch.isfinite(loss)
            optimizer.step()
    finally:
        dist.destroy_process_group()
