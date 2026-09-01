from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CORES_DIR = ROOT / "opentad" / "cores"


def _load_optimizer_module():
    package_name = "_duca_optimizer_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(CORES_DIR)]
    sys.modules[package_name] = package

    module_name = f"{package_name}.optimizer"
    spec = importlib.util.spec_from_file_location(module_name, CORES_DIR / "optimizer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


optimizer_module = _load_optimizer_module()


class _TwoParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.tensor(1.0))
        self.second = nn.Parameter(torch.tensor(2.0))


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keep = nn.Parameter(torch.tensor(1.0))
        self.excluded = nn.Parameter(torch.tensor(2.0))
        self.excluded_custom = nn.Parameter(torch.tensor(3.0))


class _Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()


class _WrappedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = _Detector()


def test_exact_coverage_accepts_each_trainable_parameter_once() -> None:
    model = _TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert optimizer_module.assert_optimizer_exact_coverage(model, optimizer) is None


def test_exact_coverage_rejects_missing_trainable_parameter() -> None:
    model = _TwoParameterModel()
    optimizer = torch.optim.SGD([model.first], lr=0.1)

    with pytest.raises(AssertionError, match=r"missing.*second"):
        optimizer_module.assert_optimizer_exact_coverage(model, optimizer)


def test_exact_coverage_rejects_parameter_duplicated_across_groups() -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.param_groups.append({"params": [model.weight]})

    with pytest.raises(AssertionError, match=r"duplicate.*weight"):
        optimizer_module.assert_optimizer_exact_coverage(model, optimizer)


def test_exact_coverage_rejects_stale_parameter_not_owned_by_model() -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.param_groups[0]["params"].append(nn.Parameter(torch.tensor(3.0)))

    with pytest.raises(AssertionError, match="stale"):
        optimizer_module.assert_optimizer_exact_coverage(model, optimizer)


def test_exact_coverage_rejects_frozen_model_parameter() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(AssertionError, match=r"frozen.*weight"):
        optimizer_module.assert_optimizer_exact_coverage(model, optimizer)


def test_exclude_requires_pre_ddp_freeze_and_preserves_custom_override() -> None:
    model = _WrappedModel()
    cfg = {
        "lr": 1.0e-4,
        "weight_decay": 0.05,
        "custom": [{"name": "custom", "lr": 2.0e-4, "weight_decay": 0.01}],
        "exclude": ["excluded"],
    }

    logger = logging.getLogger(__name__)
    with pytest.raises(RuntimeError, match="must be frozen before DDP"):
        optimizer_module.get_backbone_optim_groups(cfg, model, logger)

    optimizer_module.prepare_optimizer_parameter_freezing(
        {"type": "AdamW", "backbone": cfg}, model, logger
    )
    groups = optimizer_module.get_backbone_optim_groups(cfg, model, logger)
    grouped_param_ids = {id(param) for group in groups for param in group["params"]}

    assert model.module.backbone.excluded.requires_grad is False
    assert id(model.module.backbone.excluded) not in grouped_param_ids
    assert model.module.backbone.excluded_custom.requires_grad is True
    assert id(model.module.backbone.excluded_custom) in grouped_param_ids

    optimizer = torch.optim.SGD(groups, lr=0.1)
    assert optimizer_module.assert_optimizer_exact_coverage(model, optimizer) is None
