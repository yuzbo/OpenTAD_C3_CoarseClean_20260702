from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

from tools.bata.duca_protected_physical_training import (
    new_update_audit,
    selector_schedule_step,
    validate_update_state,
)
from tools.bata.finalize_duca_protected_physical_run import _inspect_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"
HOMOTOPY_ARM = "protected_e2e_homotopy025"


class _FakeOfficialASFormerSource(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.trunk = nn.Parameter(torch.linspace(0.5, 1.5, 96))
        self.action_head = nn.Parameter(torch.tensor(0.7))

    def forward(self, inputs, valid_mask=None):
        base = inputs.float().mean(dim=(1, 3, 4))
        hidden = base[:, :, None] * self.trunk[None, None, :]
        logits = hidden[:, :, 0] * self.action_head
        valid = valid_mask.bool()
        return {
            "actionness_logits": logits.masked_fill(~valid, -1.0e4),
            "p_action": torch.sigmoid(logits).masked_fill(~valid, 0.0),
            "coarse_hidden_features": hidden.masked_fill(~valid[:, :, None], 0.0),
            "hidden_kind": "official_asformer_encoder_hidden",
            "provenance": {
                "source_type": "test_official_asformer",
                "thumos_trained": True,
                "uses_labels": True,
                "uses_teacher": False,
                "uses_gt": False,
                "uses_prediction_cache": False,
            },
            "compute_profile": {"test": True},
        }


class _Registry:
    def register_module(self):
        return lambda module: module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_duca_modules():
    package_root = "_duca_homotopy_under_test"
    for package_name in (
        package_root,
        f"{package_root}.models",
        f"{package_root}.models.duca",
        f"{package_root}.models.selectors",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

    builder = types.ModuleType(f"{package_root}.models.builder")
    builder.SELECTORS = _Registry()
    sys.modules[builder.__name__] = builder

    acquisition = types.ModuleType(f"{package_root}.models.duca.acquisition")
    acquisition.C3CoarseProbeActionnessSource = _FakeOfficialASFormerSource
    acquisition._assert_no_forbidden_payload = lambda *_args, **_kwargs: None
    sys.modules[acquisition.__name__] = acquisition

    duca_root = ROOT / "opentad" / "models" / "duca"
    structured = _load_module(
        f"{package_root}.models.duca.structured_selection",
        duca_root / "structured_selection.py",
    )
    _load_module(
        f"{package_root}.models.duca.transition_only",
        duca_root / "transition_only.py",
    )
    selector = _load_module(
        f"{package_root}.models.selectors.duca_protected_e2e_frame_selector",
        ROOT
        / "opentad"
        / "models"
        / "selectors"
        / "duca_protected_e2e_frame_selector.py",
    )
    return structured, selector


_STRUCTURED, _SELECTOR = _load_duca_modules()
exact_uniform_reference_scores = _STRUCTURED.exact_uniform_reference_scores
physical_exact_k_forward_backward = _STRUCTURED.physical_exact_k_forward_backward
physical_exact_k_homotopy_log_potential = (
    _STRUCTURED.physical_exact_k_homotopy_log_potential
)
physical_exact_k_select = _STRUCTURED.physical_exact_k_select
physical_exact_k_viterbi = _STRUCTURED.physical_exact_k_viterbi
physical_exact_uniform_gap_cap = _STRUCTURED.physical_exact_uniform_gap_cap
DucaProtectedE2EFrameSelector = _SELECTOR.DucaProtectedE2EFrameSelector


def _selector(*, total_steps: int = 100) -> DucaProtectedE2EFrameSelector:
    return DucaProtectedE2EFrameSelector(
        in_channels=3,
        arm=HOMOTOPY_ARM,
        budget=3,
        dense_window_size=6,
        homotopy_total_steps=total_steps,
        actionness_source_cfg={
            "probe_model": "official-action-seg",
            "official_action_seg_backend": "official_asformer",
            "frozen": False,
            "trainable": True,
        },
    )


def _batch():
    generator = torch.Generator().manual_seed(20260721)
    inputs = torch.randn((1, 3, 6, 2, 2), generator=generator)
    masks = torch.ones((1, 6), dtype=torch.bool)
    metas = [
        {
            "video_name": "homotopy_test",
            "avg_fps": 10.0,
            "frame_inds": torch.arange(6, dtype=torch.long)[:, None] * 4,
        }
    ]
    segments = [torch.tensor([[1.0, 5.0]], dtype=torch.float32)]
    labels = [torch.tensor([0], dtype=torch.long)]
    boundary_validity = [torch.tensor([[True, True]])]
    return inputs, masks, metas, segments, labels, boundary_validity


def _forward_train(selector: DucaProtectedE2EFrameSelector):
    inputs, masks, metas, segments, labels, boundary_validity = _batch()
    return selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )


def _zero_grad(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _grad_mass(parameters) -> float:
    return sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())
        for parameter in parameters
    )


def test_homotopy_potential_endpoints_and_shared_physical_dag() -> None:
    learned = torch.tensor(
        [[0.0, 3.0, 0.0, 2.0, 0.0, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    valid = torch.ones_like(learned, dtype=torch.bool)
    seconds = torch.tensor(
        [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0]],
        dtype=torch.float64,
    )
    cap = physical_exact_uniform_gap_cap(seconds, valid, k=3)

    alpha_zero = physical_exact_k_homotopy_log_potential(
        learned,
        valid,
        k=3,
        alpha=0.0,
    )
    reference = exact_uniform_reference_scores(learned, valid, k=3)
    assert torch.equal(alpha_zero, reference)
    uniform = physical_exact_k_select(
        alpha_zero,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    assert uniform.hard_positions.tolist() == [[0, 2, 5]]

    alpha_one = physical_exact_k_homotopy_log_potential(
        learned,
        valid,
        k=3,
        alpha=1.0,
    )
    assert torch.equal(alpha_one, learned)
    direct = physical_exact_k_select(
        learned,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    learned_endpoint = physical_exact_k_select(
        alpha_one,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    assert torch.equal(learned_endpoint.hard_positions, direct.hard_positions)
    assert torch.equal(learned_endpoint.soft_slot_assignment, direct.soft_slot_assignment)

    mixed = physical_exact_k_homotopy_log_potential(
        learned,
        valid,
        k=3,
        alpha=0.4,
    )
    combined = physical_exact_k_select(
        mixed,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    hard = physical_exact_k_viterbi(
        mixed,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    soft = physical_exact_k_forward_backward(
        mixed,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    assert torch.equal(combined.hard_positions, hard.hard_positions)
    assert torch.equal(combined.soft_slot_assignment, soft.soft_slot_assignment)
    assert torch.equal(combined.edge_count, hard.edge_count)
    assert torch.equal(combined.edge_count, soft.edge_count)


def test_homotopy_potential_handles_short_padded_rows() -> None:
    learned = torch.tensor(
        [
            [0.0, 3.0, 0.0, 2.0, 0.0, 4.0],
            [1.0, 0.0, 2.0, 3.0, 99.0, 99.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ]
    )
    seconds = torch.tensor(
        [
            [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            [0.0, 0.4, 0.8, 1.2, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    cap = physical_exact_uniform_gap_cap(seconds, valid, k=3)
    alpha_zero = physical_exact_k_homotopy_log_potential(
        learned,
        valid,
        k=3,
        alpha=0.0,
    )
    selected = physical_exact_k_select(
        alpha_zero,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    assert selected.hard_positions.tolist() == [[0, 2, 5], [0, 2, 3]]
    assert torch.isneginf(alpha_zero[1, 4:]).all()

    alpha_one = physical_exact_k_homotopy_log_potential(
        learned,
        valid,
        k=3,
        alpha=1.0,
    )
    assert torch.equal(alpha_one[valid], learned[valid])
    assert torch.isneginf(alpha_one[~valid]).all()


def test_homotopy_alpha_schedule_endpoints_and_inference_override() -> None:
    selector = _selector()
    assert "schedule_step" in selector.state_dict()
    warmup_state = selector._policy_homotopy_state(training=True)
    assert warmup_state["alpha"] == 0.0
    assert warmup_state["alpha_zero_contract"] == "hard_forward_exact_uniform"

    selector.schedule_step.fill_(4)
    assert selector._policy_homotopy_state(training=True)["alpha"] == 0.0
    selector.schedule_step.fill_(19)
    midpoint = selector._policy_homotopy_state(training=True)
    assert midpoint["alpha"] == pytest.approx(0.5)
    assert midpoint["phase"] == "physical_exact_k_policy_homotopy"
    selector.schedule_step.fill_(34)
    transition_endpoint = selector._policy_homotopy_state(training=True)
    assert transition_endpoint["alpha"] == 1.0
    assert transition_endpoint["phase"] == "physical_exact_k_policy_homotopy"
    selector.schedule_step.fill_(35)
    learned = selector._policy_homotopy_state(training=True)
    assert learned["alpha"] == 1.0
    assert learned["phase"] == "learned_policy"

    selector.schedule_step.zero_()
    output = _forward_train(selector)
    assert output["selector_outputs"]["selected_positions"].tolist() == [[0, 2, 5]]
    assert output["selector_outputs"]["policy_homotopy"]["alpha"] == 0.0

    selector.eval()
    inputs, masks, metas, *_ = _batch()
    inference = selector.forward_test(inputs, masks, metas)
    state = inference["selector_outputs"]
    assert state["policy_homotopy"]["alpha"] == 1.0
    assert state["policy_homotopy"]["phase"] == "inference_learned_policy"
    assert torch.equal(state["policy_log_potential"], state["policy_log_probabilities"])


def test_homotopy_gradient_ownership() -> None:
    selector = _selector()
    selector.train()
    warmup = _forward_train(selector)
    source = selector.raw_actionness_source

    warmup["inputs"].square().mean().backward(retain_graph=True)
    assert _grad_mass(selector.transition_scorer.parameters()) == 0.0
    assert _grad_mass([source.trunk]) == 0.0
    assert _grad_mass([source.action_head]) == 0.0

    _zero_grad(selector)
    sum(warmup["losses"].values()).backward()
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0
    assert _grad_mass([source.trunk]) > 0.0
    assert _grad_mass([source.action_head]) > 0.0

    _zero_grad(selector)
    selector.schedule_step.fill_(19)
    transition = _forward_train(selector)
    assert transition["selector_outputs"]["policy_homotopy"]["alpha"] == pytest.approx(0.5)
    assert transition["selector_outputs"]["detector_bridge_gradient_scale"] == 0.25
    transition["inputs"].square().mean().backward()
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0
    assert _grad_mass([source.trunk]) == 0.0
    assert _grad_mass([source.action_head]) == 0.0


def test_homotopy_schedule_step_checkpoint_resume() -> None:
    selector = _selector()
    for _ in range(17):
        _forward_train(selector)
        summary = selector.after_optimizer_step()
        assert summary["updated"] is True
    checkpoint = copy.deepcopy(selector.state_dict())
    assert int(checkpoint["schedule_step"].item()) == 17

    resumed = _selector()
    resumed.load_state_dict(checkpoint)
    assert int(resumed.schedule_step.item()) == 17
    assert resumed._policy_homotopy_state(training=True) == selector._policy_homotopy_state(
        training=True
    )
    assert resumed.after_optimizer_step()["updated"] is False
    _forward_train(resumed)
    _forward_train(selector)
    resumed.after_optimizer_step()
    selector.after_optimizer_step()
    assert int(resumed.schedule_step.item()) == int(selector.schedule_step.item()) == 18


def test_formal_schedule_accounting_matches_only_successful_updates() -> None:
    selector = _selector(total_steps=100)
    selector.train()
    for _ in range(3):
        _forward_train(selector)
        assert selector.after_optimizer_step()["updated"] is True
    model = SimpleNamespace(frame_selector=selector)
    counters = new_update_audit()
    for key in (
        "attempted_batches",
        "optimizer_attempts",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        counters[key] = 3
    contract = {
        "expected_train_batches_per_epoch": 1,
        "expected_successful_optimizer_updates": 100,
        "expected_selector_schedule_updates": 100,
        "selector_schedule_enabled": True,
        "max_amp_retries_per_batch": 8,
    }
    assert selector_schedule_step(model) == 3
    validate_update_state(
        contract=contract,
        epoch=2,
        train_batches_per_epoch=1,
        update_audit=counters,
        scheduler_last_epoch=3,
        selector_step=selector_schedule_step(model),
        uses_ema=True,
    )


def test_homotopy_rejects_a_total_too_short_for_the_frozen_phases() -> None:
    with pytest.raises(ValueError, match="warmup must contain at least one step"):
        _selector(total_steps=1)


def test_amp_replay_restore_does_not_advance_the_schedule() -> None:
    selector = _selector(total_steps=100)
    selector.train()
    snapshot = selector.capture_amp_replay_state()
    _forward_train(selector)
    assert selector._pending_homotopy_schedule_advance is True

    selector.restore_amp_replay_state(snapshot)
    assert selector.after_optimizer_step()["updated"] is False
    assert int(selector.schedule_step.item()) == 0

    _forward_train(selector)
    assert selector.after_optimizer_step()["updated"] is True
    assert int(selector.schedule_step.item()) == 1


def test_terminal_checkpoint_reopens_model_and_ema_schedule(tmp_path: Path) -> None:
    metadata = {"frozen": True}
    checkpoint = {
        "epoch": 59,
        "state_dict": {
            "module.frame_selector.schedule_step": torch.tensor(6000),
        },
        "state_dict_ema": {
            "module.frame_selector.schedule_step": torch.tensor(6000),
        },
        "optimizer": {},
        "scheduler": {"last_epoch": 6000},
        "grad_scaler": {},
        "rng_state": {
            "python": None,
            "numpy": None,
            "torch_cpu": None,
            "torch_cuda": None,
        },
        "experiment_metadata": metadata,
    }
    path = tmp_path / "epoch_59.pth"
    torch.save(checkpoint, path)
    evidence = _inspect_checkpoint(
        path,
        expected_metadata=metadata,
        expected_updates=6000,
        expected_selector_step=6000,
    )
    assert evidence["selector_schedule_step"] == 6000
    assert evidence["ema_selector_schedule_step"] == 6000


def test_terminal_checkpoint_rejects_ema_schedule_drift(tmp_path: Path) -> None:
    metadata = {"frozen": True}
    checkpoint = {
        "epoch": 59,
        "state_dict": {
            "module.frame_selector.schedule_step": torch.tensor(6000),
        },
        "state_dict_ema": {
            "module.frame_selector.schedule_step": torch.tensor(5999),
        },
        "optimizer": {},
        "scheduler": {"last_epoch": 6000},
        "grad_scaler": {},
        "rng_state": {
            "python": None,
            "numpy": None,
            "torch_cpu": None,
            "torch_cuda": None,
        },
        "experiment_metadata": metadata,
    }
    path = tmp_path / "epoch_59_tampered.pth"
    torch.save(checkpoint, path)
    with pytest.raises(
        ValueError,
        match="state_dict_ema selector schedule step mismatch",
    ):
        _inspect_checkpoint(
            path,
            expected_metadata=metadata,
            expected_updates=6000,
            expected_selector_step=6000,
        )


def test_homotopy_official60_config_is_a_head_preserving_successor() -> None:
    config = Config.fromfile(
        str(
            CONFIG_ROOT
            / "duca_protected_physical_e2e_homotopy025_fixed384_official60.py"
        )
    )
    reference = Config.fromfile(
        str(CONFIG_ROOT / "duca_protected_physical_e2e_fixed384_official60.py")
    )
    selector = config.model.frame_selector
    contract = config.duca_variant_contract

    assert selector.arm == HOMOTOPY_ARM
    assert selector.homotopy_total_steps == 6000
    assert selector.detector_bridge_gradient_scale == 0.25
    assert selector.uniform_companion_fraction == 0.0
    assert contract.policy_alpha_warmup_fraction == 0.05
    assert contract.policy_alpha_transition_fraction == 0.30
    assert contract.policy_alpha_zero_contract == "hard_forward_exact_uniform"
    assert contract.inference_policy_alpha == 1.0
    assert config.model.rpn_head.to_dict() == reference.model.rpn_head.to_dict()
    assert config.model.backbone.to_dict() == reference.model.backbone.to_dict()
    assert config.dataset.to_dict() == reference.dataset.to_dict()
    assert config.optimizer.to_dict() == reference.optimizer.to_dict()
    assert config.scheduler.to_dict() == reference.scheduler.to_dict()
