import copy
import math
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from opentad.cores.scheduler import RelativeSuccessfulUpdateLR, build_scheduler
from opentad.utils.checkpoint import save_checkpoint
from tools.train import _restore_resumable_update_audit


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
STAGE1_CHECKPOINT = "/frozen/stage1/epoch_29.pth"
STAGE1_SHA256 = "b" * 64


def _scheduler(mode, optimizer):
    return RelativeSuccessfulUpdateLR(
        optimizer,
        mode=mode,
        total_updates=3000,
        warmup_updates=500,
        plateau_updates=1000,
        decay_updates=1000,
        hold_updates=500,
        terminal_factor=0.25,
        horizon_updates=6000,
    )


def _trajectory(mode, interrupt_at=None):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    second_parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter], "lr": 1.0e-5},
            {"params": [second_parameter], "lr": 5.0e-5},
        ]
    )
    scheduler = _scheduler(mode, optimizer)
    clocks = dict(
        successful_optimizer_updates=0,
        scheduler_updates=0,
        ema_updates=0,
        duca_schedule_updates=0,
    )
    trajectory = []

    for update in range(1, 3001):
        trajectory.append(tuple(group["lr"] for group in optimizer.param_groups))
        optimizer.step()
        scheduler.step()
        for name in clocks:
            clocks[name] += 1

        if interrupt_at == update:
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            scheduler_state = copy.deepcopy(scheduler.state_dict())
            saved_clocks = copy.deepcopy(clocks)

            resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
            resumed_second_parameter = torch.nn.Parameter(
                second_parameter.detach().clone()
            )
            optimizer = torch.optim.SGD(
                [
                    {"params": [resumed_parameter], "lr": 1.0e-5},
                    {"params": [resumed_second_parameter], "lr": 5.0e-5},
                ]
            )
            scheduler = _scheduler(mode, optimizer)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            clocks = saved_clocks

    return trajectory, clocks


def _load_configs(monkeypatch):
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", STAGE1_CHECKPOINT)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", STAGE1_SHA256)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    am = Config.fromfile(str(CONFIG_DIR / "duca_h65_60_stage2_am_rpch25.py"))
    long = Config.fromfile(
        str(CONFIG_DIR / "duca_h65_60_stage2_longcosine_h6000.py")
    )
    return am, long


def test_am_rpch25_exact_frozen_points_and_exposure():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = _scheduler("am_rpch25", optimizer)

    expected = {
        1: 0.0,
        500: 1.0,
        501: 1.0,
        1500: 1.0,
        2000: 0.625,
        2500: 0.25,
        2501: 0.25,
        3000: 0.25,
    }
    for update, factor in expected.items():
        assert scheduler.factor_for_update(update) == pytest.approx(factor, abs=1e-12)

    exposure = sum(scheduler.factor_for_update(update) for update in range(1, 3001))
    assert exposure == pytest.approx(1999.625, abs=1e-9)


def test_longcosine_h6000_exact_terminal_and_relative_group_ratios():
    trajectory, clocks = _trajectory("longcosine_h6000")
    expected_terminal = 0.5 * (
        1.0 + math.cos(math.pi * (3000 - 500) / (6000 - 500))
    )
    assert trajectory[0] == (0.0, 0.0)
    assert trajectory[499] == pytest.approx((1.0e-5, 5.0e-5), abs=1e-15)
    assert trajectory[-1] == pytest.approx(
        (1.0e-5 * expected_terminal, 5.0e-5 * expected_terminal),
        abs=1e-15,
    )
    assert trajectory[-1][1] / trajectory[-1][0] == pytest.approx(5.0)
    assert expected_terminal == pytest.approx(0.5711574191366426, abs=1e-12)
    assert set(clocks.values()) == {3000}


@pytest.mark.parametrize("mode", ["am_rpch25", "longcosine_h6000"])
@pytest.mark.parametrize("interrupt_at", [500, 1000, 1500, 2000, 2500])
def test_resume_is_elementwise_identical(mode, interrupt_at):
    uninterrupted, uninterrupted_clocks = _trajectory(mode)
    resumed, resumed_clocks = _trajectory(mode, interrupt_at=interrupt_at)
    assert resumed == uninterrupted
    assert resumed_clocks == uninterrupted_clocks


def test_builder_preserves_successful_update_units_and_max_epoch():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-4)
    config = dict(
        type="RelativeSuccessfulUpdateLR",
        mode="am_rpch25",
        max_epoch=30,
        total_updates=3000,
        warmup_updates=500,
        plateau_updates=1000,
        decay_updates=1000,
        hold_updates=500,
        terminal_factor=0.25,
        horizon_updates=6000,
    )
    scheduler, max_epoch = build_scheduler(copy.deepcopy(config), optimizer, 137)
    assert max_epoch == 30
    assert scheduler.total_updates == 3000
    assert scheduler.factor_for_update(500) == 1.0


def test_resolved_configs_match_except_frozen_attribution_fields(monkeypatch):
    am, long = _load_configs(monkeypatch)

    for cfg in (am, long):
        assert cfg.seed == 3407
        assert cfg.total_epochs == 30
        assert cfg.max_updates == 3000
        assert cfg.workflow.end_epoch == 30
        assert cfg.workflow.expected_train_batches_per_epoch == 100
        assert cfg.workflow.expected_successful_optimizer_updates == 3000
        assert cfg.workflow.formal_successful_update_contract is False
        assert cfg.workflow.require_resumable_training_state is True
        assert cfg.workflow.primary_checkpoint_epoch == 29
        assert cfg.workflow.primary_checkpoint_state_key == "state_dict_ema"
        assert cfg.workflow.checkpoint_criterion == "terminal_epoch_29_state_dict_ema"
        assert cfg.workflow.model_initialization.expected_checkpoint_epoch == 29
        assert cfg.scheduler.type == "RelativeSuccessfulUpdateLR"
        assert set(cfg.scheduler) == {
            "type",
            "mode",
            "max_epoch",
            "total_updates",
            "warmup_updates",
            "plateau_updates",
            "decay_updates",
            "hold_updates",
            "terminal_factor",
            "horizon_updates",
        }
        assert cfg.scheduler.max_epoch == 30
        assert cfg.scheduler.total_updates == 3000
        schedule = cfg.model.frame_selector.loss_weight_schedule
        assert schedule.transition_steps == 2000
        assert schedule.detector_gradient.warmup_steps == 1000
        assert schedule.detector_gradient.transition_steps == 1000
        assert schedule.detector_contribution.warmup_steps == 1000
        assert schedule.detector_contribution.transition_steps == 1000

    am_dict = am.to_dict()
    long_dict = long.to_dict()
    for config in (am_dict, long_dict):
        config["scheduler"].pop("mode")
        config["duca_sampling_rate_contract"].pop("route")
        config["duca_sampling_rate_contract"].pop("stage")
        config["workflow"].pop("training_profile")
        config.pop("work_dir")
        config.pop("filename", None)
    assert am_dict == long_dict


def test_launcher_has_distinct_fresh_and_resume_pre_run_modes():
    launcher = (
        ROOT / "scripts" / "run_duca_h65_60_lr_schedule_n16r4.sbatch"
    ).read_text(encoding="utf-8")
    assert "PRE_RUN_ONLY and PRE_RUN_RESUME_ONLY are mutually exclusive" in launcher
    assert '"workflow.end_epoch=1"' in launcher
    assert '"workflow.end_epoch=2"' in launcher
    assert 'DUCA_RESUME_CHECKPOINT:-' in launcher
    assert '"dataset.train.data_path=$VIDEO_ROOT"' in launcher
    assert '"dataset.val.data_path=$VIDEO_ROOT"' in launcher
    assert '"dataset.test.data_path=$VIDEO_ROOT"' in launcher
    assert '"dataset.train.ann_file=$ANNOTATION_PATH"' in launcher
    assert '"dataset.test.class_map=$CATEGORY_PATH"' in launcher
    assert '"evaluation.ground_truth_filename=$ANNOTATION_PATH"' in launcher
    assert '"data.train.ann_file=' not in launcher


def test_checkpoint_roundtrip_preserves_cumulative_update_audit(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    update_audit = {
        "attempted_batches": 500,
        "optimizer_attempts": 503,
        "successful_optimizer_updates": 500,
        "amp_skipped_attempts": 3,
        "replayed_batches": 3,
        "replay_exhaustions": 0,
        "scheduler_updates": 500,
        "ema_updates": 500,
        "duca_schedule_updates": 500,
        "forced_amp_overflow_attempts": 0,
        "max_amp_retries_observed": 1,
        "nonfinite_loss_attempts": 0,
        "nonfinite_loss_replays": 0,
        "nonfinite_loss_replay_exhaustions": 0,
        "max_nonfinite_loss_retries_observed": 0,
        "replay_state_restorations": 3,
    }
    checkpoint_path = save_checkpoint(
        model,
        None,
        optimizer,
        scheduler,
        4,
        work_dir=str(tmp_path),
        update_audit_state=update_audit,
        successful_optimizer_updates=500,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    template = {
        key: 0
        for key in (
            "attempted_batches",
            "optimizer_attempts",
            "successful_optimizer_updates",
            "amp_skipped_attempts",
            "replayed_batches",
            "replay_exhaustions",
            "scheduler_updates",
            "ema_updates",
            "duca_schedule_updates",
            "forced_amp_overflow_attempts",
            "max_amp_retries_observed",
        )
    }
    restored = _restore_resumable_update_audit(checkpoint, template)
    assert restored == update_audit
    restored["attempted_batches"] += 1
    assert checkpoint["update_audit_state"]["attempted_batches"] == 500


def test_resume_rejects_update_audit_count_disagreement():
    checkpoint = {
        "successful_optimizer_updates": 500,
        "update_audit_state": {
            "attempted_batches": 500,
            "successful_optimizer_updates": 499,
        },
    }
    with pytest.raises(RuntimeError, match="disagrees"):
        _restore_resumable_update_audit(
            checkpoint,
            {"attempted_batches": 0, "successful_optimizer_updates": 0},
        )
