from __future__ import annotations

import copy
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/adatad/thumos"


def _load_stage2(monkeypatch, filename: str) -> Config:
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "stage1_epoch19.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "19")
    return Config.fromfile(str(CONFIG_DIR / filename))


def _without_course(cfg: Config) -> dict:
    model = copy.deepcopy(cfg.model.to_dict())
    model["frame_selector"].pop("loss_weight_schedule", None)
    return model


def test_stage1_is_only_a_twenty_epoch_override_of_historical_h65() -> None:
    historical = Config.fromfile(
        str(CONFIG_DIR / "duca_sampling_rate_curriculum_stage1_uniform384.py")
    )
    compressed = Config.fromfile(
        str(CONFIG_DIR / "duca_h65_60_stage1_uniform20.py")
    )
    assert compressed.model.to_dict() == historical.model.to_dict()
    assert compressed.workflow.end_epoch == 20
    assert compressed.workflow.expected_successful_optimizer_updates == 2000
    assert compressed.workflow.primary_checkpoint_epoch == 19
    assert compressed.scheduler.max_epoch == 20
    assert compressed.workflow.checkpoint_interval == 5
    assert compressed.workflow.intermediate_validation_selects_checkpoint is False


def test_stage2_changes_only_the_course_and_duration(monkeypatch) -> None:
    historical = _load_stage2(
        monkeypatch, "duca_sampling_rate_curriculum_stage2_joint384.py"
    )
    compressed = _load_stage2(
        monkeypatch, "duca_h65_60_stage2_transition20_joint20.py"
    )
    assert _without_course(compressed) == _without_course(historical)
    assert compressed.optimizer.to_dict() == historical.optimizer.to_dict()
    assert compressed.workflow.end_epoch == 40
    assert compressed.workflow.expected_successful_optimizer_updates == 4000
    assert compressed.workflow.primary_checkpoint_epoch == 39
    assert compressed.scheduler.max_epoch == 40
    assert compressed.workflow.intermediate_validation_selects_checkpoint is False


def test_stage2_has_one_twenty_epoch_transition_then_twenty_epoch_joint(monkeypatch) -> None:
    cfg = _load_stage2(
        monkeypatch, "duca_h65_60_stage2_transition20_joint20.py"
    )
    schedule = cfg.model.frame_selector.loss_weight_schedule
    for name in (
        "actionness",
        "transition",
        "transition_boundary",
        "policy_alpha",
        "asformer_adapt",
    ):
        assert schedule[name].warmup_steps == 0
        assert schedule[name].transition_steps == 2000
    for name in ("detector_gradient", "detector_contribution"):
        assert schedule[name].warmup_steps == 667
        assert schedule[name].transition_steps == 1333
    assert schedule.policy_alpha.start == 0.0 and schedule.policy_alpha.end == 1.0
    assert schedule.detector_gradient.end == 0.25
    assert schedule.detector_contribution.end == 1.0
    assert schedule.asformer_adapt.end == 1.0


def test_stage2_handoff_is_epoch19_ema_and_resets_only_course_clock(monkeypatch) -> None:
    cfg = _load_stage2(
        monkeypatch, "duca_h65_60_stage2_transition20_joint20.py"
    )
    init = cfg.workflow.model_initialization
    assert init.state_key == "state_dict_ema"
    assert init.expected_checkpoint_epoch == 19
    assert init.reset_state_keys == ["frame_selector._loss_weight_schedule_step"]
    assert cfg.duca_sampling_rate_contract.optimizer_scheduler_amp_state_reset is True


def test_launcher_has_no_model_route_or_second_optimizer_restart() -> None:
    text = (ROOT / "scripts/run_duca_h65_60_curriculum_n16r4.sbatch").read_text()
    assert "STAGE1_20|STAGE2_40" in text
    assert "STAGE3" not in text
    assert "duca_h65_60_stage1_uniform20.py" in text
    assert "duca_h65_60_stage2_transition20_joint20.py" in text
    assert "DUCA_STAGE1_CHECKPOINT_EPOCH=19" in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert 'export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1' in text
    assert 'source /etc/profile' in text
    assert text.count('exec "$PYTHON" tools/train.py') == 1


def test_implementation_does_not_touch_model_sources() -> None:
    stage1 = (CONFIG_DIR / "duca_h65_60_stage1_uniform20.py").read_text()
    stage2 = (
        CONFIG_DIR / "duca_h65_60_stage2_transition20_joint20.py"
    ).read_text()
    assert "opentad/models" not in stage1 + stage2
    assert "single_clock" not in stage1 + stage2
    assert "physical_time" not in stage1 + stage2
    assert "dynamic_outer_k" not in stage1 + stage2

