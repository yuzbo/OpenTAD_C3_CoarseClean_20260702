from __future__ import annotations

from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_sampling_rate_both_asformer_full_mini_visual.py"
RUNNER = ROOT / "scripts" / "run_duca_sampling_rate_mini_visual_gpu1.sh"


def test_mini_visual_config_performs_real_bounded_training() -> None:
    cfg = Config.fromfile(str(CONFIG))
    contract = cfg.duca_mini_visual_contract
    assert contract.purpose == "trained_small_sample_mechanism_diagnostic_not_official_map"
    assert int(contract.train_epochs) == 10
    assert int(contract.train_updates_per_epoch) == 4
    assert int(contract.expected_optimizer_updates) == 40
    assert cfg.workflow.formal_protocol == ""
    assert int(cfg.workflow.end_epoch) == 10
    assert int(cfg.workflow.max_train_iters) == 4
    assert int(cfg.workflow.checkpoint_interval) == 1
    assert int(cfg.solver.train.batch_size) == 1
    schedule = cfg.model.frame_selector.loss_weight_schedule
    assert float(schedule.policy_alpha.end) == 1.0
    assert float(schedule.detector_contribution.end) == 1.0
    assert float(schedule.asformer_adapt.end) == 1.0


def test_mini_visual_runner_exports_trained_and_inference_evidence() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "tools/train.py" in source
    assert "max_train_iters=4" not in source
    assert "export_duca_training_attribution" in source
    assert "export_duca_selection_quality" in source
    assert "plot_duca_training_attribution" in source
    assert "--limit-batches 2" in source
    assert "epoch in 0 4 9" in source
    assert "official_map_reported\": False" in source
