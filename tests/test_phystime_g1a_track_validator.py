import copy

import pytest
from mmengine.config import Config

from tools.bata.validate_phystime_g1a_track import DEFAULT_CONFIGS, validate_track


def test_g1a_track_validator_closes_coordinate_only_contract(tmp_path):
    output = tmp_path / "g1a_contract.json"
    report = validate_track(DEFAULT_CONFIGS, output=output)

    assert report["contract_pass"] is True
    assert report["K_raw_observations"] == 384
    assert report["J_native_tubelet_tokens"] == 192
    assert report["Q0_base_candidates"] == 192
    assert report["Q_level_lengths"] == [192, 96, 48, 24, 12, 6]
    assert report["Q_total_candidates"] == 378
    assert report["model_config_exact_match"] is True
    assert report["dataset_config_match_except_coordinate_mode"] is True
    assert report["coordinate_modes"] == {
        "selected_axis": "uniform_rank_seconds",
        "physical_metric": "physical_time_seconds",
    }
    assert report["train_window_crop_uses_gt"] is True
    assert report["within_window_subsample_uses_gt"] is False
    assert report["amp_contract"] == {
        "enabled": True,
        "init_scale": 1024.0,
        "fp16_compress": False,
        "fail_on_non_finite_grad": True,
        "max_consecutive_skips": 4,
        "max_total_skips_per_epoch": 8,
    }
    assert report["structural_lineage"]["transformer_depth"] == 12
    assert report["structural_lineage"]["adapter_indices"] == list(range(12))
    assert report["production_engine_contract"] == {
        "train_batch_size": 2,
        "train_num_workers": 2,
        "ema": True,
        "scheduler": "LinearWarmupCosineAnnealingLR",
        "warmup_epoch": 5,
        "optimizer_lr": 1.0e-4,
        "adapter_lr": 2.0e-4,
    }
    assert output.is_file()


def test_g1a_track_validator_rejects_any_non_coordinate_pipeline_difference(tmp_path):
    bad_config = Config.fromfile(DEFAULT_CONFIGS["selected_axis"], lazy_import=False)
    next(
        step for step in bad_config.dataset.train.pipeline if step["type"] == "LoadFrames"
    )["target_len"] = 256
    configs = copy.copy(DEFAULT_CONFIGS)
    configs["selected_axis"] = bad_config

    with pytest.raises(RuntimeError, match="K=384"):
        validate_track(configs)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cfg: cfg.solver.train.update(batch_size=1), "batch size"),
        (lambda cfg: cfg.solver.update(ema=False), "EMA"),
        (lambda cfg: cfg.optimizer.update(lr=0.0), "optimizer learning rate"),
        (
            lambda cfg: cfg.optimizer.backbone.custom[0].update(lr=0.0),
            "adapter learning rate",
        ),
        (lambda cfg: cfg.scheduler.update(type="MultiStepLR"), "scheduler"),
    ],
)
def test_g1a_track_validator_rejects_non_production_engine_contract(mutation, message):
    configs = {}
    for name, path in DEFAULT_CONFIGS.items():
        cfg = Config.fromfile(path, lazy_import=False)
        mutation(cfg)
        configs[name] = cfg

    with pytest.raises(RuntimeError, match=message):
        validate_track(configs)
