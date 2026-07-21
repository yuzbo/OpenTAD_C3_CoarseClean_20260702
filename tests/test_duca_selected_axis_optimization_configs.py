from __future__ import annotations

import json
from pathlib import Path

from mmengine.config import Config

from tools.bata import duca_selected_axis_training
from tools.bata.validate_duca_protected_e2e_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
FULL_MODEL_GATE = ROOT / "tools" / "bata" / "run_duca_protected_e2e_exact_full_model_gate.py"
GATE_LAUNCHER = ROOT / "scripts" / "run_duca_selected_axis_optimization_gate_gpu1.sh"
VARIANTS = {
    "exact_uniform": "duca_exact_uniform_fixed384_official60.py",
    "direct": "duca_protected_e2e_direct025_fixed384_official60.py",
    "homotopy": "duca_protected_e2e_homotopy025_fixed384_official60.py",
    "companion": (
        "duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py"
    ),
}


def test_selected_axis_optimization_configs_pass_the_official60_contract() -> None:
    payloads = {
        name: validate_config(CONFIG_DIR / filename)
        for name, filename in VARIANTS.items()
    }

    assert all(payload["ok"] for payload in payloads.values())
    assert payloads["direct"]["training_uniform_companion_fraction"] == 0.0
    assert payloads["homotopy"]["training_uniform_companion_fraction"] == 0.0
    assert payloads["companion"]["training_uniform_companion_fraction"] == 0.5


def test_selected_axis_configs_use_their_60_epoch_formal_training_protocol() -> None:
    for filename in VARIANTS.values():
        cfg = Config.fromfile(str(CONFIG_DIR / filename))
        assert (
            cfg.workflow.formal_protocol
            == duca_selected_axis_training.FORMAL_PROTOCOL
        )
        contract = duca_selected_axis_training.formal_training_contract(cfg)
        assert contract is not None
        assert contract["end_epoch"] == 60
        assert contract["expected_train_batches_per_epoch"] == 100
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert contract["checkpoint_criterion"] == (
            "terminal_epoch_59_state_dict_ema"
        )


def test_selected_axis_formal_protocol_rejects_semantic_cli_overrides() -> None:
    duca_selected_axis_training.assert_safe_cfg_options(
        {
            "work_dir": "/tmp/run",
            "model": {"backbone": {"custom": {"pretrain": "/tmp/pretrain"}}},
        },
        entrypoint="tools/train.py",
    )
    try:
        duca_selected_axis_training.assert_safe_cfg_options(
            {"workflow": {"end_epoch": 1}},
            entrypoint="tools/train.py",
        )
    except RuntimeError as error:
        assert "workflow.end_epoch" in str(error)
    else:
        raise AssertionError("semantic training override was accepted")


def test_selected_axis_runtime_binding_reopens_the_exact_gate_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commit = "a" * 40
    variant = "direct025"
    config_path = CONFIG_DIR / duca_selected_axis_training.VARIANT_CONFIGS[variant]
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    pretrain.write_bytes(b"pretrain")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")

    full_gate_path = tmp_path / "full_model" / f"{config_path.stem}.json"
    full_gate_path.parent.mkdir()
    full_gate = {
        "ok": True,
        "config_sha256": duca_selected_axis_training.sha256_file(config_path),
        "runtime": {"git_commit": commit},
        "adatad_pretrain": {
            "sha256": duca_selected_axis_training.sha256_file(pretrain)
        },
    }
    full_gate_path.write_text(
        json.dumps(full_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate_suite_path = tmp_path / "gate_suite.json"
    gate_suite = {
        "schema": "duca_selected_axis_optimization_gate_v1",
        "ok": True,
        "formal_training_unlocked": True,
        "git_commit": commit,
        "artifacts": [
            {
                "path": str(full_gate_path),
                "sha256": duca_selected_axis_training.sha256_file(full_gate_path),
            }
        ],
    }
    gate_suite_path.write_text(
        json.dumps(gate_suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DUCA_SELECTED_OPT_GATE_SUITE", str(gate_suite_path))
    monkeypatch.setenv(
        "DUCA_SELECTED_OPT_GATE_SUITE_SHA256",
        duca_selected_axis_training.sha256_file(gate_suite_path),
    )

    cfg = Config.fromfile(str(config_path))
    evaluation_config = cfg.evaluation.to_dict()
    evaluation_config["ground_truth_filename"] = str(annotation)
    bindings = duca_selected_axis_training.build_runtime_bindings(
        git_commit=commit,
        variant=variant,
        seed=3407,
        slurm_job_id="1",
        source_config_path=config_path,
        source_config_sha256=duca_selected_axis_training.sha256_file(config_path),
        resolved_config_sha256="b" * 64,
        runtime_config_sha256="c" * 64,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config=evaluation_config,
        runtime_pretrain_path=pretrain,
    )
    assert bindings["variant"] == variant
    assert bindings["gate_suite_sha256"] == (
        duca_selected_axis_training.sha256_file(gate_suite_path)
    )
    assert bindings["full_model_gate_sha256"] == (
        duca_selected_axis_training.sha256_file(full_gate_path)
    )


def test_selected_axis_variants_share_one_official_head_and_differ_only_in_training_policy() -> None:
    configs = {
        name: Config.fromfile(str(CONFIG_DIR / filename))
        for name, filename in VARIANTS.items()
    }

    first_head = configs["direct"].model.rpn_head.to_dict()
    for name, cfg in configs.items():
        selector = cfg.model.frame_selector
        assert cfg.model.rpn_head.to_dict() == first_head
        assert "physical_grid_actionformer" not in cfg.model.rpn_head
        assert selector.detector_output_coordinate_space == "selected_axis_index"
        assert selector.remap_gt_to_selected_axis is True
        expected_gradient_mode = (
            "none" if name == "exact_uniform" else "protected_structured_transport"
        )
        assert selector.detector_gradient_mode == expected_gradient_mode
        assert int(selector.budget) == 384
        assert int(selector.dense_window_size) == 768

    direct = configs["direct"].model.frame_selector.loss_weight_schedule
    homotopy = configs["homotopy"].model.frame_selector.loss_weight_schedule
    companion = configs["companion"].model.frame_selector
    assert float(direct.policy_alpha.start) == float(direct.policy_alpha.end) == 1.0
    assert float(direct.detector_gradient.start) == 0.0
    assert float(direct.detector_gradient.end) == 0.25
    assert int(direct.detector_gradient.warmup_steps) == 2100
    assert int(direct.detector_gradient.transition_steps) == 1500
    assert float(homotopy.policy_alpha.start) == 0.0
    assert float(homotopy.policy_alpha.end) == 1.0
    assert float(companion.training_uniform_companion_fraction) == 0.5


def test_selected_axis_full_model_gate_reuses_the_production_amp_replay_path() -> None:
    source = FULL_MODEL_GATE.read_text(encoding="utf-8")
    launcher = GATE_LAUNCHER.read_text(encoding="utf-8")

    assert "train_one_epoch(" in source
    assert "force_amp_overflow_attempts=1" in source
    assert "max_amp_retries_per_batch=int(cfg.workflow.max_amp_retries_per_batch)" in source
    assert "fail_on_amp_replay_exhaustion=True" in source
    assert "ModelEma(ddp)" in source
    assert "_position_scheduler_at_successful_step(" in source
    assert "initial_nonzero_scheduler_lrs" in source
    assert "scheduler_updates" in source
    assert "duca_schedule_updates" in source
    assert "FORMAL_SEED = 3407" in source
    assert "exact_uniform_positions(" in source
    full_model_block = launcher.split("full_model_configs=(", 1)[1].split(")", 1)[0]
    assert "duca_exact_uniform_fixed384_official60.py" in full_model_block
    assert '"four_matched_variants_full_model_gate_passed"' in launcher
