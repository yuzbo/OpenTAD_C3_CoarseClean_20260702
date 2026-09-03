import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
ADATAD_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_adatad_precheck.py"
ZEROSHOT_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_zeroshot_actionness_precheck.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_online_adatad_precheck.py"
ADATAD_LAUNCHER = ROOT / "scripts" / "run_duca_online_adatad_precheck_gpu1.sh"
ZEROSHOT_LAUNCHER = ROOT / "scripts" / "run_duca_online_zeroshot_actionness_precheck_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_duca_online_adatad_precheck_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(path):
    return Config.fromfile(str(path))


def _assert_common_config_contract(cfg):
    assert cfg.model.type == "SingleStageDetector"
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.rpn_head.type == "DucaOnlinePrecheckHead"
    assert int(cfg.model.frame_selector.budget) <= 384
    assert int(cfg.duca_online_precheck_contract.budget_max) <= 384
    assert cfg.duca_online_precheck_contract.no_ledger_decision is True
    assert cfg.duca_online_precheck_contract.coordinate_space == "original_time"
    assert cfg.duca_online_precheck_contract.teacher_free_eval is True
    assert cfg.duca_online_precheck_contract.selected_axis_remap_required is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    selector_text = repr(cfg.model.frame_selector).lower()
    assert "ledger_path" not in selector_text
    assert "value_transport" not in selector_text


def test_duca_online_adatad_precheck_config_is_registry_buildable_contract():
    cfg = _cfg(ADATAD_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.duca_online_precheck_contract.actionness_source == "online_probe"
    assert cfg.model.frame_selector.actionness_source_cfg.type == "DucaOnlineProbeActionnessSource"
    assert cfg.model.rpn_head.teacher_cfg.train_only is True
    assert cfg.model.rpn_head.teacher_cfg.enabled_for_inference is False


def test_duca_online_zeroshot_precheck_config_uses_zero_shot_actionness_source():
    cfg = _cfg(ZEROSHOT_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.duca_online_precheck_contract.actionness_source == "zero_shot_motion"
    assert cfg.model.frame_selector.actionness_source_cfg.type == "ZeroShotMotionActionnessSource"
    assert cfg.model.frame_selector.actionness_source_cfg.no_train_gt is True
    assert cfg.model.frame_selector.actionness_source_cfg.no_teacher is True


def test_duca_online_validator_json_summary_fields(monkeypatch):
    monkeypatch.setenv("DUCA_ONLINE_PRECHECK_RUNTIME", "0")
    validator = _load_validator()

    summary = validator.validate_config(str(ADATAD_CONFIG), run_runtime=False)

    assert summary["config_import"] is True
    assert summary["build_detector"] == "skipped"
    assert summary["no_ledger"] is True
    assert summary["teacher_only_train_loss"] is True
    assert summary["no_teacher_in_inference"] is True
    assert summary["gt_reaches_detector_train"] == "runtime_required"
    assert summary["budget_lte_384"] is True
    assert summary["selected_positions_original_time"] == "runtime_required"
    assert summary["masks_selected_count"] == "runtime_required"
    assert summary["remap_metadata_present"] == "runtime_required"
    assert summary["raw_prediction_cache_forbidden"] is True

    output = subprocess.check_output(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(ADATAD_CONFIG),
            "--no-runtime",
        ],
        cwd=str(ROOT),
        text=True,
    )
    cli_summary = json.loads(output)
    assert cli_summary["config_path"].endswith("duca_online_adatad_precheck.py")
    assert cli_summary["build_detector"] == "skipped"


def test_duca_online_launchers_are_precheck_first_gpu1_guarded_and_fail_closed():
    for launcher in (ADATAD_LAUNCHER, ZEROSHOT_LAUNCHER):
        text = launcher.read_text(encoding="utf-8")
        assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
        assert "FULLTRAIN_CANDIDATE" in text
        assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
        assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
        assert "validate_duca_online_adatad_precheck.py" in text
        assert "tests/test_duca_online_precheck_config.py" in text
        assert "formal full train must run inside a Slurm allocation/step" in text
        assert "tools/train.py" in text
