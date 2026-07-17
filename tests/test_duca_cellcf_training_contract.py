import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.duca_cellcf_training import (
    assert_safe_cfg_options,
    build_runtime_bindings,
    canonical_sha256,
    expected_runtime_config_sha256,
    formal_training_contract,
)
from tools.bata.duca_p0_evaluation import evaluation_config_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cellcf_formal_contract_preserves_132_epoch_successful_update_protocol() -> None:
    cfg = Config.fromfile(str(CONFIG))

    contract = formal_training_contract(cfg)

    assert contract["formal_protocol"] == "duca_cellcf_v1"
    assert contract["end_epoch"] == 132
    assert contract["checkpoint_interval"] == 5
    assert contract["expected_successful_optimizer_updates"] == 13200
    assert contract["primary_checkpoint_epoch"] == 131
    assert contract["primary_checkpoint_state_key"] == "state_dict_ema"


def test_cellcf_runtime_bindings_reject_artifact_replacement(tmp_path, monkeypatch) -> None:
    annotation = tmp_path / "annotations.json"
    class_map = tmp_path / "classes.txt"
    pretrain = tmp_path / "videomae.pth"
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("1 Action\n", encoding="utf-8")
    pretrain.write_bytes(b"formal-test-checkpoint")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol_sha = "1" * 64
    order_sha = "2" * 64
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "duca_cellcf_real_loader_cuda_gate_v1",
                "ok": True,
                "git_commit": commit,
                "synthetic_gate_sha256": "3" * 64,
                "config_contract": {
                    "training_profile": "exposure132",
                },
                "evaluation_annotation_sha256": _sha(annotation),
                "evaluation_class_map_sha256": _sha(class_map),
                "dataset": {
                    "annotation_sha256": _sha(annotation),
                    "class_map_sha256": _sha(class_map),
                },
                "assets": {
                    "videomae_checkpoint": {
                        "path": str(pretrain),
                        "sha256": _sha(pretrain),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "schema": "duca_cellcf_ddp_pilot_suite_v1",
                "ok": True,
                "git_commit": commit,
                "training_profile": "exposure132",
                "real_loader_gate_sha256": _sha(gate),
                "shared_protocol_sha256": protocol_sha,
                "ordered_exposure_sha256": order_sha,
                "variant_order": ["uniform", "transition_beta0", "cellcf"],
            }
        ),
        encoding="utf-8",
    )
    evaluation = {
        "type": "mAP",
        "ground_truth_filename": str(annotation),
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "subset": "validation",
    }
    resolved_sha = "4" * 64
    monkeypatch.setenv("DUCA_EXPECTED_COMMIT", commit)
    monkeypatch.setenv("DUCA_CELLCF_VARIANT", "cellcf")
    monkeypatch.setenv("DUCA_CELLCF_RESOLVED_CONFIG_SHA256", resolved_sha)
    monkeypatch.setenv("DUCA_CELLCF_RUNTIME_CONFIG_SHA256", "5" * 64)
    monkeypatch.setenv("DUCA_CELLCF_PROTOCOL_SHA256", protocol_sha)
    monkeypatch.setenv("DUCA_CELLCF_ORDER_SHA256", order_sha)
    monkeypatch.setenv("DUCA_CELLCF_GATE_JSON", str(gate))
    monkeypatch.setenv("DUCA_CELLCF_GATE_SHA256", _sha(gate))
    monkeypatch.setenv("DUCA_CELLCF_DDP_PILOT_JSON", str(pilot))
    monkeypatch.setenv("DUCA_CELLCF_DDP_PILOT_SHA256", _sha(pilot))
    monkeypatch.setenv("DUCA_CELLCF_ANNOTATION_SHA256", _sha(annotation))
    monkeypatch.setenv("DUCA_CELLCF_CLASS_MAP_SHA256", _sha(class_map))
    monkeypatch.setenv("DUCA_CELLCF_EVALUATION_CONFIG_SHA256", evaluation_config_sha256(evaluation))
    import tools.bata.validate_duca_cellcf_real_loader_gate as gate_module
    import tools.bata.validate_duca_cellcf_ddp_pilot as pilot_module

    monkeypatch.setattr(
        gate_module,
        "validate_real_loader_gate_artifact",
        lambda *args, **kwargs: {"sha256": _sha(gate)},
    )
    monkeypatch.setattr(
        pilot_module,
        "validate_pilot_artifact",
        lambda *args, **kwargs: {"sha256": _sha(pilot)},
    )

    bindings = build_runtime_bindings(
        git_commit=commit,
        variant="",
        seed=0,
        slurm_job_id="12345",
        source_config_path=CONFIG,
        source_config_sha256=_sha(CONFIG),
        resolved_config_sha256=resolved_sha,
        runtime_config_sha256="5" * 64,
        runtime_pretrain_path=pretrain,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config=evaluation,
    )
    assert bindings["real_loader_gate_sha256"] == _sha(gate)
    assert bindings["ddp_pilot_sha256"] == _sha(pilot)

    monkeypatch.setenv("DUCA_CELLCF_GATE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="gate hash"):
        build_runtime_bindings(
            git_commit=commit,
            variant="",
            seed=0,
            slurm_job_id="12345",
            source_config_path=CONFIG,
            source_config_sha256=_sha(CONFIG),
            resolved_config_sha256=resolved_sha,
            runtime_config_sha256="5" * 64,
            runtime_pretrain_path=pretrain,
            evaluation_annotation_path=annotation,
            evaluation_class_map_path=class_map,
            evaluation_config=evaluation,
        )


def test_cellcf_rejects_semantic_runtime_overrides(tmp_path) -> None:
    cfg = Config.fromfile(str(CONFIG))
    with pytest.raises(RuntimeError, match="semantic --cfg-options"):
        assert_safe_cfg_options(
            cfg,
            {"model.frame_selector.counterfactual_utility_distillation_weight": 0.0},
            entrypoint="tools/train.py",
        )
    with pytest.raises(RuntimeError, match="raw-prediction"):
        assert_safe_cfg_options(
            cfg,
            {"inference.load_from_raw_predictions": True},
            entrypoint="tools/test.py",
        )

    work_dir = tmp_path / "work"
    expected = expected_runtime_config_sha256(
        CONFIG,
        {
            "work_dir": str(work_dir),
            "model.backbone.custom.pretrain": str(tmp_path / "model.pth"),
        },
        experiment_id=0,
        gpu_num=1,
        entrypoint="tools/train.py",
    )
    assert len(expected) == 64


def test_cellcf_entrypoints_and_finalizer_bind_effective_configs() -> None:
    train_source = (ROOT / "tools/train.py").read_text(encoding="utf-8")
    binding_source = (ROOT / "tools/bata/duca_cellcf_training.py").read_text(
        encoding="utf-8"
    )
    test_source = (ROOT / "tools/test.py").read_text(encoding="utf-8")
    finalizer_source = (ROOT / "tools/bata/finalize_duca_cellcf_run.py").read_text(
        encoding="utf-8"
    )
    launcher_source = (ROOT / "scripts/run_duca_cellcf_variant.sh").read_text(
        encoding="utf-8"
    )
    assert "duca_cellcf_training.assert_safe_cfg_options" in train_source
    assert "DUCA_CELLCF_RUNTIME_CONFIG_SHA256" in binding_source
    assert "duca_cellcf_training.assert_safe_cfg_options" in test_source
    assert "DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256" in test_source
    assert '"runtime_config_sha256": "runtime_config_sha256"' in finalizer_source
    assert 'manifest.get("evaluation_runtime_config_sha256")' in finalizer_source
    assert "expected_runtime_config_sha256" in launcher_source
