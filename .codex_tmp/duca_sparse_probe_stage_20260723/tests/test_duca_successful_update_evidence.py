from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from mmengine.config import Config

from tools.bata.duca_p0_training import (
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    atomic_write_json,
    build_checkpoint_metadata,
    build_training_audit,
    canonical_sha256,
    formal_training_contract,
    new_update_audit,
    restore_training_state,
    sha256_file,
)
from tools.bata import finalize_duca_transition_only_p0_run as finalizer
from tools.bata.duca_p0_evaluation import (
    evaluation_config_sha256,
    official_evaluator_identity,
    recompute_official_map,
)
from tools.bata.validate_duca_transition_only_p0_suite import (
    validate_post_run_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    ROOT
    / "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)


def _contract() -> dict:
    return {
        "expected_train_batches_per_epoch": 100,
        "expected_successful_optimizer_updates": 13200,
        "end_epoch": 132,
        "max_amp_retries_per_batch": 8,
        "primary_checkpoint_epoch": 131,
        "primary_checkpoint_state_key": "state_dict_ema",
        "checkpoint_criterion": "terminal_epoch_131_state_dict_ema",
        "checkpoint_interval": 5,
    }


def _bindings(
    tmp_path: Path, *, eval_config_sha256: str = "6" * 64
) -> dict:
    annotation = (tmp_path / "annotation.json").resolve()
    class_map = (tmp_path / "category_idx.txt").resolve()
    return {
        "git_commit": "a" * 40,
        "variant": "uniform",
        "seed": 0,
        "slurm_job_id": "123",
        "source_config_path": str(FORMAL_CONFIG.resolve()),
        "source_config_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
        "runtime_config_sha256": "d" * 64,
        "shared_protocol_sha256": "e" * 64,
        "variant_contract_sha256": "f" * 64,
        "core_gate_sha256": "1" * 64,
        "ddp_pilot_sha256": "2" * 64,
        "canonical_env_sha256": "3" * 64,
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": (
            sha256_file(annotation) if annotation.is_file() else "4" * 64
        ),
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": (
            sha256_file(class_map) if class_map.is_file() else "5" * 64
        ),
        "evaluation_config_sha256": eval_config_sha256,
    }


def _complete_audit(
    tmp_path: Path, *, eval_config_sha256: str = "6" * 64
) -> dict:
    counters = new_update_audit()
    for key in (
        "attempted_batches",
        "optimizer_attempts",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        counters[key] = 13200
    records = [{"epoch": epoch} for epoch in range(132)]
    return build_training_audit(
        contract=_contract(),
        bindings=_bindings(tmp_path, eval_config_sha256=eval_config_sha256),
        epoch=131,
        train_batches_per_epoch=100,
        update_audit=counters,
        epoch_records=records,
        scheduler_last_epoch=13200,
        selector_step=13200,
        scaler_scale=32768.0,
        uses_ema=True,
        complete=True,
    )


def test_formal_config_freezes_successful_update_contract() -> None:
    contract = formal_training_contract(Config.fromfile(str(FORMAL_CONFIG)))
    assert contract is not None
    assert contract["expected_successful_optimizer_updates"] == 13200
    assert contract["checkpoint_interval"] == 5
    assert contract["primary_checkpoint_epoch"] == 131


def test_resume_metadata_rejects_tampered_training_audit(tmp_path: Path) -> None:
    audit = _complete_audit(tmp_path)
    metadata = build_checkpoint_metadata(audit)
    counters, records = restore_training_state(
        {"experiment_metadata": metadata},
        contract=_contract(),
        bindings=_bindings(tmp_path),
    )
    assert counters["successful_optimizer_updates"] == 13200
    assert len(records) == 132

    tampered = json.loads(json.dumps(metadata))
    tampered["training_audit"]["update_audit"]["successful_optimizer_updates"] -= 1
    tampered_unsigned = dict(tampered)
    tampered_unsigned.pop("metadata_sha256")
    tampered["metadata_sha256"] = canonical_sha256(tampered_unsigned)
    with pytest.raises(RuntimeError, match="training audit hash"):
        restore_training_state(
            {"experiment_metadata": tampered},
            contract=_contract(),
            bindings=_bindings(tmp_path),
        )


def test_global_rng_restore_forces_serialized_tensors_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class DeviceState:
        def __init__(self, name: str):
            self.name = name

        def cpu(self):
            return f"cpu:{self.name}"

    fake_cuda = SimpleNamespace(
        set_rng_state_all=lambda states: captured.setdefault("cuda", states)
    )
    fake_torch = SimpleNamespace(
        set_rng_state=lambda state: captured.setdefault("cpu", state),
        cuda=fake_cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from tools.bata.duca_p0_training import restore_global_rng_state

    restore_global_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": DeviceState("main"),
            "torch_cuda": [DeviceState("cuda0")],
        }
    )
    assert captured["cpu"] == "cpu:main"
    assert captured["cuda"] == ["cpu:cuda0"]


def test_train_resume_loads_checkpoint_on_cpu() -> None:
    source = (ROOT / "tools/train.py").read_text(encoding="utf-8")
    assert 'torch.load(args.resume, map_location="cpu")' in source
    assert "torch.load(args.resume, map_location=device)" not in source


def test_checkpoint_writer_saves_scaler_metadata_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_save(payload, path):
        captured.update(payload)
        Path(path).write_bytes(b"checkpoint")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(save=fake_save))
    spec = importlib.util.spec_from_file_location(
        "duca_checkpoint_writer_under_test",
        ROOT / "opentad/utils/checkpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    stateful = SimpleNamespace(state_dict=lambda: {"value": 1})
    model_ema = SimpleNamespace(module=stateful)
    scaler = SimpleNamespace(state_dict=lambda: {"scale": 1024.0})
    metadata = build_checkpoint_metadata(_complete_audit(tmp_path))
    checkpoint = module.save_checkpoint(
        stateful,
        model_ema,
        stateful,
        stateful,
        131,
        work_dir=str(tmp_path),
        scaler=scaler,
        rng_state={"python": "state"},
        experiment_metadata=metadata,
        experiment_sidecar_schema=DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    )
    assert captured["grad_scaler"] == {"scale": 1024.0}
    assert captured["rng_state"] == {"python": "state"}
    sidecar = json.loads(Path(checkpoint + ".metadata.json").read_text())
    assert sidecar["checkpoint_sha256"] == sha256_file(checkpoint)
    with pytest.raises(FileExistsError):
        module.save_checkpoint(
            stateful,
            model_ema,
            stateful,
            stateful,
            131,
            work_dir=str(tmp_path),
            experiment_metadata=metadata,
            experiment_sidecar_schema=DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
        )


def test_finalizer_reopens_complete_artifact_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        finalizer,
        "_inspect_checkpoint_payload",
        lambda _path, _metadata: {
            "payload_reopened": True,
            "epoch": 131,
            "scheduler_last_epoch": 13200,
            "selector_schedule_steps": {
                "state_dict": 13200,
                "state_dict_ema": 13200,
            },
            "grad_scaler_present": True,
            "global_rng_state_present": True,
            "embedded_metadata_exact": True,
        },
    )
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "category_idx.txt"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "video_validation_0001": {
                        "subset": "validation",
                        "annotations": [
                            {"label": "Action", "segment": [0.0, 1.0]}
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    class_map.write_text("1 Action\n", encoding="utf-8")
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": str(annotation.resolve()),
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 1,
    }
    eval_config_sha256 = evaluation_config_sha256(evaluation_config)
    audit = _complete_audit(
        tmp_path, eval_config_sha256=eval_config_sha256
    )
    audit_path = tmp_path / "training_audit.json"
    atomic_write_json(audit_path, audit)

    checkpoint = tmp_path / "epoch_131.pth"
    checkpoint.write_bytes(b"terminal-checkpoint")
    metadata = build_checkpoint_metadata(audit)
    sidecar = {
        "schema_version": DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_metadata": metadata,
    }
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    sidecar_path = tmp_path / "epoch_131.pth.metadata.json"
    atomic_write_json(sidecar_path, sidecar)

    prediction = tmp_path / "result_detection.json"
    prediction.write_text(
        json.dumps(
            {
                "results": {
                    "video_validation_0001": [
                        {
                            "label": "Action",
                            "segment": [0.0, 1.0],
                            "score": 0.9,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    recomputed = recompute_official_map(prediction, evaluation_config)
    evaluation = {
        "schema_version": "duca_p0_terminal_evaluation_v3",
        "git_commit": audit["git_commit"],
        "config_path": str(FORMAL_CONFIG.resolve()),
        "config_sha256": audit["source_config_sha256"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": 131,
        "checkpoint_state_key": "state_dict_ema",
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": sha256_file(prediction),
        "metrics": recomputed["metrics"],
        "result_count": recomputed["result_count"],
        "video_count": recomputed["video_count"],
        "evaluator": official_evaluator_identity(),
        "evaluation_config": evaluation_config,
        "evaluation_config_sha256": eval_config_sha256,
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": sha256_file(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": sha256_file(class_map),
    }
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    evaluation_path = tmp_path / "terminal_evaluation.json"
    atomic_write_json(evaluation_path, evaluation)

    bindings = _bindings(
        tmp_path, eval_config_sha256=eval_config_sha256
    )
    manifest = {
        "variant": "uniform",
        "git_commit": bindings["git_commit"],
        "seed": 0,
        "config_sha256": bindings["source_config_sha256"],
        "resolved_config_sha256": bindings["resolved_config_sha256"],
        "variant_contract_sha256": bindings["variant_contract_sha256"],
        "shared_protocol_sha256": bindings["shared_protocol_sha256"],
        "core_gate_json_sha256": bindings["core_gate_sha256"],
        "ddp_pilot_json_sha256": bindings["ddp_pilot_sha256"],
        "canonical_env_sha256": bindings["canonical_env_sha256"],
        "evaluation_annotation_sha256": bindings[
            "evaluation_annotation_sha256"
        ],
        "evaluation_class_map_sha256": bindings[
            "evaluation_class_map_sha256"
        ],
        "evaluation_config_sha256": bindings["evaluation_config_sha256"],
    }
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    evidence = finalizer.finalize_run(
        variant="uniform",
        run_manifest_path=manifest_path,
        training_audit_path=audit_path,
        checkpoint_path=checkpoint,
        checkpoint_sidecar_path=sidecar_path,
        evaluation_path=evaluation_path,
    )
    evidence_path = tmp_path / "post_run_evidence.json"
    atomic_write_json(evidence_path, evidence)
    expected_bindings = {
        "git_commit": bindings["git_commit"],
        "seed": 0,
        "config_sha256": bindings["source_config_sha256"],
        "resolved_config_sha256": bindings["resolved_config_sha256"],
        "variant_contract_sha256": bindings["variant_contract_sha256"],
        "core_gate_sha256": bindings["core_gate_sha256"],
        "ddp_pilot_sha256": bindings["ddp_pilot_sha256"],
        "evaluation_annotation_sha256": bindings[
            "evaluation_annotation_sha256"
        ],
        "evaluation_class_map_sha256": bindings[
            "evaluation_class_map_sha256"
        ],
        "evaluation_config_sha256": bindings["evaluation_config_sha256"],
    }
    result = validate_post_run_evidence(
        evidence_path,
        variant="uniform",
        protocol_sha256=bindings["shared_protocol_sha256"],
        bindings=expected_bindings,
    )
    assert result["validated"] is True

    forged = json.loads(json.dumps(evaluation))
    forged["metrics"]["average_mAP"] = 0.65
    unsigned = dict(forged)
    unsigned.pop("evaluation_sha256")
    forged["evaluation_sha256"] = canonical_sha256(unsigned)
    forged_path = tmp_path / "forged_terminal_evaluation.json"
    atomic_write_json(forged_path, forged)
    with pytest.raises(ValueError, match="official prediction recomputation"):
        finalizer.finalize_run(
            variant="uniform",
            run_manifest_path=manifest_path,
            training_audit_path=audit_path,
            checkpoint_path=checkpoint,
            checkpoint_sidecar_path=sidecar_path,
            evaluation_path=forged_path,
        )

    wrong_config = json.loads(json.dumps(evaluation))
    wrong_config["config_sha256"] = "0" * 64
    unsigned = dict(wrong_config)
    unsigned.pop("evaluation_sha256")
    wrong_config["evaluation_sha256"] = canonical_sha256(unsigned)
    wrong_config_path = tmp_path / "wrong_config_terminal_evaluation.json"
    atomic_write_json(wrong_config_path, wrong_config)
    with pytest.raises(ValueError, match="detector config differs"):
        finalizer.finalize_run(
            variant="uniform",
            run_manifest_path=manifest_path,
            training_audit_path=audit_path,
            checkpoint_path=checkpoint,
            checkpoint_sidecar_path=sidecar_path,
            evaluation_path=wrong_config_path,
        )

    prediction.write_text('{"results": {"tampered": []}}', encoding="utf-8")
    with pytest.raises(ValueError, match="prediction hash"):
        finalizer.finalize_run(
            variant="uniform",
            run_manifest_path=manifest_path,
            training_audit_path=audit_path,
            checkpoint_path=checkpoint,
            checkpoint_sidecar_path=sidecar_path,
            evaluation_path=evaluation_path,
        )
