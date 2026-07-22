from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from tools.bata import duca_selected_axis_training as training


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
SUPPORTED_BOUNDARY_VARIANTS = (
    "two_stage_exact_uniform",
    "gaussian_matched_g0",
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
)


def test_train_entrypoint_dispatches_selected_axis_runtime_binder(monkeypatch) -> None:
    source_path = ROOT / "tools" / "train.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_select_duca_training", "_dispatch_duca_runtime_bindings"}
    ]
    namespace = {
        "duca_cellcf_training": SimpleNamespace(),
        "duca_p0_training": SimpleNamespace(),
        "duca_protected_physical_training": SimpleNamespace(
            FORMAL_PROTOCOL="protected"
        ),
        "duca_selected_axis_training": training,
    }
    exec(
        compile(
            ast.Module(body=selected_functions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    observed = {}

    def fake_binder(**kwargs):
        observed.update(kwargs)
        return {"bound": True}

    monkeypatch.setattr(training, "build_runtime_bindings", fake_binder)
    selected = namespace["_select_duca_training"](training.FORMAL_PROTOCOL)
    assert namespace["_select_duca_training"](training.R5_FORMAL_PROTOCOL) is training
    result = namespace["_dispatch_duca_runtime_bindings"](
        selected,
        {"git_commit": "a" * 40},
        selector_initialization={"enabled": True},
        formal_protocol=training.R5_FORMAL_PROTOCOL,
        r5_cell={"seed": 3407},
    )

    assert selected is training
    assert result == {"bound": True}
    assert observed == {
        "git_commit": "a" * 40,
        "selector_initialization": {"enabled": True},
        "formal_protocol": training.R5_FORMAL_PROTOCOL,
        "r5_cell": {"seed": 3407},
    }


def test_submit_frozen_pretrain_binding_rejects_replacement(tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain.pth"
    pretrain.write_bytes(b"original")
    digest = training.sha256_file(pretrain)
    assert training.validate_frozen_pretrain_binding(
        runtime_path=pretrain,
        expected_path=pretrain,
        expected_sha256=digest,
    )["sha256"] == digest

    pretrain.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="content drifted after submission"):
        training.validate_frozen_pretrain_binding(
            runtime_path=pretrain,
            expected_path=pretrain,
            expected_sha256=digest,
        )


def test_submit_frozen_pretrain_binding_rejects_path_drift(tmp_path: Path) -> None:
    expected = tmp_path / "expected.pth"
    runtime = tmp_path / "runtime.pth"
    expected.write_bytes(b"same")
    runtime.write_bytes(b"same")
    with pytest.raises(RuntimeError, match="path drifted after submission"):
        training.validate_frozen_pretrain_binding(
            runtime_path=runtime,
            expected_path=expected,
            expected_sha256=training.sha256_file(expected),
        )


@pytest.mark.parametrize("variant", sorted(training.LOCKED_ALIGNMENT_VARIANTS))
def test_unaligned_feedback_variants_are_locked_at_production_binding(
    variant: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="requires real legal hard-swap alignment"):
        training.build_runtime_bindings(
            git_commit="a" * 40,
            variant=variant,
            seed=3407,
            slurm_job_id="1",
            source_config_path=tmp_path / "unused.py",
            source_config_sha256="b" * 64,
            resolved_config_sha256="c" * 64,
            runtime_config_sha256="d" * 64,
            evaluation_annotation_path=tmp_path / "unused.json",
            evaluation_class_map_path=tmp_path / "unused.txt",
            evaluation_config={},
            runtime_pretrain_path=tmp_path / "unused.pth",
        )


@pytest.mark.parametrize("variant", sorted(training.BOUNDARY_ALIGNMENT_VARIANTS))
def test_boundary_feedback_runtime_requires_exact_alignment_authorization(
    variant: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tools.bata import duca_boundary_burst_hard_swap_alignment as alignment

    commit = "a" * 40
    config = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    frontend = tmp_path / "frontend.pth"
    for path, content in (
        (pretrain, b"pretrain"),
        (frontend, b"frontend"),
    ):
        path.write_bytes(content)
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    frontend_sha = training.sha256_file(frontend)
    full_gate = tmp_path / "feedback_full_gate.json"
    _write_json(
        full_gate,
        {
            "schema": "duca_protected_e2e_exact_full_model_gradient_gate_v1",
            "ok": True,
            "config_sha256": training.sha256_file(config),
            "runtime": {"git_commit": commit},
            "adatad_pretrain": {"sha256": training.sha256_file(pretrain)},
            "selector_initialization": {
                "checkpoint_sha256": frontend_sha,
                "checkpoint_epoch": 19,
                "checkpoint_state_key": "state_dict_ema",
                "detector_state_loaded": False,
                "optimizer_state_loaded": False,
                "scheduler_state_loaded": False,
                "receipt_sha256": "b" * 64,
            },
        },
    )
    suite = tmp_path / "gate_suite.json"
    _write_json(
        suite,
        {
            "schema": training.BOUNDARY_BURST_GATE_SCHEMA,
            "ok": True,
            "formal_training_unlocked": True,
            "git_commit": commit,
            "artifacts": [],
        },
    )
    monkeypatch.setenv("DUCA_SELECTED_OPT_GATE_SUITE", str(suite))
    monkeypatch.setenv("DUCA_SELECTED_OPT_GATE_SUITE_SHA256", training.sha256_file(suite))

    observed = {}

    def authorize(**kwargs):
        observed.update(kwargs)
        return {
            "path": str(tmp_path / "alignment.json"),
            "sha256": "c" * 64,
            "self_sha256": "d" * 64,
            "context_sha256": "e" * 64,
            "selected_weakest_projected_family": "R2Q3_privileged_boundary_burst",
            "selected_g0_checkpoint_sha256": "f" * 64,
            "terminal_suite_sha256": "1" * 64,
            "full_model_gate": {
                "path": str(full_gate),
                "sha256": training.sha256_file(full_gate),
            },
        }

    monkeypatch.setattr(alignment, "validate_alignment_artifact", authorize)
    bindings = training.build_runtime_bindings(
        git_commit=commit,
        variant=variant,
        seed=3407,
        slurm_job_id="1",
        source_config_path=config,
        source_config_sha256=training.sha256_file(config),
        resolved_config_sha256="2" * 64,
        runtime_config_sha256="3" * 64,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config={
            "type": "mAP",
            "ground_truth_filename": str(annotation),
            "subset": "validation",
            "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        },
        runtime_pretrain_path=pretrain,
        selector_initialization={
            "enabled": True,
            "checkpoint_path": str(frontend),
            "checkpoint_sha256": frontend_sha,
            "expected_checkpoint_epoch": 19,
            "state_key": "state_dict_ema",
            "reset_state_keys": [],
        },
    )

    assert observed["expected_variant"] == variant
    assert observed["source_config_sha256"] == training.sha256_file(config)
    assert bindings["hard_swap_alignment"]["self_sha256"] == "d" * 64
    assert bindings["full_model_gate_sha256"] == training.sha256_file(full_gate)


def test_boundary_feedback_runtime_fails_closed_when_alignment_rejects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tools.bata import duca_boundary_burst_hard_swap_alignment as alignment

    def reject(**_kwargs):
        raise RuntimeError("sealed alignment rejected")

    monkeypatch.setattr(alignment, "validate_alignment_artifact", reject)
    variant = "boundary_burst_r2q3_g1"
    config = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
    with pytest.raises(RuntimeError, match="sealed alignment rejected"):
        training.build_runtime_bindings(
            git_commit="a" * 40,
            variant=variant,
            seed=3407,
            slurm_job_id="1",
            source_config_path=config,
            source_config_sha256=training.sha256_file(config),
            resolved_config_sha256="b" * 64,
            runtime_config_sha256="c" * 64,
            evaluation_annotation_path=tmp_path / "unused.json",
            evaluation_class_map_path=tmp_path / "unused.txt",
            evaluation_config={},
            runtime_pretrain_path=tmp_path / "unused.pth",
        )


def test_p0_frontend_and_gate_consume_the_submit_frozen_pretrain_contract() -> None:
    submit = (
        ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh"
    ).read_text(encoding="utf-8")
    assert "export DUCA_ADATAD_PRETRAIN_PATH=" in submit
    assert "export DUCA_ADATAD_PRETRAIN_SHA256=" in submit
    for relative in (
        "scripts/run_duca_boundary_burst_p0_gpu1.sh",
        "scripts/run_duca_frontend_pretrain_variant_gpu1.sh",
        "scripts/run_duca_boundary_burst_gate_gpu1.sh",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "DUCA_ADATAD_PRETRAIN_PATH" in source
        assert "DUCA_ADATAD_PRETRAIN_SHA256" in source
        assert "validate_frozen_pretrain_binding" in source
    official = (
        ROOT / "scripts" / "run_duca_two_stage_curriculum_variant_gpu1.sh"
    ).read_text(encoding="utf-8")
    assert '"${DUCA_ADATAD_PRETRAIN_SHA256}"' in official
    assert '"${ADATAD_PRETRAIN_SHA256}"' not in official


def test_sealed_decision_and_full_gate_are_replayed_by_every_consumer() -> None:
    p0 = (ROOT / "scripts" / "run_duca_boundary_burst_p0_gpu1.sh").read_text(
        encoding="utf-8"
    )
    gate = (ROOT / "scripts" / "run_duca_boundary_burst_gate_gpu1.sh").read_text(
        encoding="utf-8"
    )
    official = (
        ROOT / "scripts" / "run_duca_two_stage_curriculum_variant_gpu1.sh"
    ).read_text(encoding="utf-8")

    assert "--p0-real-gate" in p0
    assert "--p0-real-gate-sha256" in p0
    assert "P0_REAL_GATE_SHA256" in p0
    assert "create_p0_training_asformer_consumer_receipt" in p0
    assert "validate_frontend_decision" in gate
    assert "validate_full_model_gate" in gate
    assert '"p0_real_gate"' in gate
    assert "validate_frontend_decision" in official
    assert "validate_full_model_gate" in official
    assert "completion propagation evidence drift" in official


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_runtime_binding_preserves_optional_diagnostic_variant_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commit = "a" * 40
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    checkpoint = tmp_path / "frontend.pth"
    pretrain.write_bytes(b"pretrain")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    checkpoint.write_bytes(b"frontend")
    checkpoint_sha256 = training.sha256_file(checkpoint)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")

    artifacts = []
    for variant in SUPPORTED_BOUNDARY_VARIANTS:
        config_path = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
        initialized = variant != "two_stage_exact_uniform"
        gate = {
            "schema": "duca_protected_e2e_exact_full_model_gradient_gate_v1",
            "ok": True,
            "config_sha256": training.sha256_file(config_path),
            "runtime": {"git_commit": commit},
            "adatad_pretrain": {"sha256": training.sha256_file(pretrain)},
            "selector_initialization": (
                {
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_epoch": 19,
                    "checkpoint_state_key": "state_dict_ema",
                    "detector_state_loaded": False,
                    "optimizer_state_loaded": False,
                    "scheduler_state_loaded": False,
                    "receipt_sha256": "b" * 64,
                }
                if initialized
                else None
            ),
        }
        gate_path = tmp_path / "full_model" / f"{config_path.stem}.json"
        _write_json(gate_path, gate)
        artifacts.append(
            {"path": str(gate_path), "sha256": training.sha256_file(gate_path)}
        )

    suite_path = tmp_path / "gate_suite.json"
    _write_json(
        suite_path,
        {
            "schema": training.BOUNDARY_BURST_GATE_SCHEMA,
            "ok": True,
            "formal_training_unlocked": True,
            "git_commit": commit,
            "artifacts": artifacts,
        },
    )
    monkeypatch.setenv("DUCA_SELECTED_OPT_GATE_SUITE", str(suite_path))
    monkeypatch.setenv(
        "DUCA_SELECTED_OPT_GATE_SUITE_SHA256",
        training.sha256_file(suite_path),
    )

    for variant in SUPPORTED_BOUNDARY_VARIANTS:
        config_path = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
        cfg = Config.fromfile(str(config_path))
        evaluation = cfg.evaluation.to_dict()
        evaluation["ground_truth_filename"] = str(annotation)
        bindings = training.build_runtime_bindings(
            git_commit=commit,
            variant=variant,
            seed=3407,
            slurm_job_id="1",
            source_config_path=config_path,
            source_config_sha256=training.sha256_file(config_path),
            resolved_config_sha256="c" * 64,
            runtime_config_sha256="d" * 64,
            evaluation_annotation_path=annotation,
            evaluation_class_map_path=class_map,
            evaluation_config=evaluation,
            runtime_pretrain_path=pretrain,
            selector_initialization=cfg.workflow.get("selector_initialization", None),
        )
        assert bindings["variant"] == variant
        expected_gate = tmp_path / "full_model" / f"{config_path.stem}.json"
        assert bindings["full_model_gate_sha256"] == training.sha256_file(
            expected_gate
        )


def _terminal_checkpoint_case(tmp_path: Path, monkeypatch, *, r5: bool = False):
    variant = "actionformer_learned_k256_s5801" if r5 else "two_stage_exact_uniform"
    seed = 5801 if r5 else 3407
    protocol = training.R5_FORMAL_PROTOCOL if r5 else training.FORMAL_PROTOCOL
    commit = "a" * 40
    work_dir = tmp_path / "gpu1_id0"
    checkpoint = work_dir / "checkpoint" / "epoch_59.pth"
    audit_path = work_dir / training.DUCA_TRAINING_AUDIT_FILENAME
    config = tmp_path / (
        "actionformer_learned_k256_s5801.py"
        if r5
        else training.VARIANT_CONFIGS[variant]
    )
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    pretrain = tmp_path / "pretrain.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("# config\n", encoding="utf-8")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    pretrain.write_bytes(b"pretrain")
    bindings = {
        "git_commit": commit,
        "variant": variant,
        "seed": seed,
        "slurm_job_id": "7",
        "source_config_path": str(config.resolve()),
        "source_config_sha256": training.sha256_file(config),
        "resolved_config_sha256": "b" * 64,
        "runtime_config_sha256": "c" * 64,
        "pretrain_path": str(pretrain.resolve()),
        "pretrain_sha256": training.sha256_file(pretrain),
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": training.sha256_file(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": training.sha256_file(class_map),
        "evaluation_config_sha256": "f" * 64,
    }
    if r5:
        bindings.update(
            {
                "matrix_summary_path": str((tmp_path / "matrix.json").resolve()),
                "matrix_summary_sha256": "d" * 64,
                "mechanism_gate_path": str((tmp_path / "mechanism.json").resolve()),
                "mechanism_gate_sha256": "e" * 64,
            }
        )
    else:
        bindings.update(
            {
                "gate_suite_sha256": "d" * 64,
                "full_model_gate_sha256": "e" * 64,
            }
        )
    counters = {
        "attempted_batches": 6000,
        "optimizer_attempts": 6000,
        "successful_optimizer_updates": 6000,
        "amp_skipped_attempts": 0,
        "replayed_batches": 0,
        "replay_exhaustions": 0,
        "scheduler_updates": 6000,
        "ema_updates": 6000,
        "duca_schedule_updates": 6000,
        "forced_amp_overflow_attempts": 0,
        "max_amp_retries_observed": 0,
    }
    contract = {
        "formal_protocol": protocol,
        "training_profile": "official60",
        "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
        "primary_checkpoint_epoch": 59,
        "primary_checkpoint_state_key": "state_dict_ema",
        "expected_train_batches_per_epoch": 100,
        "expected_successful_optimizer_updates": 6000,
        "max_amp_retries_per_batch": 3,
    }
    audit = training.build_training_audit(
        contract=contract,
        bindings=bindings,
        epoch=59,
        train_batches_per_epoch=100,
        update_audit=counters,
        epoch_records=[{"epoch": index} for index in range(60)],
        scheduler_last_epoch=6000,
        selector_step=6000,
        scaler_scale=32768.0,
        uses_ema=True,
        complete=True,
    )
    _write_json(audit_path, audit)
    metadata = training.build_checkpoint_metadata(audit)
    sidecar = {
        "schema_version": training.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": training.sha256_file(checkpoint),
        "experiment_metadata": metadata,
    }
    sidecar["sidecar_sha256"] = training.canonical_sha256(sidecar)
    _write_json(Path(f"{checkpoint}.metadata.json"), sidecar)
    monkeypatch.setattr(training, "build_runtime_bindings", lambda **_: bindings)
    return {
        "checkpoint": checkpoint,
        "checkpoint_payload": {
            "epoch": 59,
            "state_dict_ema": {},
            "experiment_metadata": metadata,
        },
        "commit": commit,
        "variant": variant,
        "seed": seed,
        "formal_protocol": protocol,
        "r5_cell": {"seed": seed} if r5 else None,
        "config": config,
        "annotation": annotation,
        "class_map": class_map,
        "pretrain": pretrain,
    }


def test_terminal_checkpoint_binding_validates_complete_training_chain(
    tmp_path: Path, monkeypatch
) -> None:
    case = _terminal_checkpoint_case(tmp_path, monkeypatch)
    identity = training.validate_terminal_checkpoint_binding(
        checkpoint_path=case["checkpoint"],
        checkpoint=case["checkpoint_payload"],
        git_commit=case["commit"],
        variant=case["variant"],
        seed=3407,
        slurm_job_id="7",
        source_config_path=case["config"],
        source_config_sha256=training.sha256_file(case["config"]),
        resolved_config_sha256="b" * 64,
        checkpoint_epoch=59,
        checkpoint_state_key="state_dict_ema",
        evaluation_annotation_path=case["annotation"],
        evaluation_class_map_path=case["class_map"],
        evaluation_config={},
        runtime_pretrain_path=case["pretrain"],
        frozen_pretrain_path=case["pretrain"],
        frozen_pretrain_sha256=training.sha256_file(case["pretrain"]),
    )

    assert identity["variant"] == case["variant"]
    assert identity["successful_optimizer_updates"] == 6000


def test_r5_terminal_checkpoint_binding_returns_r5_evidence_fields(
    tmp_path: Path, monkeypatch
) -> None:
    case = _terminal_checkpoint_case(tmp_path, monkeypatch, r5=True)
    identity = training.validate_terminal_checkpoint_binding(
        checkpoint_path=case["checkpoint"],
        checkpoint=case["checkpoint_payload"],
        git_commit=case["commit"],
        variant=case["variant"],
        seed=case["seed"],
        slurm_job_id="7",
        source_config_path=case["config"],
        source_config_sha256=training.sha256_file(case["config"]),
        resolved_config_sha256="b" * 64,
        checkpoint_epoch=59,
        checkpoint_state_key="state_dict_ema",
        evaluation_annotation_path=case["annotation"],
        evaluation_class_map_path=case["class_map"],
        evaluation_config={},
        runtime_pretrain_path=case["pretrain"],
        frozen_pretrain_path=case["pretrain"],
        frozen_pretrain_sha256=training.sha256_file(case["pretrain"]),
        formal_protocol=case["formal_protocol"],
        r5_cell=case["r5_cell"],
    )

    assert identity["matrix_summary_sha256"] == "d" * 64
    assert identity["mechanism_gate_sha256"] == "e" * 64
    assert "gate_suite_sha256" not in identity


def test_terminal_checkpoint_binding_rejects_resealed_sidecar_drift(
    tmp_path: Path, monkeypatch
) -> None:
    case = _terminal_checkpoint_case(tmp_path, monkeypatch)
    sidecar_path = Path(f"{case['checkpoint']}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = "0" * 64
    sidecar.pop("sidecar_sha256")
    sidecar["sidecar_sha256"] = training.canonical_sha256(sidecar)
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeError, match="checkpoint/sidecar drift"):
        training.validate_terminal_checkpoint_binding(
            checkpoint_path=case["checkpoint"],
            checkpoint=case["checkpoint_payload"],
            git_commit=case["commit"],
            variant=case["variant"],
            seed=3407,
            slurm_job_id="7",
            source_config_path=case["config"],
            source_config_sha256=training.sha256_file(case["config"]),
            resolved_config_sha256="b" * 64,
            checkpoint_epoch=59,
            checkpoint_state_key="state_dict_ema",
            evaluation_annotation_path=case["annotation"],
            evaluation_class_map_path=case["class_map"],
            evaluation_config={},
            runtime_pretrain_path=case["pretrain"],
            frozen_pretrain_path=case["pretrain"],
            frozen_pretrain_sha256=training.sha256_file(case["pretrain"]),
        )


def test_boundary_gate_launcher_names_artifacts_by_config_stem_and_unlocks_training() -> None:
    launcher = (
        ROOT / "scripts" / "run_duca_boundary_burst_gate_gpu1.sh"
    ).read_text(encoding="utf-8")

    assert 'config_stem="$(basename "${config}" .py)"' in launcher
    assert 'full_model/${config_stem}.json' in launcher
    assert '"schema": "duca_boundary_burst_full_model_gate_v1"' in launcher
    assert '"formal_training_unlocked": True' in launcher
