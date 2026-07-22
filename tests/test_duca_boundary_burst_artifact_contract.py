from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata import aggregate_duca_boundary_burst_results as aggregate_module
from tools.bata import duca_selected_axis_training as selected_axis_training
from tools.bata.create_duca_frontend_split import (
    create_split,
    validate_split_manifest,
)
from tools.bata.aggregate_duca_boundary_burst_results import (
    EXPECTED_VARIANTS,
    aggregate,
    validate_suite_self_hash,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    select_variants,
    validate_r0_headroom_summary,
    validate_r0_runtime_bindings,
)
from tools.bata.duca_p0_evaluation import (
    canonical_sha256,
    normalize_evaluation_config,
    official_evaluator_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split(tmp_path: Path) -> tuple[dict, Path]:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    **{
                        f"video_train_{index:03d}": {"subset": "training"}
                        for index in range(10)
                    },
                    "video_test_000": {"subset": "validation"},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = create_split(annotation, tmp_path / "split", seed=3407)
    return payload, tmp_path / "split" / "frontend_split_manifest.json"


def test_split_manifest_reopens_every_sealed_artifact(tmp_path: Path) -> None:
    payload, manifest = _split(tmp_path)
    binding = validate_split_manifest(
        manifest,
        annotation_path=payload["annotation_path"],
        train_block_list=payload["train_block_list"],
        holdout_block_list=payload["holdout_block_list"],
    )

    assert binding["ok"] is True
    assert binding["schema"] == "duca_frontend_train_holdout_split_v2"
    assert binding["annotation_sha256"] == payload["annotation_sha256"]
    assert binding["train_block_list_sha256"] == payload["train_block_list_sha256"]
    assert binding["holdout_block_list_sha256"] == payload["holdout_block_list_sha256"]


@pytest.mark.parametrize(
    "field",
    ("annotation_path", "train_block_list", "holdout_block_list"),
)
def test_split_manifest_fails_closed_on_artifact_content_drift(
    tmp_path: Path,
    field: str,
) -> None:
    payload, manifest = _split(tmp_path)
    Path(payload[field]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash drift"):
        validate_split_manifest(manifest)


def test_split_manifest_fails_closed_on_runtime_path_substitution(tmp_path: Path) -> None:
    payload, manifest = _split(tmp_path)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text(Path(payload["train_block_list"]).read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime path"):
        validate_split_manifest(manifest, train_block_list=replacement)


def test_candidate_selector_reopens_split_reference_hashes(tmp_path: Path) -> None:
    split, manifest = _split(tmp_path)
    Path(split["holdout_block_list"]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash drift"):
        select_variants(
            expected_commit="a" * 40,
            split_manifest=manifest,
            split_manifest_sha256=_sha256(manifest),
            receipt_paths=[],
            output_path=tmp_path / "decision.json",
        )


def test_r0_runtime_reopens_split_reference_hashes(tmp_path: Path) -> None:
    split, manifest = _split(tmp_path)
    checkpoint = tmp_path / "r0.pth"
    pretrain = tmp_path / "pretrain.pth"
    checkpoint.write_bytes(b"checkpoint-v1")
    pretrain.write_bytes(b"pretrain-v1")
    Path(split["annotation_path"]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash drift"):
        validate_r0_runtime_bindings(
            split_manifest=manifest,
            split_manifest_sha256=_sha256(manifest),
            annotation_path=split["annotation_path"],
            annotation_sha256=split["annotation_sha256"],
            train_block_list=split["train_block_list"],
            train_block_list_sha256=split["train_block_list_sha256"],
            holdout_block_list=split["holdout_block_list"],
            holdout_block_list_sha256=split["holdout_block_list_sha256"],
            checkpoint_path=checkpoint,
            checkpoint_sha256=_sha256(checkpoint),
            pretrain_path=pretrain,
            pretrain_sha256=_sha256(pretrain),
        )


@pytest.mark.parametrize("tampered", ("checkpoint", "pretrain"))
def test_r0_runtime_binding_fails_closed_on_weight_drift(
    tmp_path: Path,
    tampered: str,
) -> None:
    split, manifest = _split(tmp_path)
    checkpoint = tmp_path / "r0.pth"
    pretrain = tmp_path / "pretrain.pth"
    checkpoint.write_bytes(b"checkpoint-v1")
    pretrain.write_bytes(b"pretrain-v1")
    expected = {"checkpoint": _sha256(checkpoint), "pretrain": _sha256(pretrain)}
    Path({"checkpoint": checkpoint, "pretrain": pretrain}[tampered]).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="path/hash drift"):
        validate_r0_runtime_bindings(
            split_manifest=manifest,
            split_manifest_sha256=_sha256(manifest),
            annotation_path=split["annotation_path"],
            annotation_sha256=split["annotation_sha256"],
            train_block_list=split["train_block_list"],
            train_block_list_sha256=split["train_block_list_sha256"],
            holdout_block_list=split["holdout_block_list"],
            holdout_block_list_sha256=split["holdout_block_list_sha256"],
            checkpoint_path=checkpoint,
            checkpoint_sha256=expected["checkpoint"],
            pretrain_path=pretrain,
            pretrain_sha256=expected["pretrain"],
        )


def test_p0_rejects_legacy_r0_summary_without_recomputed_identity(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "legacy_r0_summary.json"
    _write_json(
        summary,
        {
            "schema": "duca_r0_selected_axis_boundary_burst_map_v2",
            "ok": True,
            "git_commit": "a" * 40,
        },
    )
    with pytest.raises(RuntimeError):
        validate_r0_headroom_summary(
            summary_path=summary,
            summary_sha256=_sha256(summary),
            expected_commit="a" * 40,
        )


def _terminal_suite(tmp_path: Path) -> tuple[dict, list[Path], list[str]]:
    split, split_manifest = _split(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    pretrain = shared / "pretrain.pth"
    class_map = shared / "class_map.txt"
    pretrain.write_bytes(b"shared-pretrain")
    class_map.write_text("action\n", encoding="utf-8")
    decision = tmp_path / "frontend_decision.json"
    _write_json(
        decision,
        {
            "schema": "duca_boundary_burst_frontend_decision_v1",
            "ok": True,
            "git_commit": "a" * 40,
            "split_manifest_path": str(split_manifest.resolve()),
            "split_manifest_sha256": _sha256(split_manifest),
            "split_binding": validate_split_manifest(
                split_manifest,
                expected_manifest_sha256=_sha256(split_manifest),
            ),
        },
    )
    gate = tmp_path / "gate_suite.json"
    _write_json(gate, {"ok": True, "git_commit": "a" * 40})
    decision_sha = _sha256(decision)
    gate_sha = _sha256(gate)
    completions = []
    completion_shas = []
    for index, variant in enumerate(EXPECTED_VARIANTS):
        root = tmp_path / variant
        config = root / f"{variant}.py"
        annotation = Path(split["annotation_path"])
        prediction = root / "prediction.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"# {variant}\n", encoding="utf-8")
        checkpoint = root / "epoch_59.pth"
        checkpoint.write_bytes(f"checkpoint-{variant}".encode())
        metrics = {"average_mAP": 0.60 + index * 0.01, "mAP@0.5": 0.70}
        _write_json(
            prediction,
            {"metrics": metrics, "result_count": 3, "video_count": 1},
        )
        evaluation_config = normalize_evaluation_config(
            {
                "type": "mAP",
                "ground_truth_filename": str(annotation.resolve()),
                "subset": "validation",
                "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
                "top_k": None,
                "blocked_videos": None,
                "thread": 16,
            }
        )
        bindings = {
            "git_commit": "a" * 40,
            "variant": variant,
            "seed": 3407,
            "source_config_path": str(config.resolve()),
            "source_config_sha256": _sha256(config),
            "resolved_config_sha256": "c" * 64,
            "gate_suite_sha256": gate_sha,
            "full_model_gate_sha256": "d" * 64,
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha256(pretrain),
            "evaluation_config_sha256": canonical_sha256(evaluation_config),
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": _sha256(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": _sha256(class_map),
        }
        contract = {
            "formal_protocol": selected_axis_training.FORMAL_PROTOCOL,
            "training_profile": "official60",
            "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
            "primary_checkpoint_epoch": 59,
            "primary_checkpoint_state_key": "state_dict_ema",
            "expected_train_batches_per_epoch": 100,
            "expected_successful_optimizer_updates": 6000,
            "max_amp_retries_per_batch": 3,
        }
        counters = selected_axis_training.new_update_audit()
        for key in (
            "attempted_batches",
            "optimizer_attempts",
            "successful_optimizer_updates",
            "scheduler_updates",
            "ema_updates",
            "duca_schedule_updates",
        ):
            counters[key] = 6000
        audit = selected_axis_training.build_training_audit(
            contract=contract,
            bindings=bindings,
            epoch=59,
            train_batches_per_epoch=100,
            update_audit=counters,
            epoch_records=[{"epoch": epoch} for epoch in range(60)],
            scheduler_last_epoch=6000,
            selector_step=6000,
            scaler_scale=32768.0,
            uses_ema=True,
            complete=True,
        )
        audit_path = root / "duca_selected_axis_training_audit.json"
        _write_json(audit_path, audit)
        metadata = selected_axis_training.build_checkpoint_metadata(audit)
        sidecar = {
            "schema_version": "duca_p0_checkpoint_sidecar_v2",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "experiment_metadata": metadata,
        }
        sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
        sidecar_path = Path(f"{checkpoint}.metadata.json")
        _write_json(sidecar_path, sidecar)
        identity = {
            "variant": variant,
            "seed": 3407,
            "successful_optimizer_updates": 6000,
            "checkpoint_sidecar_path": str(sidecar_path.resolve()),
            "checkpoint_sidecar_sha256": _sha256(sidecar_path),
            "training_audit_path": str(audit_path.resolve()),
            "training_audit_sha256": _sha256(audit_path),
            "training_audit_self_sha256": audit["audit_sha256"],
            "gate_suite_sha256": gate_sha,
            "full_model_gate_sha256": "d" * 64,
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha256(pretrain),
            "frontend_initialization": None,
        }
        evaluation = root / "evaluation.json"
        evaluation_payload = {
            "schema_version": "duca_selected_axis_terminal_evaluation_v1",
            "git_commit": "a" * 40,
            "task": "offline_temporal_action_detection",
            "seed": 3407,
            "variant": variant,
            "config_path": str(config.resolve()),
            "config_sha256": _sha256(config),
            "resolved_config_sha256": "c" * 64,
            "runtime_config_sha256": "e" * 64,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": 59,
            "checkpoint_state_key": "state_dict_ema",
            "prediction_path": str(prediction.resolve()),
            "prediction_sha256": _sha256(prediction),
            "metrics": metrics,
            "result_count": 3,
            "video_count": 1,
            "evaluator": official_evaluator_identity(),
            "evaluation_config": evaluation_config,
            "evaluation_config_sha256": canonical_sha256(evaluation_config),
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": _sha256(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": _sha256(class_map),
            "training_identity": identity,
        }
        evaluation_payload["evaluation_sha256"] = canonical_sha256(
            evaluation_payload
        )
        _write_json(evaluation, evaluation_payload)
        launch = root / "launch_manifest.json"
        _write_json(
            launch,
            {
                "schema": "duca_two_stage_curriculum_launch_v1",
                "git_commit": "a" * 40,
                "variant": variant,
                "seed": 3407,
                "config_sha256": _sha256(config),
                "gate_suite_sha256": gate_sha,
                "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
                "official_training_successful_updates": 6000,
            },
        )
        completion = root / "completion.json"
        _write_json(
            completion,
            {
                "schema": "duca_two_stage_curriculum_completion_v1",
                "ok": True,
                "git_commit": "a" * 40,
                "variant": variant,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "evaluation_path": str(evaluation.resolve()),
                "evaluation_sha256": _sha256(evaluation),
                "evaluation_self_sha256": evaluation_payload["evaluation_sha256"],
                "prediction_path": str(prediction.resolve()),
                "prediction_sha256": _sha256(prediction),
                "metrics": metrics,
                "training_identity": identity,
                "launch_manifest_path": str(launch.resolve()),
                "launch_manifest_sha256": _sha256(launch),
                "frontend_decision_sha256": decision_sha,
                "gate_suite_sha256": gate_sha,
            },
        )
        completions.append(completion)
        completion_shas.append(_sha256(completion))
    return (
        {"decision": decision, "gate": gate},
        completions,
        completion_shas,
    )


def _stub_official_recompute(monkeypatch) -> None:
    def recompute(path, _evaluation_config):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload

    monkeypatch.setattr(aggregate_module, "recompute_official_map", recompute)


def _reseal_arm_after_identity_mutation(
    completion_path: Path,
    *,
    mutation: str,
) -> str:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    evaluation_path = Path(completion["evaluation_path"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    identity = evaluation["training_identity"]
    audit_path = Path(identity["training_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    root = completion_path.parent

    if mutation == "evaluation_annotation":
        replacement = root / "replacement_annotation.json"
        replacement.write_text(
            Path(evaluation["evaluation_annotation_path"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        evaluation["evaluation_annotation_path"] = str(replacement.resolve())
        evaluation["evaluation_annotation_sha256"] = _sha256(replacement)
        evaluation["evaluation_config"]["ground_truth_filename"] = str(
            replacement.resolve()
        )
        evaluation["evaluation_config_sha256"] = canonical_sha256(
            normalize_evaluation_config(evaluation["evaluation_config"])
        )
        audit["evaluation_annotation_path"] = evaluation["evaluation_annotation_path"]
        audit["evaluation_annotation_sha256"] = evaluation["evaluation_annotation_sha256"]
        audit["evaluation_config_sha256"] = evaluation["evaluation_config_sha256"]
    elif mutation == "class_map":
        replacement = root / "replacement_class_map.txt"
        replacement.write_text("replacement-action\n", encoding="utf-8")
        evaluation["evaluation_class_map_path"] = str(replacement.resolve())
        evaluation["evaluation_class_map_sha256"] = _sha256(replacement)
        audit["evaluation_class_map_path"] = evaluation["evaluation_class_map_path"]
        audit["evaluation_class_map_sha256"] = evaluation["evaluation_class_map_sha256"]
    elif mutation == "evaluation_target":
        replacement = root / "replacement_blocked_videos.json"
        replacement.write_text("[]\n", encoding="utf-8")
        evaluation["evaluation_config"]["blocked_videos"] = str(replacement.resolve())
        evaluation["evaluation_config_sha256"] = canonical_sha256(
            normalize_evaluation_config(evaluation["evaluation_config"])
        )
        audit["evaluation_config_sha256"] = evaluation["evaluation_config_sha256"]
    elif mutation == "adatad_pretrain":
        replacement = root / "replacement_pretrain.pth"
        replacement.write_bytes(b"replacement-pretrain")
        identity["pretrain_path"] = str(replacement.resolve())
        identity["pretrain_sha256"] = _sha256(replacement)
        audit["pretrain_path"] = identity["pretrain_path"]
        audit["pretrain_sha256"] = identity["pretrain_sha256"]
    else:
        raise AssertionError(f"unsupported identity mutation: {mutation}")

    audit.pop("audit_sha256")
    audit["audit_sha256"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    sidecar_path = Path(identity["checkpoint_sidecar_path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["experiment_metadata"] = selected_axis_training.build_checkpoint_metadata(audit)
    sidecar.pop("sidecar_sha256")
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    _write_json(sidecar_path, sidecar)
    identity["training_audit_sha256"] = _sha256(audit_path)
    identity["training_audit_self_sha256"] = audit["audit_sha256"]
    identity["checkpoint_sidecar_sha256"] = _sha256(sidecar_path)
    evaluation["training_identity"] = identity
    evaluation.pop("evaluation_sha256")
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    _write_json(evaluation_path, evaluation)
    completion["evaluation_sha256"] = _sha256(evaluation_path)
    completion["evaluation_self_sha256"] = evaluation["evaluation_sha256"]
    completion["training_identity"] = identity
    _write_json(completion_path, completion)
    return _sha256(completion_path)


def test_terminal_aggregate_rejects_tampered_completion_metric_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    payload = json.loads(completions[0].read_text(encoding="utf-8"))
    payload["metrics"]["average_mAP"] = 0.99
    _write_json(completions[0], payload)
    completion_shas[0] = _sha256(completions[0])

    with pytest.raises(RuntimeError, match="copied completion metrics mismatch"):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=tmp_path / "aggregate.json",
        )


def test_terminal_aggregate_uses_verified_evaluation_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)

    payload = aggregate(
        expected_commit="a" * 40,
        decision_path=roots["decision"],
        decision_sha256=_sha256(roots["decision"]),
        gate_path=roots["gate"],
        gate_sha256=_sha256(roots["gate"]),
        completion_paths=completions,
        completion_sha256s=completion_shas,
        output_path=tmp_path / "aggregate.json",
    )

    assert [row["average_mAP"] for row in payload["results"]] == pytest.approx(
        [0.60, 0.61, 0.62, 0.63]
    )
    written = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert written == payload
    assert validate_suite_self_hash(written) == payload["suite_sha256"]
    assert written["matched_arm_identity"]["frontend_split_annotation"]["sha256"] == (
        written["matched_arm_identity"]["evaluation_annotation"]["sha256"]
    )


@pytest.mark.parametrize("mutation", ("class_map", "evaluation_target", "adatad_pretrain"))
def test_terminal_aggregate_rejects_resealed_cross_arm_identity_drift(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    completion_shas[1] = _reseal_arm_after_identity_mutation(
        completions[1], mutation=mutation
    )

    with pytest.raises(RuntimeError, match="cross-arm identity drift"):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=tmp_path / "aggregate.json",
        )


def test_terminal_aggregate_rejects_resealed_frontend_split_binding_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    decision = json.loads(roots["decision"].read_text(encoding="utf-8"))
    decision["split_binding"]["annotation_sha256"] = "b" * 64
    _write_json(roots["decision"], decision)

    with pytest.raises(RuntimeError, match="frontend split binding mismatch"):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=tmp_path / "aggregate.json",
        )


def test_terminal_aggregate_rejects_resealed_annotation_path_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    completion_shas[1] = _reseal_arm_after_identity_mutation(
        completions[1], mutation="evaluation_annotation"
    )

    with pytest.raises(
        RuntimeError, match="frontend split/evaluation annotation mismatch"
    ):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=tmp_path / "aggregate.json",
        )


def test_terminal_aggregate_replaces_only_after_a_complete_sealed_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    output = tmp_path / "aggregate.json"
    output.write_text('{"partial": true}\n', encoding="utf-8")

    payload = aggregate(
        expected_commit="a" * 40,
        decision_path=roots["decision"],
        decision_sha256=_sha256(roots["decision"]),
        gate_path=roots["gate"],
        gate_sha256=_sha256(roots["gate"]),
        completion_paths=completions,
        completion_sha256s=completion_shas,
        output_path=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".aggregate.json.*.tmp"))

    def interrupted_replace(_source, _destination) -> None:
        raise OSError("simulated replace interruption")

    monkeypatch.setattr(aggregate_module.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated replace interruption"):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=output,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".aggregate.json.*.tmp"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("git_commit", "b" * 40),
        ("variant", "wrong_variant"),
        ("checkpoint_epoch", 58),
        ("checkpoint_state_key", "state_dict"),
        ("config_sha256", "b" * 64),
        ("evaluator", {"module": "wrong"}),
        ("evaluation_annotation_sha256", "b" * 64),
        ("prediction_sha256", "b" * 64),
    ),
)
def test_terminal_aggregate_rejects_resealed_identity_mutation(
    tmp_path: Path,
    monkeypatch,
    field: str,
    replacement,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path)
    completion = json.loads(completions[0].read_text(encoding="utf-8"))
    evaluation_path = Path(completion["evaluation_path"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation[field] = replacement
    evaluation.pop("evaluation_sha256")
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    _write_json(evaluation_path, evaluation)
    completion["evaluation_sha256"] = _sha256(evaluation_path)
    completion["evaluation_self_sha256"] = evaluation["evaluation_sha256"]
    _write_json(completions[0], completion)
    completion_shas[0] = _sha256(completions[0])

    with pytest.raises(RuntimeError):
        aggregate(
            expected_commit="a" * 40,
            decision_path=roots["decision"],
            decision_sha256=_sha256(roots["decision"]),
            gate_path=roots["gate"],
            gate_sha256=_sha256(roots["gate"]),
            completion_paths=completions,
            completion_sha256s=completion_shas,
            output_path=tmp_path / "aggregate.json",
        )


def test_submission_dag_requires_r0_before_every_learned_stage() -> None:
    source = (ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh").read_text(
        encoding="utf-8"
    )

    assert 'p0_dependency="afterok:${r0}"' in source
    assert 'submit_and_record "p0" "${p0_dependency}" "${P0_SBATCH}"' in source
    assert '"p0": "afterok:r0_holdout_map"' in source
    assert '"gate": "afterok:p0"' in source
    assert '"official60_arms": "afterok:gate"' in source
    assert '"r0_positive_headroom_required": True' in source


def test_p0_blocks_nonpositive_r0_headroom_before_training() -> None:
    source = (ROOT / "scripts" / "run_duca_boundary_burst_p0_gpu1.sh").read_text(
        encoding="utf-8"
    )

    headroom_gate = source.index("validate_r0_headroom_summary")
    real_gate = source.index("run_duca_frontend_p0_real_gate.py")
    first_variant = source.index("run_duca_frontend_pretrain_variant_gpu1.sh")
    assert headroom_gate < real_gate < first_variant
    assert "validate_r0_headroom_summary" in source
    assert "R0_SUMMARY_SHA256_FILE" in source


def test_artifact_consumers_use_upstream_sha256_seals() -> None:
    source = (ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh").read_text(
        encoding="utf-8"
    )

    assert 'frontend_decision.sha256' in source
    assert 'gate_suite.sha256' in source
    assert 'completion.sha256' in source
    assert 'sha256sum "${DUCA_FRONTEND_DECISION_JSON}"' not in source
    assert 'sha256sum "${DUCA_SELECTED_OPT_GATE_SUITE}"' not in source
    assert "--completion-sha256" in source


def test_submit_prefreezes_all_r0_weight_and_split_bindings() -> None:
    submit = (ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh").read_text(
        encoding="utf-8"
    )
    r0 = (ROOT / "scripts" / "run_duca_boundary_burst_r0_holdout_map_gpu1.sh").read_text(
        encoding="utf-8"
    )

    for name in (
        "R0_CHECKPOINT_SHA256",
        "ADATAD_PRETRAIN_SHA256",
        "SPLIT_ANNOTATION_SHA256",
        "SPLIT_TRAIN_BLOCK_LIST_SHA256",
        "SPLIT_HOLDOUT_BLOCK_LIST_SHA256",
    ):
        assert name in submit
    assert "validate_r0_runtime_bindings" in r0
    assert "DUCA_FRONTEND_SPLIT_MANIFEST" in r0


def test_uniform_arm_never_claims_a_gaussian_frontend_checkpoint() -> None:
    source = (ROOT / "scripts" / "run_duca_two_stage_curriculum_variant_gpu1.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [[ "${VARIANT}" == "two_stage_exact_uniform" ]]' in source
    assert "unset DUCA_FRONTEND_CHECKPOINT" in source
    assert 'FRONTEND_BINDING="not_applicable_exact_uniform"' in source
    assert "FRONTEND_CHECKPOINT_SHA256_JSON=null" in source
    assert '"two_stage_exact_uniform": "gaussian_matched"' not in source
    assert '"frontend_checkpoint_sha256": ${FRONTEND_CHECKPOINT_SHA256_JSON}' in source


def test_all_split_consumers_reopen_the_shared_hash_contract() -> None:
    paths = (
        "scripts/run_duca_boundary_burst_p0_gpu1.sh",
        "scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh",
        "scripts/run_duca_frontend_pretrain_variant_gpu1.sh",
        "tools/bata/run_duca_frontend_p0_real_gate.py",
        "tools/bata/select_duca_boundary_burst_candidates.py",
    )
    for relative in paths:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "validate_split_manifest" in source or "--validate-manifest" in source
