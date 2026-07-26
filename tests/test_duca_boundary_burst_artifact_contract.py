from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata import aggregate_duca_boundary_burst_results as aggregate_module
from tools.bata import duca_selected_axis_training as selected_axis_training
from tools.bata import select_duca_boundary_burst_candidates as selector_module
from tools.bata.create_duca_frontend_split import (
    create_split,
    validate_split_manifest,
)
from tools.bata.aggregate_duca_boundary_burst_results import (
    aggregate,
    validate_suite_self_hash,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    FULL_MODEL_ARTIFACT_SCHEMA,
    FULL_MODEL_GATE_SCHEMA,
    PREREGISTERED_PROJECTED_FAMILY,
    R0_PROJECTED_FAMILY_ROUTES,
    UNIFORM_OFFICIAL_VARIANT,
    _family_routing_contract,
    _normalized_lf_sha256,
    create_family_routing_manifest,
    create_p0_training_asformer_consumer_receipt,
    select_variants,
    validate_family_routing_manifest,
    validate_p0_real_gate,
    validate_p0_training_asformer_consumer_receipt,
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
            preregistered_family=PREREGISTERED_PROJECTED_FAMILY,
            family_manifest_path=tmp_path / "missing_family_manifest.json",
            family_manifest_sha256="0" * 64,
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


@pytest.mark.parametrize(
    ("selected_family", "p0_variant", "official_variant"),
    (
        (
            "R2Q3_privileged_boundary_burst",
            "burst_r2q3",
            "boundary_burst_r2q3_g0",
        ),
        (
            "R4Q5_privileged_boundary_burst",
            "burst_r4q5",
            "boundary_burst_r4q5_g0",
        ),
    ),
)
def test_r0_selected_family_has_one_exact_p0_and_official_route(
    selected_family: str,
    p0_variant: str,
    official_variant: str,
) -> None:
    routing = _family_routing_contract(selected_family)

    assert routing["selected_p0_variant"] == p0_variant
    assert routing["required_p0_variants"] == [p0_variant]
    assert routing["selected_official60_variant"] == official_variant
    assert routing["required_official60_variants"] == [
        "two_stage_exact_uniform",
        official_variant,
    ]
    assert "gaussian_matched" in routing["diagnostic_p0_variants"]
    assert "gaussian_matched_g0" in routing["diagnostic_official60_variants"]
    assert routing["simple_delta_role"] == "no_training_same_feasible_control"


def test_family_manifest_binds_distinct_r0_producer_and_p0_commits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    producer_commit = "a" * 40
    consumer_commit = "b" * 40
    summary = tmp_path / "r0_summary.json"
    _write_json(
        summary,
        {
            "schema": "focused_r0_summary_v1",
            "git_commit": producer_commit,
            "selected_weakest_projected_family": "R4Q5_privileged_boundary_burst",
        },
    )

    def replay_focused_r0(
        *, summary_path: str | Path, summary_sha256: str, expected_commit: str
    ) -> dict:
        path = Path(summary_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _sha256(path) != summary_sha256 or payload["git_commit"] != expected_commit:
            raise RuntimeError("sealed R0 producer drift")
        return {
            "schema": "duca_r0_headroom_gate_v2",
            "ok": True,
            "git_commit": expected_commit,
            "r0_summary_path": str(path),
            "r0_summary_sha256": summary_sha256,
            "selected_weakest_projected_family": payload[
                "selected_weakest_projected_family"
            ],
            "eligible_projected_families": [
                payload["selected_weakest_projected_family"]
            ],
            "test_subset_consumed": False,
        }

    monkeypatch.setattr(
        selector_module, "validate_r0_headroom_summary", replay_focused_r0
    )
    manifest_path = tmp_path / "family_routing_manifest.json"
    create_family_routing_manifest(
        summary_path=summary,
        summary_sha256=_sha256(summary),
        expected_commit=consumer_commit,
        preregistered_family=PREREGISTERED_PROJECTED_FAMILY,
        r0_expected_commit=producer_commit,
        output_path=manifest_path,
    )
    manifest = validate_family_routing_manifest(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        expected_commit=consumer_commit,
    )

    assert manifest["git_commit"] == consumer_commit
    assert manifest["r0_producer_commit"] == producer_commit
    assert manifest["r0_headroom_gate"]["git_commit"] == producer_commit
    assert (
        manifest["r0_diagnostic_provenance"][
            "reported_selected_weakest_projected_family"
        ]
        == "R4Q5_privileged_boundary_burst"
    )
    assert manifest["r0_diagnostic_provenance"]["routing_authority"] is False
    assert (
        manifest["family_routing"]["preregistered_projected_family"]
        == PREREGISTERED_PROJECTED_FAMILY
    )
    assert manifest["family_routing"]["selected_p0_variant"] == "burst_r2q3"
    assert (
        manifest["family_routing"]["selected_official60_variant"]
        == "boundary_burst_r2q3_g0"
    )


@pytest.mark.parametrize("wrong_family", ("Gaussian", "R2Q3", "burst_r4q5"))
def test_family_routing_rejects_wrong_or_unprojected_family(
    wrong_family: str,
) -> None:
    with pytest.raises(RuntimeError, match="unsupported projected family"):
        _family_routing_contract(wrong_family)


def _terminal_suite(
    tmp_path: Path,
    monkeypatch,
    *,
    selected_family: str = "R2Q3_privileged_boundary_burst",
) -> tuple[dict, list[Path], list[str]]:
    commit = "a" * 40
    split, split_manifest = _split(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    pretrain = shared / "pretrain.pth"
    class_map = shared / "class_map.txt"
    pretrain.write_bytes(b"shared-pretrain")
    class_map.write_text("action\n", encoding="utf-8")

    r0_summary = tmp_path / "r0_summary.json"
    _write_json(
        r0_summary,
        {
            "schema": "focused_r0_summary_v1",
            "git_commit": commit,
            "selected_weakest_projected_family": selected_family,
        },
    )

    def replay_focused_r0(
        *, summary_path: str | Path, summary_sha256: str, expected_commit: str
    ) -> dict:
        path = Path(summary_path).resolve()
        if not path.is_file() or _sha256(path) != summary_sha256:
            raise RuntimeError("sealed R0 summary drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("git_commit") != expected_commit:
            raise RuntimeError("sealed R0 summary commit drift")
        selected = payload.get("selected_weakest_projected_family")
        _family_routing_contract(selected)
        return {
            "schema": "duca_r0_headroom_gate_v2",
            "ok": True,
            "git_commit": expected_commit,
            "r0_summary_path": str(path),
            "r0_summary_sha256": summary_sha256,
            "selected_weakest_projected_family": selected,
            "eligible_projected_families": [selected],
            "positive_headroom_required": True,
            "test_subset_consumed": False,
        }

    monkeypatch.setattr(
        selector_module, "validate_r0_headroom_summary", replay_focused_r0
    )
    family_manifest_path = tmp_path / "family_routing_manifest.json"
    create_family_routing_manifest(
        summary_path=r0_summary,
        summary_sha256=_sha256(r0_summary),
        expected_commit=commit,
        preregistered_family=PREREGISTERED_PROJECTED_FAMILY,
        output_path=family_manifest_path,
    )
    family_manifest = validate_family_routing_manifest(
        manifest_path=family_manifest_path,
        manifest_sha256=_sha256(family_manifest_path),
        expected_commit=commit,
    )
    routing = family_manifest["family_routing"]

    source = shared / "official" / "ASFormer" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"class ASFormer:\r\n    pass\r\n")
    normalized_source_sha = _normalized_lf_sha256(source)
    official_asformer = {
        "path": str(source.resolve()),
        "sha256": _sha256(source),
        "normalized_lf_sha256": normalized_source_sha,
        "config_declared_normalized_lf_sha256": normalized_source_sha,
    }
    p0_config = (ROOT / routing["selected_p0_config"]).resolve()
    p0_gate = tmp_path / "p0_real_gate.json"
    _write_json(
        p0_gate,
        {
            "schema": "duca_frontend_p0_real_cuda_gate_v1",
            "ok": True,
            "fail_closed": True,
            "git_binding": {"git_commit": commit},
            "final_git_binding": {"git_commit": commit},
            "config_path": str(p0_config),
            "config_sha256": _sha256(p0_config),
            "assets": {
                "videomae_checkpoint": {
                    "path": str(pretrain.resolve()),
                    "sha256": _sha256(pretrain),
                },
                "official_asformer_source": official_asformer,
            },
        },
    )
    p0_gate_binding = validate_p0_real_gate(
        gate_path=p0_gate,
        gate_sha256=_sha256(p0_gate),
        expected_commit=commit,
    )
    p0_consumer_path = tmp_path / "p0_asformer_consumer.json"
    create_p0_training_asformer_consumer_receipt(
        gate_path=p0_gate,
        gate_sha256=_sha256(p0_gate),
        expected_commit=commit,
        selected_config_path=p0_config,
        output_path=p0_consumer_path,
    )
    p0_consumer = validate_p0_training_asformer_consumer_receipt(
        receipt_path=p0_consumer_path,
        receipt_sha256=_sha256(p0_consumer_path),
        expected_commit=commit,
        expected_p0_gate=p0_gate_binding,
        expected_config_path=p0_config,
    )

    p0_winner_root = tmp_path / "selected_p0_winner"
    p0_winner_root.mkdir()
    p0_checkpoint = p0_winner_root / "epoch_4.pth"
    p0_summary = p0_winner_root / "selection_quality_summary.json"
    p0_records = p0_winner_root / "selection_quality_records.jsonl"
    p0_checkpoint.write_bytes(b"selected-p0-checkpoint")
    _write_json(p0_summary, {"ok": True})
    p0_records.write_text('{"ok": true}\n', encoding="utf-8")
    selected_p0 = routing["selected_p0_variant"]
    winner = {
        "variant": selected_p0,
        "epoch_one_based": 5,
        "checkpoint_path": str(p0_checkpoint.resolve()),
        "checkpoint_sha256": _sha256(p0_checkpoint),
        "summary_path": str(p0_summary.resolve()),
        "summary_sha256": _sha256(p0_summary),
        "records_path": str(p0_records.resolve()),
        "records_sha256": _sha256(p0_records),
        "all_sanity_gates_pass": True,
    }
    decision = tmp_path / "frontend_decision.json"
    _write_json(
        decision,
        {
            "schema": "duca_boundary_burst_frontend_decision_v2",
            "ok": True,
            "fail_closed": True,
            "status": "GO_TO_PREREGISTERED_R2Q3_P0_AND_MATCHED_OFFICIAL60",
            "git_commit": commit,
            "test_subset_consumed": False,
            "split_manifest_path": str(split_manifest.resolve()),
            "split_manifest_sha256": _sha256(split_manifest),
            "split_binding": validate_split_manifest(
                split_manifest,
                expected_manifest_sha256=_sha256(split_manifest),
            ),
            "family_manifest": family_manifest,
            "preregistered_projected_family": family_manifest[
                "preregistered_projected_family"
            ],
            "r0_diagnostic_provenance": family_manifest[
                "r0_diagnostic_provenance"
            ],
            "r0_headroom_gate": family_manifest["r0_headroom_gate"],
            "family_routing": routing,
            "continuation_rule": routing["continuation_rule"],
            "p0_real_gate": p0_gate_binding,
            "p0_training_asformer_consumer": p0_consumer,
            "winners": {selected_p0: winner},
            "candidates": {
                variant: ([dict(winner) for _ in range(4)] if variant == selected_p0 else [])
                for variant in selector_module.VARIANT_SPECS
            },
            "diagnostic_failures_block_main": False,
        },
    )
    decision_sha = _sha256(decision)

    required_configs = {
        UNIFORM_OFFICIAL_VARIANT: routing["uniform_official60_config"],
        routing["selected_official60_variant"]: routing[
            "selected_official60_config"
        ],
    }
    gate_artifacts = []
    for variant, relative_config in required_configs.items():
        config = (ROOT / relative_config).resolve()
        initialization = None
        if variant != UNIFORM_OFFICIAL_VARIANT:
            initialization = {
                "schema": "duca_frontend_initialization_v1",
                "checkpoint_path": winner["checkpoint_path"],
                "checkpoint_sha256": winner["checkpoint_sha256"],
                "checkpoint_epoch": winner["epoch_one_based"] - 1,
                "checkpoint_state_key": "state_dict_ema",
                "loaded_selector_state_count": 1,
                "reset_state_keys": [],
                "detector_state_loaded": False,
                "optimizer_state_loaded": False,
                "scheduler_state_loaded": False,
            }
            initialization["receipt_sha256"] = canonical_sha256(initialization)
        artifact = tmp_path / "full_model" / f"{config.stem}.json"
        _write_json(
            artifact,
            {
                "schema": FULL_MODEL_ARTIFACT_SCHEMA,
                "ok": True,
                "status": "p1_p2_exact_full_model_amp_ddp_gate_passed",
                "runtime": {"git_commit": commit},
                "config_contract": {
                    "ok": True,
                    "task": "offline_temporal_action_detection",
                    "config": str(config),
                },
                "config_sha256": _sha256(config),
                "real_thumos_loader_executed": True,
                "optimizer_exact_coverage": True,
                "gt_boundary_validity": {
                    "batch_size": 1,
                    "endpoint_count": 2,
                    "valid_endpoint_count": 2,
                },
                "adatad_pretrain": p0_gate_binding["adatad_pretrain"],
                "official_asformer_source": p0_consumer[
                    "official_asformer_source"
                ],
                "selector_initialization": initialization,
            },
        )
        gate_artifacts.append(
            {"path": str(artifact.resolve()), "sha256": _sha256(artifact)}
        )
    gate = tmp_path / "gate_suite.json"
    _write_json(
        gate,
        {
            "schema": FULL_MODEL_GATE_SCHEMA,
            "ok": True,
            "fail_closed": True,
            "formal_training_unlocked": True,
            "status": "matched_u_selected_g0_full_model_gate_passed",
            "git_commit": commit,
            "frontend_decision_path": str(decision.resolve()),
            "frontend_decision_sha256": decision_sha,
            "gated_variants": routing["required_official60_variants"],
            "required_official60_variants": routing[
                "required_official60_variants"
            ],
            "family_manifest": family_manifest,
            "r0_headroom_gate": family_manifest["r0_headroom_gate"],
            "family_routing": routing,
            "p0_real_gate": p0_gate_binding,
            "p0_training_asformer_consumer": p0_consumer,
            "artifacts": gate_artifacts,
        },
    )
    gate_sha = _sha256(gate)
    completions = []
    completion_shas = []
    for index, variant in enumerate(routing["required_official60_variants"]):
        root = tmp_path / variant
        config = (ROOT / required_configs[variant]).resolve()
        annotation = Path(split["annotation_path"])
        prediction = root / "prediction.json"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "epoch_59.pth"
        checkpoint.write_bytes(f"checkpoint-{variant}".encode())
        metrics = {"average_mAP": 0.60 + index * 0.01, "mAP@0.5": 0.70}
        frontend_initialization = (
            None
            if variant == UNIFORM_OFFICIAL_VARIANT
            else {
                "checkpoint_path": winner["checkpoint_path"],
                "checkpoint_sha256": winner["checkpoint_sha256"],
                "checkpoint_epoch": winner["epoch_one_based"] - 1,
                "checkpoint_state_key": "state_dict_ema",
                "reset_state_keys": [],
                "gate_receipt_sha256": "f" * 64,
            }
        )
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
            "git_commit": commit,
            "variant": variant,
            "seed": 3407,
            "source_config_path": str(config.resolve()),
            "source_config_sha256": _sha256(config),
            "resolved_config_sha256": "c" * 64,
            "gate_suite_sha256": gate_sha,
            "full_model_gate_sha256": gate_sha,
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha256(pretrain),
            "evaluation_config_sha256": canonical_sha256(evaluation_config),
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": _sha256(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": _sha256(class_map),
            "selector_initialization_contract": frontend_initialization,
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
            "full_model_gate_sha256": gate_sha,
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha256(pretrain),
            "frontend_initialization": frontend_initialization,
        }
        evaluation = root / "evaluation.json"
        evaluation_payload = {
            "schema_version": "duca_selected_axis_terminal_evaluation_v1",
            "git_commit": commit,
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
                "fail_closed": True,
                "git_commit": commit,
                "variant": variant,
                "execution_role": "required_main",
                "seed": 3407,
                "config_sha256": _sha256(config),
                "frontend_decision_path": str(decision.resolve()),
                "frontend_decision_sha256": decision_sha,
                "family_manifest": family_manifest,
                "r0_headroom_gate": family_manifest["r0_headroom_gate"],
                "family_routing": routing,
                "p0_training_asformer_consumer": p0_consumer,
                "gate_path": str(gate.resolve()),
                "gate_suite_sha256": gate_sha,
                "frontend_checkpoint_binding": (
                    "not_applicable_exact_uniform"
                    if variant == UNIFORM_OFFICIAL_VARIANT
                    else "variant_matched_p0_winner"
                ),
                "frontend_checkpoint_sha256": (
                    None
                    if variant == UNIFORM_OFFICIAL_VARIANT
                    else winner["checkpoint_sha256"]
                ),
                "frontend_checkpoint_epoch_zero_based": (
                    None
                    if variant == UNIFORM_OFFICIAL_VARIANT
                    else winner["epoch_one_based"] - 1
                ),
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
                "fail_closed": True,
                "git_commit": commit,
                "variant": variant,
                "execution_role": "required_main",
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
                "frontend_decision_path": str(decision.resolve()),
                "frontend_decision_sha256": decision_sha,
                "family_manifest": family_manifest,
                "r0_headroom_gate": family_manifest["r0_headroom_gate"],
                "family_routing": routing,
                "p0_training_asformer_consumer": p0_consumer,
                "gate_path": str(gate.resolve()),
                "gate_suite_sha256": gate_sha,
            },
        )
        completions.append(completion)
        completion_shas.append(_sha256(completion))
    return (
        {
            "decision": decision,
            "gate": gate,
            "r0_summary": r0_summary,
            "family_manifest": family_manifest_path,
            "p0_consumer": p0_consumer_path,
        },
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
    replacement_pretrain_path: Path | None = None,
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
        replacement = replacement_pretrain_path or root / "replacement_pretrain.pth"
        if replacement_pretrain_path is None:
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
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
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


@pytest.mark.parametrize(
    "r0_reported_family",
    (
        "R2Q3_privileged_boundary_burst",
        "R4Q5_privileged_boundary_burst",
    ),
)
def test_terminal_aggregate_keeps_preregistered_r2q3_when_r0_report_differs(
    tmp_path: Path,
    monkeypatch,
    r0_reported_family: str,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(
        tmp_path, monkeypatch, selected_family=r0_reported_family
    )

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
        [0.60, 0.61]
    )
    assert [row["variant"] for row in payload["results"]] == [
        "two_stage_exact_uniform",
        "boundary_burst_r2q3_g0",
    ]
    assert (
        payload["family_routing"]["preregistered_projected_family"]
        == PREREGISTERED_PROJECTED_FAMILY
    )
    assert (
        payload["r0_headroom_gate"]["selected_weakest_projected_family"]
        == r0_reported_family
    )
    written = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert written == payload
    assert validate_suite_self_hash(written) == payload["suite_sha256"]
    assert written["matched_arm_identity"]["frontend_split_annotation"]["sha256"] == (
        written["matched_arm_identity"]["evaluation_annotation"]["sha256"]
    )


def test_missing_and_failed_diagnostics_do_not_block_main_aggregate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    failed_gaussian = tmp_path / "official60" / "gaussian_matched_g0" / "diagnostic_status.json"
    _write_json(failed_gaussian, {"ok": False, "status": "diagnostic_failed"})

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

    assert payload["diagnostic_failures_block_main"] is False
    assert "gaussian_matched_g0" in payload["diagnostic_official60_variants"]
    assert len(payload["results"]) == 2


def test_terminal_aggregate_rejects_sealed_r0_summary_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    r0_summary = json.loads(roots["r0_summary"].read_text(encoding="utf-8"))
    r0_summary["selected_weakest_projected_family"] = (
        "R4Q5_privileged_boundary_burst"
    )
    _write_json(roots["r0_summary"], r0_summary)

    with pytest.raises(RuntimeError, match="sealed R0 summary drift"):
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


@pytest.mark.parametrize("reseal_gate_index", (False, True))
def test_full_model_gate_reopens_artifact_hash_and_pass_content(
    tmp_path: Path,
    monkeypatch,
    reseal_gate_index: bool,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    gate_payload = json.loads(roots["gate"].read_text(encoding="utf-8"))
    artifact_record = gate_payload["artifacts"][0]
    artifact_path = Path(artifact_record["path"])
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_payload["optimizer_exact_coverage"] = False
    _write_json(artifact_path, artifact_payload)
    expected_error = "path/hash drift"
    if reseal_gate_index:
        artifact_record["sha256"] = _sha256(artifact_path)
        _write_json(roots["gate"], gate_payload)
        expected_error = "did not pass"

    with pytest.raises(RuntimeError, match=expected_error):
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


def test_terminal_aggregate_rejects_unselected_family_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    completion = json.loads(completions[1].read_text(encoding="utf-8"))
    completion["variant"] = "boundary_burst_r4q5_g0"
    _write_json(completions[1], completion)
    completion_shas[1] = _sha256(completions[1])

    with pytest.raises(RuntimeError, match="evaluation run identity mismatch"):
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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("class_map", "cross-arm identity drift"),
        ("evaluation_target", "cross-arm identity drift"),
        (
            "adatad_pretrain",
            "terminal AdaTAD pretrain differs from sealed P0/full-model gate",
        ),
    ),
)
def test_terminal_aggregate_rejects_resealed_cross_arm_identity_drift(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    expected_error: str,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    completion_shas[1] = _reseal_arm_after_identity_mutation(
        completions[1], mutation=mutation
    )

    with pytest.raises(RuntimeError, match=expected_error):
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


def test_terminal_aggregate_rejects_same_content_different_path_pretrain_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    first_completion = json.loads(completions[0].read_text(encoding="utf-8"))
    first_evaluation = json.loads(
        Path(first_completion["evaluation_path"]).read_text(encoding="utf-8")
    )
    sealed_pretrain = Path(
        first_evaluation["training_identity"]["pretrain_path"]
    )
    relocated_pretrain = tmp_path / "relocated_same_content_pretrain.pth"
    relocated_pretrain.write_bytes(sealed_pretrain.read_bytes())
    for index, completion in enumerate(completions):
        completion_shas[index] = _reseal_arm_after_identity_mutation(
            completion,
            mutation="adatad_pretrain",
            replacement_pretrain_path=relocated_pretrain,
        )

    with pytest.raises(
        RuntimeError,
        match="terminal AdaTAD pretrain differs from sealed P0/full-model gate",
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


def test_terminal_aggregate_rejects_resealed_frontend_split_binding_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_official_recompute(monkeypatch)
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
    decision = json.loads(roots["decision"].read_text(encoding="utf-8"))
    decision["split_binding"]["annotation_sha256"] = "b" * 64
    _write_json(roots["decision"], decision)

    with pytest.raises(RuntimeError, match="boundary-burst decision split drift"):
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
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
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
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
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
    roots, completions, completion_shas = _terminal_suite(tmp_path, monkeypatch)
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
    assert '"two_stage_exact_uniform": "afterok:gate"' in source
    assert '"r0_selected_boundary_burst_g0": "afterok:gate"' in source
    assert '"aggregate": "afterok:matched_u_plus_r0_selected_g0"' in source
    assert '"aggregate_inputs": "matched_u_plus_r0_selected_g0_only"' in source
    assert '"diagnostic_failures_block_main": False' in source
    assert '"diagnostic_submission_policy": "not_submitted_by_main_dag"' in source
    assert '"diagnostic_executed": False' not in source
    assert 'role_status=\\$?' in source
    assert '"required_main_failure_blocks_final_aggregate": True' in source
    assert '"worker_failed_aggregate_will_adjudicate_required_main"' in source
    assert 'selected_variant="$(${PYTHON}' not in source
    assert 'selected_variant="$("${PYTHON}"' in source
    assert 'elif worker_role == "r0_selected_boundary_burst_g0":' in source
    assert 'variant = decision["family_routing"]["selected_official60_variant"]' in source
    assert 'submit_and_record "two_stage_exact_uniform" "${main_dependency}"' in source
    assert 'submit_and_record "r0_selected_boundary_burst_g0" "${main_dependency}"' in source
    assert 'aggregate_dependency="afterok:${uniform}:${selected_g0}"' in source
    assert 'submit_and_record "gaussian_matched_g0"' not in source
    assert 'submit_and_record "boundary_burst_r2q3_g0"' not in source
    assert 'submit_and_record "boundary_burst_r4q5_g0"' not in source
    assert '"r0_selected_boundary_burst_g0": "r0_selected_boundary_burst_g0.sbatch"' in source
    assert '"r0_positive_headroom_required": True' in source


def test_p0_blocks_nonpositive_r0_headroom_before_training() -> None:
    source = (ROOT / "scripts" / "run_duca_boundary_burst_p0_gpu1.sh").read_text(
        encoding="utf-8"
    )

    headroom_gate = source.index("validate_r0_headroom_summary")
    real_gate = source.index("run_duca_frontend_p0_real_gate.py")
    asformer_consumer = source.index(
        "create_p0_training_asformer_consumer_receipt"
    )
    first_variant = source.index("run_duca_frontend_pretrain_variant_gpu1.sh")
    assert headroom_gate < real_gate < asformer_consumer < first_variant
    assert "validate_r0_headroom_summary" in source
    assert "create_family_routing_manifest" in source
    assert "PREREGISTERED_FAMILY" in source
    assert "preregistered_family=sys.argv[5]" in source
    assert '--preregistered-family "${PREREGISTERED_FAMILY}"' in source
    assert "R0_SUMMARY_SHA256_FILE" in source
    assert 'R0_PRODUCER_COMMIT="${DUCA_R0_PRODUCER_COMMIT:-${EXPECTED_COMMIT}}"' in source
    assert "r0_expected_commit=sys.argv[4]" in source


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
    assert 'FRONTEND_CHECKPOINT_SHA256_VALUE=""' in source
    assert '"two_stage_exact_uniform": "gaussian_matched"' not in source
    assert '"frontend_checkpoint_sha256": frontend_checkpoint_sha256 or None' in source


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
