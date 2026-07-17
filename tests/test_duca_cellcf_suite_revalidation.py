from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tools.bata.validate_duca_cellcf_suite as suite
from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    build_profile_summary,
)
from tools.bata.profile_duca_full_stack_cost import load_cellcf_cost_binding
from tools.bata.summarize_duca_cellcf_cost import summarize


TREE_BINDING = {
    "trained_opentad_tree_oid": "1" * 40,
    "evidence_opentad_tree_oid": "1" * 40,
    "trained_adatad_thumos_config_tree_oid": "2" * 40,
    "evidence_adatad_thumos_config_tree_oid": "2" * 40,
    "model_and_config_trees_equal": True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_evidence(path: Path, payload: dict[str, Any]) -> None:
    payload.pop("artifact_chain_sha256", None)
    payload["artifact_chain_sha256"] = suite._canonical_sha256(payload)
    _write_json(path, payload)


def _post_run_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = repo_root / "config.py"
    pretrain = tmp_path / "videomae.pth"
    gate = tmp_path / "gate.json"
    pilot = tmp_path / "pilot.json"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "category_idx.txt"
    evaluator_source = tmp_path / "mAP.py"
    checkpoint = tmp_path / "epoch_131.pth"
    prediction = tmp_path / "result_detection.json"
    config.write_text("model = {}\n", encoding="utf-8")
    pretrain.write_bytes(b"pretrain")
    _write_json(gate, {"ok": True})
    _write_json(pilot, {"ok": True})
    _write_json(annotation, {"database": {}})
    class_map.write_text("1 Action\n", encoding="utf-8")
    evaluator_source.write_text("class mAP:\n    pass\n", encoding="utf-8")
    checkpoint.write_bytes(b"terminal-checkpoint")
    _write_json(prediction, {"results": {"video": [{"score": 0.5}]}})

    commit = "a" * 40
    seed = 7
    resolved_config_sha256 = "b" * 64
    protocol_sha256 = "c" * 64
    order_sha256 = "d" * 64
    evaluation_config = {
        "ground_truth_filename": str(annotation.resolve()),
        "blocked_videos": None,
    }
    evaluation_config_sha256 = suite._canonical_sha256(evaluation_config)
    evaluator = {
        "module": "opentad.evaluations.mAP",
        "class_name": "mAP",
        "source_path": str(evaluator_source.resolve()),
        "source_sha256": _sha256(evaluator_source),
    }
    audit = {
        "variant": "uniform",
        "git_commit": commit,
        "seed": seed,
        "source_config_path": str(config.resolve()),
        "source_config_sha256": _sha256(config),
        "runtime_pretrain_path": str(pretrain.resolve()),
        "runtime_pretrain_sha256": _sha256(pretrain),
        "real_loader_gate_json": str(gate.resolve()),
        "real_loader_gate_sha256": _sha256(gate),
        "ddp_pilot_json": str(pilot.resolve()),
        "ddp_pilot_sha256": _sha256(pilot),
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": _sha256(class_map),
    }
    audit_path = tmp_path / "training_audit.json"
    _write_json(audit_path, audit)
    sidecar = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "experiment_metadata": {"training_audit": audit},
    }
    sidecar_path = tmp_path / "epoch_131.pth.metadata.json"
    _write_json(sidecar_path, sidecar)
    evaluation = {
        "git_commit": commit,
        "config_path": str(config.resolve()),
        "config_sha256": _sha256(config),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": suite.TERMINAL_EPOCH,
        "checkpoint_state_key": suite.TERMINAL_STATE_KEY,
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": _sha256(prediction),
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config": evaluation_config,
        "evaluator": evaluator,
    }
    evaluation_path = tmp_path / "terminal_evaluation.json"
    _write_json(evaluation_path, evaluation)
    manifest = {
        "variant": "uniform",
        "git_commit": commit,
        "seed": seed,
        "config": "config.py",
        "config_sha256": _sha256(config),
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    metrics = {
        "average_mAP": 0.5,
        "mAP@0.3": 0.7,
        "mAP@0.4": 0.6,
        "mAP@0.5": 0.5,
        "mAP@0.6": 0.4,
        "mAP@0.7": 0.3,
    }
    payload = {
        "schema": suite.POST_RUN_SCHEMA,
        "ok": True,
        "variant": "uniform",
        "git_commit": commit,
        "seed": seed,
        "config_sha256": _sha256(config),
        "resolved_config_sha256": resolved_config_sha256,
        "protocol_sha256": protocol_sha256,
        "ordered_exposure_sha256": order_sha256,
        "real_loader_gate_sha256": _sha256(gate),
        "ddp_pilot_sha256": _sha256(pilot),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config_sha256": evaluation_config_sha256,
        "successful_optimizer_updates": suite.EXPECTED_UPDATES,
        "checkpoint_epoch": suite.TERMINAL_EPOCH,
        "checkpoint_state_key": suite.TERMINAL_STATE_KEY,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_sidecar_path": str(sidecar_path.resolve()),
        "checkpoint_sidecar_sha256": _sha256(sidecar_path),
        "checkpoint_payload_contract": {
            "payload_reopened": True,
            "epoch": suite.TERMINAL_EPOCH,
        },
        "training_audit_path": str(audit_path.resolve()),
        "training_audit_sha256": _sha256(audit_path),
        "terminal_evaluation_path": str(evaluation_path.resolve()),
        "terminal_evaluation_sha256": _sha256(evaluation_path),
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": _sha256(prediction),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": _sha256(manifest_path),
        "evaluator": evaluator,
        "metrics": metrics,
    }
    evidence_path = tmp_path / "post_run_evidence.json"
    _write_evidence(evidence_path, payload)
    regenerated = deepcopy(payload)
    calls: list[dict[str, Any]] = []

    def fake_finalize(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return deepcopy(regenerated)

    monkeypatch.setattr(suite, "finalize_cellcf_run", fake_finalize)
    kwargs = {
        "repo_root": repo_root,
        "variant": "uniform",
        "commit": commit,
        "seed": seed,
        "config_path": config,
        "config_sha256": _sha256(config),
        "resolved_config_sha256": resolved_config_sha256,
        "protocol_sha256": protocol_sha256,
        "order_sha256": order_sha256,
        "gate_sha256": _sha256(gate),
        "pilot_sha256": _sha256(pilot),
        "gate_path": gate,
        "pilot_path": pilot,
        "annotation_path": annotation,
        "annotation_sha256": _sha256(annotation),
        "class_map_path": class_map,
        "class_map_sha256": _sha256(class_map),
        "evaluation_config_sha256": evaluation_config_sha256,
    }
    return {
        "evidence": evidence_path,
        "payload": payload,
        "kwargs": kwargs,
        "calls": calls,
        "terminal": {
            "checkpoint": checkpoint,
            "checkpoint sidecar": sidecar_path,
            "training audit": audit_path,
            "terminal evaluation": evaluation_path,
            "prediction": prediction,
            "run manifest": manifest_path,
        },
        "assets": {
            "manifest config": config,
            "runtime pretrain": pretrain,
            "real-loader gate": gate,
            "DDP pilot": pilot,
            "audit annotation": annotation,
            "audit class map": class_map,
            "evaluator source": evaluator_source,
        },
        "audit": audit_path,
        "sidecar": sidecar_path,
    }


def test_post_run_reopens_artifacts_and_regenerates_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)

    result = suite._validate_post_run(case["evidence"], **case["kwargs"])

    assert result["reproduced_from_terminal_artifacts"] is True
    assert set(result["artifact_revalidation"]["terminal_artifacts"]) == {
        "checkpoint",
        "checkpoint_sidecar",
        "training_audit",
        "terminal_evaluation",
        "prediction",
        "run_manifest",
    }
    assert len(case["calls"]) == 1


def test_post_run_binds_suite_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)
    forged = deepcopy(case["payload"])
    forged["seed"] += 1
    _write_evidence(case["evidence"], forged)

    with pytest.raises(ValueError, match="post-run seed mismatch"):
        suite._validate_post_run(case["evidence"], **case["kwargs"])


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("checkpoint", "checkpoint hash mismatch"),
        ("checkpoint sidecar", "checkpoint sidecar hash mismatch"),
        ("training audit", "training audit hash mismatch"),
        ("terminal evaluation", "terminal evaluation hash mismatch"),
        ("prediction", "prediction hash mismatch"),
        ("run manifest", "run manifest hash mismatch"),
    ],
)
def test_post_run_rejects_terminal_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    message: str,
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)
    with case["terminal"][artifact].open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match=message):
        suite._validate_post_run(case["evidence"], **case["kwargs"])


@pytest.mark.parametrize(
    "asset",
    [
        "manifest config",
        "runtime pretrain",
        "real-loader gate",
        "DDP pilot",
        "audit annotation",
        "audit class map",
        "evaluator source",
    ],
)
def test_post_run_rejects_exposed_asset_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset: str
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)
    with case["assets"][asset].open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="hash mismatch"):
        suite._validate_post_run(case["evidence"], **case["kwargs"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 99, "training audit seed mismatch"),
        ("variant", "cellcf", "training audit variant mismatch"),
        ("git_commit", "0" * 40, "training audit commit mismatch"),
    ],
)
def test_post_run_rejects_checkpoint_provenance_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)
    audit = json.loads(case["audit"].read_text(encoding="utf-8"))
    audit[field] = value
    _write_json(case["audit"], audit)
    sidecar = json.loads(case["sidecar"].read_text(encoding="utf-8"))
    sidecar["experiment_metadata"]["training_audit"] = audit
    _write_json(case["sidecar"], sidecar)
    forged = deepcopy(case["payload"])
    forged["training_audit_sha256"] = _sha256(case["audit"])
    forged["checkpoint_sidecar_sha256"] = _sha256(case["sidecar"])
    _write_evidence(case["evidence"], forged)

    with pytest.raises(ValueError, match=message):
        suite._validate_post_run(case["evidence"], **case["kwargs"])


def test_post_run_rejects_rehashed_summary_scalar_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _post_run_case(tmp_path, monkeypatch)
    forged = deepcopy(case["payload"])
    forged["metrics"]["average_mAP"] = 0.9
    _write_evidence(case["evidence"], forged)

    with pytest.raises(ValueError, match="not reproducible"):
        suite._validate_post_run(case["evidence"], **case["kwargs"])


def _cost_case(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    evidence_repo_root = tmp_path / "evidence_repo"
    cellcf_config = repo_root / suite.VARIANTS["cellcf"]
    bare_config = repo_root / suite.BARE_COST_CONFIG
    evidence_cellcf_config = evidence_repo_root / suite.VARIANTS["cellcf"]
    evidence_bare_config = evidence_repo_root / suite.BARE_COST_CONFIG
    cellcf_config.parent.mkdir(parents=True, exist_ok=True)
    evidence_cellcf_config.parent.mkdir(parents=True, exist_ok=True)
    cellcf_config.write_text("model = {'selector': True}\n", encoding="utf-8")
    bare_config.write_text("model = {'selector': False}\n", encoding="utf-8")
    evidence_cellcf_config.write_bytes(cellcf_config.read_bytes())
    evidence_bare_config.write_bytes(bare_config.read_bytes())
    checkpoint = tmp_path / "epoch_131.pth"
    checkpoint.write_bytes(b"terminal-cellcf-checkpoint")
    checkpoint_sha256 = _sha256(checkpoint)
    commit = "a" * 40
    evidence_commit = "e" * 40
    post_run_payload = {
        "schema": suite.POST_RUN_SCHEMA,
        "ok": True,
        "git_commit": commit,
        "seed": 0,
        "variant": "cellcf",
        "training_profile": "exposure132",
        "config_sha256": _sha256(cellcf_config),
        "resolved_config_sha256": "b" * 64,
        "runtime_config_sha256": "c" * 64,
        "evaluation_runtime_config_sha256": "d" * 64,
        "successful_optimizer_updates": suite.EXPECTED_UPDATES,
        "checkpoint_epoch": suite.TERMINAL_EPOCH,
        "checkpoint_state_key": suite.TERMINAL_STATE_KEY,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_payload_contract": {
            "payload_reopened": True,
            "epoch": suite.TERMINAL_EPOCH,
        },
    }
    post_run = tmp_path / "post_run_evidence.json"
    _write_evidence(post_run, post_run_payload)
    post_run_sha256 = _sha256(post_run)
    cost_binding = load_cellcf_cost_binding(post_run, post_run_sha256)
    cost_binding_sha256 = suite._canonical_sha256(cost_binding)
    common = {
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": "hardware",
        "host_fingerprint": "host",
        "software_fingerprint": "software",
        "config_commit": commit,
        "trained_commit": commit,
        "evidence_git_commit": evidence_commit,
        "inference_code_tree_binding": TREE_BINDING,
        "tracked_tree_clean": True,
        "dataset_fingerprint": "profile-dataset",
        "source_dataset_fingerprint": "dataset",
        "inference_fingerprint": "inference",
        "detector_stack_fingerprint": "detector",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 20,
        "amp": True,
        "random_init": False,
        "uses_ema": True,
        "power_sampling_enabled": True,
        "power_interval_ms": 20,
        "power_gpu_id": "GPU-1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_epoch": suite.TERMINAL_EPOCH,
        "checkpoint_state_key": suite.TERMINAL_STATE_KEY,
        "checkpoint_sha256": checkpoint_sha256,
        "cellcf_cost_binding": cost_binding,
        "cellcf_cost_binding_sha256": cost_binding_sha256,
        "weight_source": "cellcf_trained_terminal_state_dict_ema",
        "dense_full_stack_savings_claimed": False,
    }
    cellcf_paths = []
    bare_paths = []
    for repeat in range(3):
        repeat_index = repeat + 1
        cellcf_path = tmp_path / f"cellcf_repeat{repeat}.json"
        bare_path = tmp_path / f"bare_repeat{repeat}.json"
        cellcf_metadata = {
            **common,
            "method": "cellcf-fixed384",
            "config_path": str(evidence_cellcf_config.resolve()),
            "profile_config_sha256": _sha256(evidence_cellcf_config),
            "profile_resolved_config_sha256": cost_binding[
                "resolved_config_sha256"
            ],
            "frontend_variant": "cellcf",
            "checkpoint_dropped_prefixes": [],
            "checkpoint_dropped_key_count": 0,
            "profile_session_id": "slurm-test",
            "profile_pair_id": f"repeat-{repeat_index}",
            "profile_repeat_index": repeat_index,
            "profile_order_position": (
                1 if repeat_index % 2 == 1 else 2
            ),
        }
        bare_metadata = {
            **common,
            "method": "bare-uniform384",
            "config_path": str(evidence_bare_config.resolve()),
            "profile_config_sha256": _sha256(evidence_bare_config),
            "profile_resolved_config_sha256": "e" * 64,
            "frontend_variant": "bare_exact_uniform_lower_bound",
            "checkpoint_dropped_prefixes": ["frame_selector."],
            "checkpoint_dropped_key_count": 1,
            "profile_session_id": "slurm-test",
            "profile_pair_id": f"repeat-{repeat_index}",
            "profile_repeat_index": repeat_index,
            "profile_order_position": (
                2 if repeat_index % 2 == 1 else 1
            ),
        }

        def profile_sample(end_to_end_ms: float, selector_ms: float) -> dict:
            return {
                "input_pipeline_serial_ms": 1.0,
                "h2d_ms": 1.0,
                "model_forward_ms": end_to_end_ms - 3.0,
                "postprocess_ms": 1.0,
                "frame_selector_total_ms": selector_ms,
                "coarse_probe_ms": selector_ms * 0.5,
                "backbone_wrapper_total_ms": 3.0,
                "heavy_backbone_ms": 2.0,
                "projection_ms": 1.0,
                "neck_ms": 0.5,
                "head_ms": 0.5,
                "selected_count": 384,
            }

        _write_json(
            cellcf_path,
            build_profile_summary(
                [profile_sample(12.0 + repeat, 1.0)] * 500,
                metadata=cellcf_metadata,
            ),
        )
        _write_json(
            bare_path,
            build_profile_summary(
                [profile_sample(10.0 + repeat, 0.0)] * 500,
                metadata=bare_metadata,
            ),
        )
        cellcf_paths.append(str(cellcf_path.resolve()))
        bare_paths.append(str(bare_path.resolve()))
    cost_payload = summarize(
        cellcf_paths,
        bare_paths,
        post_run_evidence_path=post_run,
        post_run_evidence_sha256=post_run_sha256,
    )
    cost_path = tmp_path / "cellcf_vs_bare_uniform.json"
    _write_json(cost_path, cost_payload)
    return {
        "repo_root": repo_root,
        "evidence_repo_root": evidence_repo_root,
        "evidence_commit": evidence_commit,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "commit": commit,
        "post_run": post_run,
        "post_run_sha256": post_run_sha256,
        "cost": cost_path,
        "cellcf_profiles": [Path(path) for path in cellcf_paths],
        "bare_profiles": [Path(path) for path in bare_paths],
    }


def test_cost_evidence_reopens_profiles_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )

    result = suite.validate_cost_evidence(
        case["cost"],
        repo_root=case["repo_root"],
        expected_commit=case["commit"],
        evidence_repo_root=case["evidence_repo_root"],
        expected_evidence_commit=case["evidence_commit"],
        expected_checkpoint_path=case["checkpoint"],
        expected_checkpoint_sha256=case["checkpoint_sha256"],
        expected_post_run_evidence_path=case["post_run"],
        expected_post_run_evidence_sha256=case["post_run_sha256"],
    )

    assert result["validated"] is True
    assert len(result["cellcf_profiles"]) == 3
    assert len(result["bare_uniform_profiles"]) == 3
    assert result["cost_producer_evidence_commit"] == case["evidence_commit"]


def test_cost_evidence_rejects_profile_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )
    profile_path = case["cellcf_profiles"][0]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["stages"]["end_to_end_serial_ms"]["p50"] += 5.0
    _write_json(profile_path, profile)

    with pytest.raises(ValueError, match="profile artifact.*hash mismatch"):
        suite.validate_cost_evidence(
            case["cost"],
            repo_root=case["repo_root"],
            expected_commit=case["commit"],
            evidence_repo_root=case["evidence_repo_root"],
            expected_evidence_commit=case["evidence_commit"],
            expected_checkpoint_path=case["checkpoint"],
            expected_checkpoint_sha256=case["checkpoint_sha256"],
            expected_post_run_evidence_path=case["post_run"],
            expected_post_run_evidence_sha256=case["post_run_sha256"],
        )


def test_cost_evidence_rejects_checkpoint_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )
    case["checkpoint"].write_bytes(b"changed-checkpoint")

    with pytest.raises(ValueError, match="cost checkpoint hash mismatch"):
        suite.validate_cost_evidence(
            case["cost"],
            repo_root=case["repo_root"],
            expected_commit=case["commit"],
            evidence_repo_root=case["evidence_repo_root"],
            expected_evidence_commit=case["evidence_commit"],
            expected_checkpoint_path=case["checkpoint"],
            expected_checkpoint_sha256=case["checkpoint_sha256"],
            expected_post_run_evidence_path=case["post_run"],
            expected_post_run_evidence_sha256=case["post_run_sha256"],
        )


def test_cost_evidence_rejects_wrong_producer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )

    with pytest.raises(
        ValueError, match="cost producer evidence commit mismatch"
    ):
        suite.validate_cost_evidence(
            case["cost"],
            repo_root=case["repo_root"],
            expected_commit=case["commit"],
            evidence_repo_root=case["evidence_repo_root"],
            expected_evidence_commit="f" * 40,
            expected_checkpoint_path=case["checkpoint"],
            expected_checkpoint_sha256=case["checkpoint_sha256"],
            expected_post_run_evidence_path=case["post_run"],
            expected_post_run_evidence_sha256=case["post_run_sha256"],
        )


def test_cost_evidence_rejects_profile_config_outside_evidence_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    profile_path = case["cellcf_profiles"][0]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["config_path"] = str(
        (case["repo_root"] / suite.VARIANTS["cellcf"]).resolve()
    )
    _write_json(profile_path, profile)
    cost_payload = summarize(
        [str(path) for path in case["cellcf_profiles"]],
        [str(path) for path in case["bare_profiles"]],
        post_run_evidence_path=case["post_run"],
        post_run_evidence_sha256=case["post_run_sha256"],
    )
    _write_json(case["cost"], cost_payload)
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )

    with pytest.raises(ValueError, match="escaped the evidence repository"):
        suite.validate_cost_evidence(
            case["cost"],
            repo_root=case["repo_root"],
            expected_commit=case["commit"],
            evidence_repo_root=case["evidence_repo_root"],
            expected_evidence_commit=case["evidence_commit"],
            expected_checkpoint_path=case["checkpoint"],
            expected_checkpoint_sha256=case["checkpoint_sha256"],
            expected_post_run_evidence_path=case["post_run"],
            expected_post_run_evidence_sha256=case["post_run_sha256"],
        )


def test_cost_evidence_rejects_symlinked_evidence_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _cost_case(tmp_path)
    config = (
        case["evidence_repo_root"] / suite.VARIANTS["cellcf"]
    )
    outside = tmp_path / "outside-cellcf-config.py"
    outside.write_bytes(config.read_bytes())
    config.unlink()
    try:
        config.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    monkeypatch.setattr(
        suite,
        "_validate_exact_repository",
        lambda root, **_kwargs: Path(root).resolve(),
    )
    monkeypatch.setattr(
        suite,
        "_expected_inference_code_tree_binding",
        lambda *_args, **_kwargs: TREE_BINDING,
    )

    with pytest.raises(ValueError, match="symbolic-link component"):
        suite.validate_cost_evidence(
            case["cost"],
            repo_root=case["repo_root"],
            expected_commit=case["commit"],
            evidence_repo_root=case["evidence_repo_root"],
            expected_evidence_commit=case["evidence_commit"],
            expected_checkpoint_path=case["checkpoint"],
            expected_checkpoint_sha256=case["checkpoint_sha256"],
            expected_post_run_evidence_path=case["post_run"],
            expected_post_run_evidence_sha256=case["post_run_sha256"],
        )


def test_suite_required_cost_interface_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    for relative in suite.VARIANTS.values():
        config = root / relative
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("config = True\n", encoding="utf-8")
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "category_idx.txt"
    gate = tmp_path / "gate.json"
    pilot = tmp_path / "pilot.json"
    _write_json(annotation, {})
    class_map.write_text("1 Action\n", encoding="utf-8")
    _write_json(gate, {})
    _write_json(pilot, {})
    commit = "a" * 40

    class FakeConfig:
        def to_dict(self) -> dict[str, Any]:
            return {}

    def fake_git(_root: Path, *args: str) -> str:
        return commit if args == ("rev-parse", "HEAD") else ""

    data = {
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config_sha256": "e" * 64,
    }
    monkeypatch.setattr(suite, "_git", fake_git)
    monkeypatch.setattr(suite.Config, "fromfile", lambda _path: FakeConfig())
    monkeypatch.setattr(suite, "_shared_protocol", lambda _cfg: {"protocol": True})
    monkeypatch.setattr(
        suite, "_variant_contract", lambda _cfg, variant: {"variant": variant}
    )
    monkeypatch.setattr(
        suite,
        "validate_config",
        lambda variant, _path, **_kwargs: {"variant": variant},
    )
    monkeypatch.setattr(suite, "_reference_data", lambda _cfg: data)
    monkeypatch.setattr(
        suite,
        "_validate_gate",
        lambda _path, _commit: {
            "path": str(gate.resolve()),
            "sha256": _sha256(gate),
            "payload": {
                "dataset": {
                    "annotation_sha256": data["evaluation_annotation_sha256"],
                    "class_map_sha256": data["evaluation_class_map_sha256"],
                }
            },
        },
    )
    monkeypatch.setattr(
        suite,
        "_validate_pilot",
        lambda *_args, **_kwargs: {
            "path": str(pilot.resolve()),
            "sha256": _sha256(pilot),
            "payload": {},
        },
    )

    with pytest.raises(ValueError, match="required CellCF cost evidence is missing"):
        suite.validate_suite(
            repo_root=root,
            seed=0,
            expected_commit=commit,
            require_clean=True,
            gate_json=gate,
            pilot_json=pilot,
            require_cost_evidence=True,
        )


def test_suite_status_distinguishes_cost_pending_from_formal_completion() -> None:
    completed = {variant: {"path": variant} for variant in suite.VARIANT_ORDER}
    assert suite._suite_status({}, {}) == "deployable_not_submitted"
    assert suite._suite_status(completed, {}) == "runs_complete_cost_pending"
    assert suite._suite_status(completed, {"validated": True}) == "complete"
