from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.create_duca_frontend_split import create_split
from tools.bata.duca_p0_evaluation import (
    canonical_sha256,
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
    sha256_file,
)
from tools.bata.finalize_duca_r0_boundary_burst import (
    EVALUATION_SCHEMA,
    FAMILY_ORDER,
    finalize_r0,
    revalidate_r0_summary,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, mixed_projected_gain: bool = False) -> dict:
    database = {
        f"video_{index}": {
            "subset": "training",
            "duration": 30.0,
            "frame": 900,
            "annotations": [{"segment": [2.0, 4.0], "label": "Action"}],
        }
        for index in range(6)
    }
    annotation = tmp_path / "annotation.json"
    _write_json(annotation, {"database": database})
    split_dir = tmp_path / "split"
    split = create_split(annotation, split_dir, seed=3407, holdout_fraction=0.4)
    holdout = list(split["holdout_videos"])
    train = list(split["train_videos"])
    blocked = tmp_path / "evaluation_blocked.json"
    _write_json(blocked, train)
    class_map = tmp_path / "classes.txt"
    class_map.write_text("Action\n", encoding="utf-8")
    checkpoint = tmp_path / "epoch_131.pth"
    checkpoint.write_bytes(b"checkpoint")
    pretrain = tmp_path / "pretrain.pth"
    pretrain.write_bytes(b"pretrain")
    config = tmp_path / "r0_config.py"
    config.write_text("formal = True\n", encoding="utf-8")
    artifact = tmp_path / "families.jsonl"
    artifact_rows = []
    for video in holdout:
        families = []
        for family in FAMILY_ORDER:
            contract = {"exact_k": True}
            if family in FAMILY_ORDER[1:3]:
                contract.update(
                    {
                        "global_coverage_enforced": True,
                        "max_unselected_hole": 2,
                    }
                )
            if family == FAMILY_ORDER[3]:
                contract.update(
                    {
                        "global_coverage_enforced": False,
                        "coverage_cap": "unrestricted",
                        "projected_into_deployment_feasible_set": False,
                    }
                )
            families.append({"family_key": family, "r0_contract": contract})
        row = {
            "schema_version": "duca_allocation_family_ceiling_record_v1",
            "sample_id": f"{video}|0",
            "video_id": video,
            "families": families,
        }
        row["record_sha256"] = canonical_sha256(row)
        artifact_rows.append(row)
    artifact.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifact_rows),
        encoding="utf-8",
    )
    family_summary = tmp_path / "families.summary.json"
    _write_json(
        family_summary,
        {
            "schema_version": "duca_r0_boundary_burst_oracle_summary_v1",
            "ok": True,
            "sample_count": len(holdout),
            "families": {family: len(holdout) for family in FAMILY_ORDER},
            "output_jsonl": str(artifact.resolve()),
            "output_jsonl_sha256": sha256_file(artifact),
            "max_unselected_hole": 2,
            "crop_cut_endpoints_excluded": True,
            "diagnostic_only": True,
        },
    )
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": str(annotation.resolve()),
        "subset": "training",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": str(blocked.resolve()),
        "thread": 1,
    }
    normalized = normalize_evaluation_config(
        evaluation_config,
        expected_subset="training",
    )
    family_evaluations: dict[str, Path] = {}
    for family_index, family in enumerate(FAMILY_ORDER):
        results = {}
        for video_index, video in enumerate(holdout):
            correct = family_index > 0
            if mixed_projected_gain and family in FAMILY_ORDER[1:3]:
                correct = video_index == 0
            segment = [2.0, 4.0] if correct else [20.0, 22.0]
            results[video] = [{"segment": segment, "label": "Action", "score": 0.9}]
        prediction = tmp_path / "predictions" / family / "result.json"
        _write_json(prediction, {"results": results})
        official = recompute_official_map(
            prediction,
            normalized,
            expected_subset="training",
        )
        evaluation = tmp_path / "evaluations" / family / "metrics.json"
        payload = {
            "schema_version": EVALUATION_SCHEMA,
            "git_commit": "a" * 40,
            "task": "offline_temporal_action_detection",
            "config_path": str(config.resolve()),
            "config_sha256": sha256_file(config),
            "resolved_config_sha256": canonical_sha256({"family": family}),
            "runtime_config_sha256": canonical_sha256({"runtime": family}),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": 131,
            "checkpoint_state_key": "state_dict_ema",
            "prediction_path": str(prediction.resolve()),
            "prediction_sha256": sha256_file(prediction),
            "metrics": official["metrics"],
            "result_count": official["result_count"],
            "video_count": official["video_count"],
            "evaluator": official_evaluator_identity(),
            "evaluation_config": normalized,
            "evaluation_config_sha256": canonical_sha256(normalized),
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": sha256_file(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": sha256_file(class_map),
            "seed": 3407,
            "family": family,
            "allocation_artifact_path": str(artifact.resolve()),
            "allocation_artifact_sha256": sha256_file(artifact),
            "evaluation_blocked_videos_path": str(blocked.resolve()),
            "evaluation_blocked_videos_sha256": sha256_file(blocked),
            "source_subset": "training_internal_holdout",
            "test_subset_consumed": False,
            "runtime_gt_input_to_selector": False,
        }
        payload["evaluation_sha256"] = canonical_sha256(payload)
        _write_json(evaluation, payload)
        family_evaluations[family] = evaluation
    return {
        "split": split,
        "checkpoint": checkpoint,
        "pretrain": pretrain,
        "config": config,
        "artifact": artifact,
        "family_summary": family_summary,
        "blocked": blocked,
        "family_evaluations": family_evaluations,
    }


def _finalize(
    tmp_path: Path,
    fixture: dict,
    *,
    bootstrap_workers: int = 1,
) -> tuple[dict, Path]:
    summary_path = tmp_path / "r0_summary.json"
    summary = finalize_r0(
        expected_commit="a" * 40,
        family_evaluations=fixture["family_evaluations"],
        split_manifest=fixture["split"]["manifest_path"]
        if "manifest_path" in fixture["split"]
        else Path(fixture["split"]["train_block_list"]).parent / "frontend_split_manifest.json",
        split_manifest_sha256=sha256_file(
            Path(fixture["split"]["train_block_list"]).parent / "frontend_split_manifest.json"
        ),
        checkpoint_path=fixture["checkpoint"],
        checkpoint_sha256=sha256_file(fixture["checkpoint"]),
        checkpoint_epoch=131,
        config_path=fixture["config"],
        config_sha256=sha256_file(fixture["config"]),
        allocation_artifact_path=fixture["artifact"],
        allocation_artifact_sha256=sha256_file(fixture["artifact"]),
        family_summary_path=fixture["family_summary"],
        family_summary_sha256=sha256_file(fixture["family_summary"]),
        pretrain_path=fixture["pretrain"],
        pretrain_sha256=sha256_file(fixture["pretrain"]),
        blocked_videos_path=fixture["blocked"],
        blocked_videos_sha256=sha256_file(fixture["blocked"]),
        bootstrap_output_path=tmp_path / "r0_bootstrap.json",
        summary_output_path=summary_path,
        bootstrap_samples=100,
        bootstrap_seed=3407,
        bootstrap_confidence=0.95,
        bootstrap_workers=bootstrap_workers,
        required_headroom_percentage_points=0.20,
    )
    return summary, summary_path


def test_parallel_r0_bootstrap_is_exactly_equal_to_serial(tmp_path: Path) -> None:
    serial_fixture = _fixture(tmp_path / "serial")
    parallel_fixture = _fixture(tmp_path / "parallel")
    serial_summary, _ = _finalize(
        tmp_path / "serial", serial_fixture, bootstrap_workers=1
    )
    parallel_summary, _ = _finalize(
        tmp_path / "parallel", parallel_fixture, bootstrap_workers=2
    )
    serial_bootstrap = json.loads(
        Path(serial_summary["bootstrap_path"]).read_text(encoding="utf-8")
    )
    parallel_bootstrap = json.loads(
        Path(parallel_summary["bootstrap_path"]).read_text(encoding="utf-8")
    )

    assert (
        parallel_bootstrap["sampled_average_mAP"]
        == serial_bootstrap["sampled_average_mAP"]
    )
    assert parallel_bootstrap["comparisons"] == serial_bootstrap["comparisons"]
    assert (
        parallel_summary["selected_weakest_projected_family"]
        == serial_summary["selected_weakest_projected_family"]
    )


def test_r0_consumer_revalidates_sealed_official_bootstrap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary, summary_path = _finalize(tmp_path, fixture)

    assert summary["selected_weakest_projected_family"] == FAMILY_ORDER[1]
    gate = revalidate_r0_summary(
        summary_path=summary_path,
        summary_file_sha256=sha256_file(summary_path),
        expected_commit="a" * 40,
    )
    assert gate["ok"]
    assert gate["official_evaluator_reexecuted_per_resample"]
    assert gate["consumer_revalidated_sealed_bootstrap_without_reexecution"]
    assert gate["selected_weakest_projected_family"] == FAMILY_ORDER[1]


def test_r0_consumer_rejects_resealed_bootstrap_arithmetic_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    summary, summary_path = _finalize(tmp_path, fixture)
    bootstrap_path = Path(summary["bootstrap_path"])
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["comparisons"][FAMILY_ORDER[1]]["headroom_ci_lower"] += 0.01
    bootstrap.pop("bootstrap_sha256")
    bootstrap["bootstrap_sha256"] = canonical_sha256(bootstrap)
    _write_json(bootstrap_path, bootstrap)

    summary["bootstrap_file_sha256"] = sha256_file(bootstrap_path)
    summary["bootstrap_self_sha256"] = bootstrap["bootstrap_sha256"]
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="bootstrap arithmetic mismatch"):
        revalidate_r0_summary(
            summary_path=summary_path,
            summary_file_sha256=sha256_file(summary_path),
            expected_commit="a" * 40,
        )


def test_r0_rejects_point_gain_when_bootstrap_lower_bound_fails(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, mixed_projected_gain=True)
    summary, _ = _finalize(tmp_path, fixture)

    assert summary["rows"][1]["headroom_vs_uniform_average_mAP"] > 0.0
    assert summary["rows"][1]["headroom_bootstrap_ci_lower"] <= 0.002
    assert summary["ok"] is False
    assert summary["selected_weakest_projected_family"] is None


def test_r0_rejects_resealed_family_order_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _, summary_path = _finalize(tmp_path, fixture)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["rows"][1], payload["rows"][2] = payload["rows"][2], payload["rows"][1]
    payload.pop("summary_sha256")
    payload["summary_sha256"] = canonical_sha256(payload)
    _write_json(summary_path, payload)

    with pytest.raises(RuntimeError, match="row order"):
        revalidate_r0_summary(
            summary_path=summary_path,
            summary_file_sha256=sha256_file(summary_path),
            expected_commit="a" * 40,
        )


@pytest.mark.parametrize(
    "mutation",
    ("evaluator", "subset", "blocked", "prediction", "annotation", "class_map", "config"),
)
def test_r0_rejects_resealed_identity_mutations(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    _, summary_path = _finalize(tmp_path, fixture)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary["rows"][0]
    evaluation_path = Path(row["evaluation_path"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if mutation == "evaluator":
        evaluation["evaluator"] = {"module": "wrong", "class_name": "mAP"}
    elif mutation == "subset":
        evaluation["evaluation_config"]["subset"] = "validation"
        evaluation["evaluation_config_sha256"] = canonical_sha256(
            evaluation["evaluation_config"]
        )
    elif mutation == "blocked":
        altered = tmp_path / "altered_blocked.json"
        _write_json(altered, [])
        evaluation["evaluation_config"]["blocked_videos"] = str(altered.resolve())
        evaluation["evaluation_config_sha256"] = canonical_sha256(
            evaluation["evaluation_config"]
        )
        evaluation["evaluation_blocked_videos_path"] = str(altered.resolve())
        evaluation["evaluation_blocked_videos_sha256"] = sha256_file(altered)
    elif mutation == "prediction":
        altered = tmp_path / "altered_prediction.json"
        _write_json(altered, {"results": {}})
        evaluation["prediction_path"] = str(altered.resolve())
        evaluation["prediction_sha256"] = sha256_file(altered)
        row["prediction_path"] = str(altered.resolve())
        row["prediction_sha256"] = sha256_file(altered)
    elif mutation == "annotation":
        altered = tmp_path / "altered_annotation.json"
        _write_json(altered, {"database": {}})
        evaluation["evaluation_config"]["ground_truth_filename"] = str(altered.resolve())
        evaluation["evaluation_config_sha256"] = canonical_sha256(
            evaluation["evaluation_config"]
        )
        evaluation["evaluation_annotation_path"] = str(altered.resolve())
        evaluation["evaluation_annotation_sha256"] = sha256_file(altered)
    elif mutation == "class_map":
        altered = tmp_path / "altered_classes.txt"
        altered.write_text("Wrong\n", encoding="utf-8")
        evaluation["evaluation_class_map_path"] = str(altered.resolve())
        evaluation["evaluation_class_map_sha256"] = sha256_file(altered)
        row["class_map_path"] = str(altered.resolve())
        row["class_map_sha256"] = sha256_file(altered)
    elif mutation == "config":
        altered = tmp_path / "altered_config.py"
        altered.write_text("formal = False\n", encoding="utf-8")
        evaluation["config_path"] = str(altered.resolve())
        evaluation["config_sha256"] = sha256_file(altered)
    evaluation.pop("evaluation_sha256")
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    _write_json(evaluation_path, evaluation)
    row["evaluation_file_sha256"] = sha256_file(evaluation_path)
    row["evaluation_self_sha256"] = evaluation["evaluation_sha256"]
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json(summary_path, summary)

    with pytest.raises((RuntimeError, ValueError)):
        revalidate_r0_summary(
            summary_path=summary_path,
            summary_file_sha256=sha256_file(summary_path),
            expected_commit="a" * 40,
        )
