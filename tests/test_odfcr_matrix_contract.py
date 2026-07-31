import hashlib
import json
import os
import sys

import pytest

from tools.aggregate_odfcr_internal_matrices import (
    EXPECTED_DEV_SEEDS,
    main as aggregate_main,
)
from tools.finalize_odfcr_internal_matrix import _delta_pp


def _metrics(average, tiou=None):
    if tiou is None:
        tiou = [average] * 5
    return {
        "average_mAP": average,
        "mAP_at_0_3_to_0_7": list(tiou),
    }


def _contrast(average_pp, tiou_pp=None):
    if tiou_pp is None:
        tiou_pp = [average_pp] * 5
    return {
        "average_mAP": average_pp,
        "mAP_at_0_3_to_0_7": list(tiou_pp),
        "unit": "percentage_points",
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path):
    return {
        "path": os.path.realpath(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _ensure_shared_manifests(folder):
    annotation_path = folder / "annotation.json"
    previous_path = folder / "previous.json"
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return annotation_path, previous_path, manifest_path
    video_ids = ["video_{:03d}".format(index) for index in range(200)]
    annotation = {
        "database": {
            video_id: {
                "subset": "validation",
                "annotations": [
                    {
                        "label_id": index % 20,
                        "label": "class_{:02d}".format(index % 20),
                        "segment": [1.0, 2.0],
                    }
                ],
            }
            for index, video_id in enumerate(video_ids)
        }
    }
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    annotation_sha256 = _sha256(annotation_path)
    previous_train = video_ids[:160]
    previous_holdout = video_ids[160:]
    previous = {
        "schema_version": "actionformer_dcsr_internal_holdout_v1",
        "source_split": "validation",
        "source_annotation_sha256": annotation_sha256,
        "test_annotations_used": False,
        "test_records_selected": False,
        "train_video_ids": previous_train,
        "holdout_video_ids": previous_holdout,
        "train_video_count": 160,
        "holdout_video_count": 40,
    }
    previous_path.write_text(json.dumps(previous), encoding="utf-8")
    previous_sha256 = _sha256(previous_path)
    holdout = video_ids[:40]
    train = video_ids[40:]
    manifest = {
        "schema_version": "actionformer_odfcr_internal_holdout_v2",
        "source_annotation_path": os.path.realpath(annotation_path),
        "source_annotation_sha256": annotation_sha256,
        "source_split": "validation",
        "test_annotations_used": False,
        "test_records_selected": False,
        "predictions_read": False,
        "metrics_read": False,
        "checkpoint_read": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "previous_manifest_sha256": previous_sha256,
        "previous_manifest_schema_version": (
            "actionformer_dcsr_internal_holdout_v1"
        ),
        "previous_train_video_ids": previous_train,
        "previous_holdout_video_ids": previous_holdout,
        "candidate_pool_video_ids": previous_train,
        "candidate_pool_is_previous_train_only": True,
        "new_holdout_disjoint_previous_holdout": True,
        "all_class_ids": list(range(20)),
        "train_video_ids": train,
        "holdout_video_ids": holdout,
        "train_video_count": 160,
        "holdout_video_count": 40,
        "disjoint": True,
        "train_all_class_coverage": True,
        "holdout_all_class_coverage": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return annotation_path, previous_path, manifest_path


def _write_matrix(path, seed, d3_delta_pp):
    annotation_path, previous_path, manifest_path = _ensure_shared_manifests(
        path.parent
    )
    g0_path = path.parent / ("g0_{:d}.json".format(seed))
    g0_path.write_text(
        json.dumps(
            {
                "schema_version": "actionformer_odfcr_g0_equivalence_v1",
                "gate_pass": True,
                "seed": seed,
                "git_commit": "a" * 40,
                "git_tree": "b" * 40,
                "checks": {"identity": True},
                "config_sha256": {
                    "official": "0" * 64,
                    "d1_off": "1" * 64,
                    "d1_all": "2" * 64,
                    "d3_off": "3" * 64,
                    "d3_all": "4" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    arm_metrics = {
        "d1_off": _metrics(0.50),
        "d1_all": _metrics(0.51),
        "d3_off": _metrics(0.60),
        "d3_all": _metrics(0.60 + d3_delta_pp / 100.0),
    }
    arm_bindings = {
        arm: {
            "config_sha256": str(index) * 64,
            "checkpoint_sha256": "5" * 64,
            "raw_predictions_sha256": "6" * 64,
            "eval_log_sha256": "7" * 64,
        }
        for index, arm in enumerate(
            ("d1_off", "d1_all", "d3_off", "d3_all"), start=1
        )
    }
    metric_receipts = {}
    for arm, metrics in arm_metrics.items():
        metric_path = path.parent / (
            "metric_{:d}_{:s}.json".format(seed, arm)
        )
        metric_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "actionformer_odfcr_internal_metric_v1"
                    ),
                    "validation_pass": True,
                    "arm": arm,
                    "seed": seed,
                    "source_commit": "a" * 40,
                    "source_tree": "b" * 40,
                    "metrics": metrics,
                    **arm_bindings[arm],
                }
            ),
            encoding="utf-8",
        )
        metric_receipts[arm] = _receipt(metric_path)
    path.write_text(
        json.dumps(
            {
                "schema_version": "actionformer_odfcr_internal_matrix_v1",
                "validation_pass": True,
                "paper_performance_row_allowed": False,
                "official_test_authorized": False,
                "seed": seed,
                "git_commit": "a" * 40,
                "git_tree": "b" * 40,
                "g0_exact_equivalence_pass": True,
                "annotation_sha256": _sha256(annotation_path),
                "arm_metrics": arm_metrics,
                "arm_artifact_bindings": arm_bindings,
                "contrasts": {
                    "d1_all_minus_d1_off": _contrast(1.0),
                    "d3_all_minus_d3_off": _contrast(d3_delta_pp),
                    "d1_off_minus_d3_off": _contrast(-10.0),
                    "depth_by_residual_interaction": _contrast(
                        d3_delta_pp - 1.0
                    ),
                },
                "receipts": {
                    "g0": _receipt(g0_path),
                    "internal_holdout_manifest": _receipt(manifest_path),
                    "previous_holdout_manifest": _receipt(previous_path),
                    "arm_metrics": metric_receipts,
                },
                "source_split": "validation",
                "test_gt_used": False,
                "test_predictions_used": False,
                "route_decision_allowed": False,
                "requires_three_seed_aggregate": True,
                "efficiency_claim_allowed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _run_aggregate(monkeypatch, paths, output):
    argv = ["aggregate_odfcr_internal_matrices.py"]
    for path in paths:
        argv += ["--matrix", str(path)]
    argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    aggregate_main()
    return json.loads(output.read_text(encoding="utf-8"))


def test_odfcr_delta_is_explicitly_percentage_points():
    delta = _delta_pp(
        _metrics(0.605, [0.7, 0.65, 0.61, 0.52, 0.40]),
        _metrics(0.600, [0.69, 0.64, 0.60, 0.51, 0.39]),
    )
    assert delta["unit"] == "percentage_points"
    assert delta["average_mAP"] == pytest.approx(0.5)
    assert delta["mAP_at_0_3_to_0_7"] == pytest.approx([1.0] * 5)


def test_odfcr_positive_gate_uses_paired_seed_deltas(monkeypatch, tmp_path):
    paths = []
    for seed, delta in zip(EXPECTED_DEV_SEEDS, (0.25, 0.30, -0.10)):
        path = tmp_path / ("matrix_{:d}.json".format(seed))
        _write_matrix(path, seed, delta)
        paths.append(path)
    output = tmp_path / "aggregate.json"
    payload = _run_aggregate(monkeypatch, paths, output)

    assert payload["validation_pass"] is True
    assert payload["positive_d3_residual_seed_count"] == 2
    assert payload["residual_utility_gate_pass"] is False
    assert payload["paired_subtraction_before_aggregation"] is True
    assert payload["independent_validation_splits"] is False


def test_odfcr_negative_gate_is_valid_completion(monkeypatch, tmp_path):
    paths = []
    for seed, delta in zip(EXPECTED_DEV_SEEDS, (-0.5, -0.1, 0.0)):
        path = tmp_path / ("matrix_{:d}.json".format(seed))
        _write_matrix(path, seed, delta)
        paths.append(path)
    output = tmp_path / "aggregate.json"
    payload = _run_aggregate(monkeypatch, paths, output)

    assert payload["validation_pass"] is True
    assert payload["residual_utility_gate_pass"] is False
    assert payload["next_step_if_fail"].startswith("record legal negative")


def test_odfcr_aggregate_rejects_seed_drift(monkeypatch, tmp_path):
    paths = []
    for seed in (2026073101, 2026073102, 999):
        path = tmp_path / ("matrix_{:d}.json".format(seed))
        _write_matrix(path, seed, 0.5)
        paths.append(path)
    with pytest.raises(ValueError, match="seed set"):
        _run_aggregate(
            monkeypatch,
            paths,
            tmp_path / "aggregate.json",
        )


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_odfcr_aggregate_rejects_non_finite_or_boolean_metrics(
    monkeypatch, tmp_path, bad_value
):
    paths = []
    for seed in EXPECTED_DEV_SEEDS:
        path = tmp_path / ("matrix_{:d}.json".format(seed))
        _write_matrix(path, seed, 0.5)
        paths.append(path)
    matrix = json.loads(paths[0].read_text(encoding="utf-8"))
    matrix["arm_metrics"]["d3_all"]["average_mAP"] = bad_value
    paths[0].write_text(
        json.dumps(matrix, allow_nan=True, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite|non-numeric"):
        _run_aggregate(
            monkeypatch,
            paths,
            tmp_path / "aggregate.json",
        )
