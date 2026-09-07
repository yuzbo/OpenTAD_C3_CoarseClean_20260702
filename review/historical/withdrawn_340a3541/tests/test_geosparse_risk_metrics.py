import pytest
from geosparse_ext.metrics import detection_risks, calibrate_risk_score, paired_risk_bootstrap


def payload():
    annotation = dict(database={"v": dict(annotations=[dict(label="a", segment=[0., 2.]), dict(label="a", segment=[5., 9.])])})
    predictions = dict(results={"v": [dict(label="wrong", segment=[0., 2.], score=.9)]})
    return annotation, predictions


def test_capped_boundary_counts_misses_and_separates_class_errors():
    annotation, predictions = payload()
    result = detection_risks(annotation, predictions, ["v"], 2.)
    assert result["matches"] == 0 and result["boundary_matched_fraction"] == .5
    assert result["capped_start_mae_seconds"] == 2. and result["capped_end_error_per_duration"] == .5
    assert result["start_mae_seconds"] is None
    assert result["per_video"]["v"]["per_gt"][1]["capped_end_seconds"] == 4.


def test_score_cap_and_internal_dev_calibration_scope():
    annotation, predictions = payload()
    predictions["results"]["v"] = [dict(label="wrong", segment=[20., 21.], score=.9) for _ in range(100)] + [dict(label="a", segment=[0., 2.], score=.1)]
    result = detection_risks(annotation, predictions, ["v"], 2.)
    assert result["prediction_count"] == 100 and result["matches"] == 0
    with pytest.raises(ValueError, match="internal_dev"):
        calibrate_risk_score(annotation, predictions, dict(internal_dev=["other"]), 2.)
    receipt = calibrate_risk_score(annotation, predictions, dict(internal_dev=["v"]), 2.)
    assert receipt["videos"] == ["v"] and receipt["score_threshold"] == 0.


def test_paired_bootstrap_preserves_video_clusters_and_empty_gt():
    annotation, predictions = payload()
    annotation["database"]["empty"] = dict(annotations=[])
    result = detection_risks(annotation, predictions, ["v", "empty"], 2.)
    bootstrap = paired_risk_bootstrap(result, result, samples=1000)
    assert all(set(row["draws"]) <= {0.} for row in bootstrap["metrics"].values())
    assert bootstrap["unit"] == "paired videos"
    assert detection_risks(annotation, predictions, ["empty"], 2.)["capped_start_mae_seconds"] is None


def test_duration_thresholds_use_training_and_absolute_bins_stay_fixed():
    from geosparse_ext.evaluation import duration_slices
    annotation, prediction = payload()
    annotation["database"]["train"] = dict(annotations=[dict(label="a", segment=[0., 100.])])
    risk = detection_risks(annotation, prediction, ["v"], 2.)
    sliced = duration_slices(risk, annotation, ["train"])
    assert sliced["training_duration_q3_seconds"] == 100.
    assert sliced["strata"]["middle_q2_q3"]["gt_count"] == 1
    assert sliced["strata"]["2-5s"]["gt_count"] == 1


def test_prediction_export_rejects_incomplete_or_mismatched_training(tmp_path):
    import json
    from geosparse_ext.prediction_export import training_source
    files = {"bindings.json": dict(source_commit="unit-source"), "source_commits.json": dict(source_commit="unit-source"),
             "job.json": dict(job_id="unit-train"),
             "result.json": dict(status="completed", is_mock=False, completed_epochs=59, source_commit="unit-source")}
    for name, contents in files.items():
        (tmp_path / name).write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="60-epoch"):
        training_source(tmp_path)
    files["result.json"].update(completed_epochs=60, source_commit="different-source")
    (tmp_path / "result.json").write_text(json.dumps(files["result.json"]))
    with pytest.raises(ValueError, match="provenance"):
        training_source(tmp_path)
