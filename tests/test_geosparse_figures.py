"""Only temporary unit fixtures; never written to a formal run or figure directory."""
import json
from pathlib import Path

import numpy as np
import pytest

from geosparse_ext.figures import Report, eligible_train, formal_evaluations, intervals_union, plot_pareto, plot_cases, plot_coverage
from geosparse_ext.selection_viewer import native_polygons, build as viewer
from geosparse_ext.records import checkpoint_identity


def identity(train_id, epoch):
    return checkpoint_identity(train_id, epoch, {key: "fixture" for key in
        ("source_commit", "resolved_config_sha256", "split_sha256", "weights_sha256")}, f"fixture-{train_id}-{epoch}")


def test_discontinuous_support_keeps_unobserved_gap():
    assert intervals_union([[10, 11], [30, 31], [10.5, 11.5]]) == [[10, 11.5], [30, 31]]


def test_original_coordinate_overlay_inverts_crop_and_flip():
    # Original x=.25..75 becomes the view, then flips horizontally.
    matrix = [[-2, 0, 1.5], [0, 1, 0], [0, 0, 1]]
    first = np.asarray(native_polygons(matrix, 2)[0])
    np.testing.assert_allclose(first, [[.75, 0], [.5, 0], [.5, .5], [.75, .5]])


def test_unfinished_or_incomplete_best_never_qualifies_for_main_table():
    good = dict(status="completed", is_mock=False, completed_epochs=60, best_checkpoint_selection_complete=True)
    assert eligible_train(good)
    assert not eligible_train(dict(good, completed_epochs=5))
    assert not eligible_train(dict(good, is_mock=True))
    assert not eligible_train(dict(good, best_checkpoint_selection_complete=False))


def test_checkpoint_mismatch_is_rejected(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(dict(selected_checkpoint_epoch=4)))
    receipt = dict(status="completed", is_mock=False, completed_epochs=60, best_checkpoint_selection_complete=True, selected_checkpoint_epoch=9)
    runs = dict(train=(tmp_path, dict(kind="train"), receipt), evaluate=(tmp_path, dict(kind="evaluate", source_train_id="train"), dict(status="completed", is_mock=False)))
    with pytest.raises(ValueError, match="checkpoint differ"):
        formal_evaluations(runs)


def test_single_seed_is_not_a_final_mean(tmp_path):
    report = Report(tmp_path)
    train_job = dict(job_id="unit-test", families=["F01"], dataset="fixture", model=dict(route="A"))
    evaluations = [(tmp_path, dict(seed=0), train_job, dict(source_commit="fixture"), dict(official=dict(average_mAP=.5)))]
    plot_pareto(report, {}, evaluations, [])
    assert len(report.rows) == 1 and report.rows[0]["status"] == "WAITING_DATA"
    assert not list(tmp_path.glob("*.pdf"))


def test_missing_results_are_visible_without_fabricated_points(tmp_path):
    report = Report(tmp_path)
    report.wait("accuracy", "No real evaluation yet")
    report.finish()
    assert "等待真实数据" in (tmp_path / "index.html").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.png"))


def test_three_seed_plot_uses_mean_and_sample_sd_not_best_seed(tmp_path):
    import csv
    evaluations = []
    for seed, score in enumerate((.1, .2, .3)):
        job = dict(job_id=f"unit-{seed}", route="A", families=["F01"], dataset="fixture", model=dict(axis="T", budget=.5))
        training = dict(source_commit="fixture", selected_checkpoint_epoch=4 + seed * 5)
        metrics = dict(official=dict(average_mAP=score), selected_checkpoint_epoch=training["selected_checkpoint_epoch"])
        metrics["checkpoint_identity"] = training["checkpoint_identity"] = identity(job["job_id"], training["selected_checkpoint_epoch"])
        evaluations.append((tmp_path, dict(seed=seed), job, training, metrics))
    export = dict(source_train_id="unit-0", is_final_checkpoint=True, full_split=True, selected_checkpoint_epoch=4, model_source_commit="fixture")
    export["checkpoint_identity"] = identity("unit-0", 4)
    windows = [dict(heavy_macs=1e9, model_macs_counted=2e9, mac_count_complete=True)]
    report = Report(tmp_path)
    plot_pareto(report, {}, evaluations, [(tmp_path, export, windows)])
    with (tmp_path / "main_pareto_fixture.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["mean_mAP"]) == 20
    assert float(row["sd_mAP"]) == 10
    assert int(row["seed0_epoch"]) == 5


def test_single_seed_feasibility_uses_seed0_without_invented_sd(tmp_path):
    import csv
    job = dict(job_id="unit-0", route="A", families=["F01"], dataset="fixture", model=dict(axis="T", budget=.5))
    training = dict(source_commit="fixture", selected_checkpoint_epoch=4)
    metrics = dict(official=dict(average_mAP=.25), selected_checkpoint_epoch=4)
    export = dict(source_train_id="unit-0", is_final_checkpoint=True, full_split=True, selected_checkpoint_epoch=4, model_source_commit="fixture")
    training["checkpoint_identity"] = metrics["checkpoint_identity"] = export["checkpoint_identity"] = identity("unit-0", 4)
    windows = [dict(heavy_macs=1e9, model_macs_counted=2e9, mac_count_complete=True)]
    report = Report(tmp_path)
    plot_pareto(report, {}, [(tmp_path, dict(seed=0), job, training, metrics)], [(tmp_path, export, windows)], single_seed=True)
    with (tmp_path / "feasibility_pareto_fixture.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["mean_mAP"]) == 25 and row["sd_mAP"] == "" and row["seed_count"] == "1"
    assert not (tmp_path / "main_pareto_fixture.pdf").exists()


def test_qualitative_export_round_trip_retains_disjoint_support(tmp_path):
    import base64
    from PIL import Image
    source = tmp_path / "unit_fixture"
    source.mkdir()
    for i in (0, 1):
        Image.new("RGB", (320, 80), color=(128, 128, 128)).save(source / f"unit{i}.jpg")
    receipt = dict(status="completed_selection_export", is_mock=False, source_train_id="UNIT_FIXTURE_ONLY", seed=0,
        model_source_commit="UNIT_FIXTURE_ONLY", selected_checkpoint_epoch=4, is_final_checkpoint=False, full_split=False, videos=["unit-video"])
    row = dict(window_index=0, video_id="unit-video", window_id="unit-video:0", route="A", native_shape=[2, 2, 2],
        selected_bits=base64.b64encode(np.packbits([1] * 4 + [0] * 4, bitorder="little").tobytes()).decode(),
        spatial_transform=np.eye(3).tolist(), thumbnails=[dict(tubelet=0, paths=["unit0.jpg", "unit1.jpg"])],
        source_frame_id=[[10, 30], [50, 70]], support_intervals_s=[[[10, 11], [30, 31]], [[50, 51], [70, 71]]],
        support_valid=[[True, True], [True, False]], valid_tubelets=[True, True], temporal_selected_fraction=[1., 0.], nominal_time_s=[20, 60])
    for name, value in (("export_receipt.json", receipt), ("annotations.json", {"unit-video": dict(annotations=[dict(label="fixture", segment=[20, 25])])}), ("predictions.json", dict(results={"unit-video": []}))):
        (source / name).write_text(json.dumps(value), encoding="utf-8")
    (source / "windows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert viewer(source, tmp_path / "unit_viewer.html") == 1
    text = (tmp_path / "unit_viewer.html").read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in text
    assert "[10, 11], [30, 31]" in text
    report = Report(tmp_path / "unit_plots")
    plot_cases(report, [(source, receipt, [row])])
    plot_coverage(report, [(source, receipt, [row])])
    import csv
    with (report.output / "boundary_support_UNIT_FIXTURE_ONLY.csv").open(encoding="utf-8-sig") as stream:
        endpoints = list(csv.DictReader(stream))
    assert float(endpoints[0]["nearest_selected_support_seconds"]) == 9
    assert float(endpoints[1]["nearest_selected_support_seconds"]) == 5
