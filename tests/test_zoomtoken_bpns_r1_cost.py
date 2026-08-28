from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "bata" / "profile_zoomtoken_bpns_r1_cost.py"
SPEC = importlib.util.spec_from_file_location("zoomtoken_bpns_cost", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _reference_metrics(arm="K100"):
    return {
        key: value / 100.0
        for key, value in MODULE.EXPECTED_METRICS_PERCENT[arm].items()
    }


def test_counterbalanced_order_and_frozen_arms():
    assert MODULE.PROFILE_ORDER == (
        "K100",
        "R1",
        "R1",
        "K100",
        "R1",
        "K100",
        "K100",
        "R1",
    )
    assert MODULE.PROFILE_ORDER.count("K100") == 4
    assert MODULE.PROFILE_ORDER.count("R1") == 4
    assert MODULE.ARM_SPECS["K100"]["tokens_per_tubelet"] == 100
    assert MODULE.ARM_SPECS["R1"]["tokens_per_tubelet"] == 64
    assert MODULE.ARM_SPECS["R1"]["official_support"] == "strict_rect8x8"
    assert MODULE.EXPECTED_VIDEO_COUNT == 211
    assert MODULE.EXPECTED_WINDOW_COUNT == 792


def test_statistics_and_energy_are_numerically_defined():
    summary = MODULE.summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["p50"] == 2.5
    assert summary["p95"] == 3.8499999999999996
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0)]
    assert MODULE.integrate_energy(samples, start=0.25, end=1.75) == 150.0
    assert MODULE.integrate_energy(samples, start=-1.0, end=1.0) is None


def test_boundary_quality_separates_short_action_errors():
    annotation = {
        "database": {
            "v": {
                "subset": "validation",
                "annotations": [
                    {"label": "A", "segment": [0.0, 4.0]},
                    {"label": "A", "segment": [10.0, 20.0]},
                ],
            }
        }
    }
    predictions = {
        "v": [
            {"label": "A", "segment": [0.0, 4.0], "score": 0.9},
            {"label": "A", "segment": [11.0, 20.0], "score": 0.8},
        ]
    }
    result = MODULE.boundary_quality(annotation, predictions)
    assert result["matched_count"] == 2
    assert result["short_action"]["ground_truth_count"] == 1
    assert result["short_action"]["recall_at_tiou_0.70"] == 1.0
    assert result["mean_abs_start_error_normalized"] == 0.05


def test_launcher_preserves_slurm_and_result_blind_boundaries():
    launcher = (ROOT / "scripts" / "run_zoomtoken_bpns_r1_cost_n16r4.sh").read_text(
        encoding="utf-8"
    )
    assert "SLURM_JOB_ID" in launcher
    assert "SLURM_CPUS_PER_TASK" in launcher
    assert "SLURM_JOB_NAME" in launcher
    assert "v004" in launcher
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert "PRECHECK_ONLY" in launcher
    assert "profile_zoomtoken_bpns_r1_cost.py precheck" in launcher
    assert "epoch_59.pth" in launcher
    assert "tools/train.py" not in launcher
    assert "--resume" not in launcher


def test_power_sidecar_starts_before_detector_affinity_is_narrowed():
    source = SOURCE.read_text(encoding="utf-8")
    profile_source = source[source.index("def profile(") :]
    assert source.index("preflight = precheck(args)") < source.index("sidecar.start()")
    assert source.index("sidecar.start()") < source.index(
        "os.sched_setaffinity(0, set(detector_cpus))"
    )
    assert profile_source.index("_persist_acquisition_checkpoint(") < profile_source.index(
        "sidecar.stop()"
    )


def test_population_identity_preserves_official_duplicate_loader_items():
    class Dataset:
        data_list = [
            ["video_a", {}, {}, [0, 4]],
            ["video_a", {}, {}, [0, 4]],
        ]

    old_videos = MODULE.EXPECTED_VIDEO_COUNT
    old_windows = MODULE.EXPECTED_WINDOW_COUNT
    MODULE.EXPECTED_VIDEO_COUNT = 1
    MODULE.EXPECTED_WINDOW_COUNT = 2
    try:
        manifest, videos = MODULE._population_manifest(Dataset())
    finally:
        MODULE.EXPECTED_VIDEO_COUNT = old_videos
        MODULE.EXPECTED_WINDOW_COUNT = old_windows
    assert manifest == ["0:video_a:0", "1:video_a:0"]
    assert videos == {"video_a"}


def test_accuracy_parity_accepts_known_unrounded_replay_value():
    metrics = _reference_metrics()
    metrics["mAP@0.7"] = 0.46246663

    receipt = MODULE._assert_metric_parity("K100", metrics)

    row = receipt["metrics"]["mAP@0.7"]
    assert receipt["contract"]["reference"]["precision"] == "reported_2dp"
    assert row["observed_pp_unrounded"] == pytest.approx(46.246663)
    assert row["reference_pp"] == 46.27
    assert row["absolute_difference_pp"] == pytest.approx(0.023337)
    assert row["diagnosis"] == "compatible"


def test_accuracy_parity_tolerance_is_inclusive_and_distinguishing():
    metrics = _reference_metrics()
    metrics["average_mAP"] = 0.6856
    row = MODULE._assert_metric_parity("K100", metrics)["metrics"]["average_mAP"]
    assert row["diagnosis"] == "indeterminate"

    metrics["average_mAP"] = 0.68565001
    row = MODULE._assert_metric_parity("K100", metrics)["metrics"]["average_mAP"]
    assert row["diagnosis"] == "incompatible"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing required metric"),
        (float("nan"), "is not finite"),
        (float("inf"), "is not finite"),
    ],
)
def test_accuracy_parity_rejects_incomplete_or_nonfinite(mutation, message):
    metrics = _reference_metrics()
    if mutation == "missing":
        del metrics["mAP@0.4"]
    else:
        metrics["mAP@0.4"] = mutation

    with pytest.raises(RuntimeError, match=message):
        MODULE._assert_metric_parity("K100", metrics)


def test_accuracy_parity_display_rounding_is_decoupled_from_admission():
    metrics = _reference_metrics()
    metrics["mAP@0.7"] = 0.46246663

    row = MODULE._assert_metric_parity("K100", metrics)["metrics"]["mAP@0.7"]

    assert row["display_pp_2dp_half_up"] == "46.25"
    assert row["display_pp_2dp_half_up"] != f"{row['reference_pp']:.2f}"
    assert row["diagnosis"] == "compatible"


def test_v002_observation_is_nonblocking_indeterminate():
    metrics = _reference_metrics("R1")
    metrics["mAP@0.6"] = 0.6108696090294431
    receipt = MODULE._assert_metric_parity("R1", metrics)
    row = receipt["metrics"]["mAP@0.6"]
    assert row["diagnosis"] == "indeterminate"
    assert row["reference_interval_pp"] == {
        "lower_inclusive": 61.135,
        "upper_exclusive": 61.145,
    }
    assert row["minimum_distance_to_interval_pp"] == pytest.approx(0.04803909705569)


def test_out_of_interval_history_is_diagnostic_not_a_hard_gate():
    metrics = _reference_metrics("R1")
    metrics["mAP@0.6"] = 0.0
    receipt = MODULE._assert_metric_parity("R1", metrics)
    assert receipt["status"] == "NONBLOCKING_INCOMPATIBLE_PRESENT"
    assert receipt["metrics"]["mAP@0.6"]["diagnosis"] == "incompatible"


def test_power_coverage_rejects_nonfinite_and_incomplete_traces():
    summary = MODULE._power_coverage_summary(
        [(0.0, 100.0), (0.02, 101.0), (0.04, 99.0)],
        [(0.01, 0.03)],
    )
    assert summary["status"] == "COMPLETE"
    assert summary["max_trace_gap_ms"] == pytest.approx(20.0)
    assert summary["measurement_coverage_ratio"] == 1.0
    with pytest.raises(ValueError, match="non-finite"):
        MODULE._power_coverage_summary([(0.0, 100.0), (0.02, float("nan"))], [(0.0, 0.01)])
    with pytest.raises(RuntimeError, match="does not fully cover"):
        MODULE._power_coverage_summary([(0.02, 100.0), (0.04, 100.0)], [(0.01, 0.03)])


def test_missing_cost_row_invalidates_a_pass():
    rows = [
        {
            "arm": "K100",
            "pass_index": 0,
            "end_to_end_serial_ms": 1.0,
            "gpu_energy_j": 1.0,
            "peak_gpu_allocated_mb": 1.0,
            "peak_gpu_reserved_mb": 1.0,
        }
    ]
    receipt = {
        "arm": "K100",
        "pass_index": 0,
        "metrics": _reference_metrics(),
        "accuracy_parity": MODULE._assert_metric_parity("K100", _reference_metrics()),
        "power_coverage": {"status": "COMPLETE"},
    }
    with pytest.raises(ValueError, match="expected 2"):
        MODULE._summarize_pass(rows, receipt, ["end_to_end_serial_ms"], expected_window_count=2)


def test_every_pass_persists_predictions_and_raw_evaluator_vector(tmp_path):
    receipts = []
    predictions = []
    for pass_index, arm in enumerate(MODULE.PROFILE_ORDER):
        receipts.append({"pass_index": pass_index, "arm": arm, "metrics": _reference_metrics(arm)})
        predictions.append({"results": {"v": [{"score": pass_index}]}})
    artifacts = MODULE._persist_pass_artifacts(tmp_path, receipts, predictions)
    assert len(artifacts) == 8
    assert len({row["prediction_path"] for row in artifacts}) == 8
    assert all(Path(row["prediction_path"]).is_file() for row in artifacts)
    assert all(len(row["prediction_sha256"]) == 64 for row in artifacts)
    assert all(Path(row["evaluator_raw_vector_path"]).is_file() for row in artifacts)


def test_production_short_action_config_merge_keeps_registry_type(tmp_path):
    class Config:
        evaluation = {
            "type": "mAP",
            "subset": "training",
            "tiou_thresholds": [0.3, 0.7],
            "thread": 1,
        }

    payload = MODULE._short_action_evaluator_config(
        Config(),
        predictions={"results": {}},
        ground_truth_filename=tmp_path / "short.json",
    )
    assert payload["type"] == "mAP"
    assert payload["thread"] == 1
    assert payload["subset"] == "validation"
    assert payload["prediction_filename"] == {"results": {}}


def test_actual_production_evaluator_factory_executes_known_answer(tmp_path):
    if sys.platform == "win32":
        pytest.skip("the target production factory test requires the N16R4 Torch runtime")
    from mmengine.config import Config

    cfg = Config.fromfile(str(ROOT / MODULE.ARM_SPECS["K100"]["config"]))
    result = MODULE._short_action_evaluator_known_answer(cfg, tmp_path)
    assert result["status"] == "KNOWN_ANSWER_PASS"
    assert result["registry_type"] == "mAP"
    assert result["scientific_result"] is False


def test_all_pass_prediction_sha_checks_use_frozen_v003_anchors(tmp_path, monkeypatch):
    def frozen_sha(path):
        name = Path(path).name
        if "predictions" in name:
            arm = "K100" if "_k100_" in name else "R1"
            return MODULE.FROZEN_PREDICTION_SHA256[arm]
        return "e" * 64

    monkeypatch.setattr(MODULE, "_sha256_file", frozen_sha)
    receipts = [
        {"pass_index": pass_index, "arm": arm, "metrics": _reference_metrics(arm)}
        for pass_index, arm in enumerate(MODULE.PROFILE_ORDER)
    ]
    predictions = [{"results": {}} for _ in MODULE.PROFILE_ORDER]
    artifacts = MODULE._persist_pass_artifacts(tmp_path, receipts, predictions)
    assert len(artifacts) == 8
    assert all(row["prediction_identity_match"] for row in artifacts)
    assert [row["expected_prediction_sha256"] for row in artifacts] == [
        MODULE.FROZEN_PREDICTION_SHA256[arm] for arm in MODULE.PROFILE_ORDER
    ]


def test_primary_estimator_is_median_of_four_passes():
    passes = []
    for pass_index, value in enumerate((1.0, 2.0, 100.0, 4.0)):
        passes.append(
            {
                "arm": "K100",
                "pass_index": pass_index,
                "latency_ms": {"end_to_end_serial_ms": {name: value for name in ("mean", "p50", "p95", "min", "max")}},
                "throughput_windows_per_second": value,
                "peak_gpu_allocated_mb": value,
                "peak_gpu_reserved_mb": value,
                "gross_gpu_energy_j": value,
                "gpu_energy_j_per_window": value,
                "final_ema_metrics": _reference_metrics(),
            }
        )
    summary = MODULE._median_of_four_arm_summary(passes, "K100", ["end_to_end_serial_ms"])
    assert summary["primary_estimator"] == "median_of_four_pass_estimates"
    assert summary["latency_ms"]["end_to_end_serial_ms"]["p50"] == 3.0
    assert summary["gpu_energy_j_per_window"] == 3.0


def test_pooled_rows_cannot_override_median_four_primary_estimator():
    passes = []
    for pass_index, value in enumerate((1.0, 2.0, 100.0, 4.0)):
        passes.append(
            {
                "arm": "R1",
                "pass_index": pass_index,
                "latency_ms": {"end_to_end_serial_ms": {name: value for name in ("mean", "p50", "p95", "min", "max")}},
                "throughput_windows_per_second": value,
                "peak_gpu_allocated_mb": value,
                "peak_gpu_reserved_mb": value,
                "gross_gpu_energy_j": value,
                "gpu_energy_j_per_window": value,
                "final_ema_metrics": _reference_metrics("R1"),
            }
        )
    summary = MODULE._median_of_four_arm_summary(passes, "R1", ["end_to_end_serial_ms"])
    pooled_p50 = MODULE.percentile([1.0] * 792 + [100.0] * (3 * 792), 0.5)
    assert pooled_p50 == 100.0
    assert summary["latency_ms"]["end_to_end_serial_ms"]["p50"] == 3.0


def test_controlled_failure_writes_terminal_receipt(tmp_path):
    args = argparse.Namespace(
        command="precheck",
        result_root=tmp_path / "formal_v003",
        expected_commit="a" * 40,
    )
    path = MODULE._write_failure_terminal_receipt(args, RuntimeError("identity mismatch"))
    payload = path.read_text(encoding="utf-8")
    assert '"status": "FAILED_PROTOCOL_INVALID"' in payload
    assert '"error_message": "identity mismatch"' in payload
    assert '"cost_acquisition_complete": false' in payload


def test_diagnostic_failure_preserves_complete_raw_acquisition(tmp_path):
    result_root = tmp_path / "formal_v004"
    result_root.mkdir()
    (result_root / "cost_samples.jsonl").write_text("{}\n", encoding="utf-8")
    (result_root / "power_trace.jsonl").write_text("{}\n", encoding="utf-8")
    (result_root / "acquisition_state.json").write_text(
        json.dumps(
            {
                "prediction_identity_complete": True,
                "cost_acquisition_complete": True,
                "diagnostic_complete": False,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(command="profile", result_root=result_root, expected_commit="b" * 40)
    path = MODULE._write_failure_terminal_receipt(args, RuntimeError("diagnostic failed"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED_DIAGNOSTIC_WITH_COMPLETE_RAW_COST"
    assert payload["cost_acquisition_complete"] is True
    assert payload["diagnostic_complete"] is False
    assert {"cost_samples.jsonl", "power_trace.jsonl"}.issubset(payload["produced_artifacts"])


def test_measurement_completeness_requires_eight_identity_bound_passes():
    receipts = []
    for pass_index, arm in enumerate(MODULE.PROFILE_ORDER):
        receipts.append(
            {
                "pass_index": pass_index,
                "arm": arm,
                "artifacts": {"prediction_identity_match": True},
                "power_coverage": {
                    "status": "COMPLETE",
                    "max_trace_gap_ms": 20.0,
                    "measurement_coverage_ratio": 1.0,
                },
                "cost_summary": {
                    "sample_count": MODULE.EXPECTED_WINDOW_COUNT,
                    "latency_ms": {"end_to_end_serial_ms": {"p50": 1.0, "p95": 2.0}},
                    "throughput_windows_per_second": 10.0,
                    "peak_gpu_allocated_mb": 100.0,
                    "peak_gpu_reserved_mb": 120.0,
                    "gross_gpu_energy_j": 50.0,
                },
            }
        )
    result = MODULE._measurement_completeness(receipts)
    assert result["status"] == "COMPLETE"
    assert result["pass_count"] == 8
    assert result["population_items_total"] == 8 * 792
    assert result["final_video_nms_amortized_across_window_rows"] is True


def test_complete_replay_acceptance_uses_registered_pass_level_rules():
    latency_keys = ["end_to_end_serial_ms"]
    pass_summaries = []
    for arm, p50, energy, accuracy in (
        ("K100", 100.0, 100.0, _reference_metrics("K100")),
        ("R1", 90.0, 90.0, _reference_metrics("R1")),
    ):
        for pass_index in range(4):
            pass_summaries.append(
                {
                    "arm": arm,
                    "pass_index": pass_index,
                    "latency_ms": {
                        "end_to_end_serial_ms": {
                            "mean": p50,
                            "p50": p50,
                            "p95": p50,
                            "min": p50,
                            "max": p50,
                        }
                    },
                    "throughput_windows_per_second": 1000.0 / p50,
                    "peak_gpu_allocated_mb": p50,
                    "peak_gpu_reserved_mb": p50,
                    "gross_gpu_energy_j": energy,
                    "gpu_energy_j_per_window": energy / MODULE.EXPECTED_WINDOW_COUNT,
                    "final_ema_metrics": accuracy,
                }
            )
    arm_summaries = {
        arm: MODULE._median_of_four_arm_summary(pass_summaries, arm, latency_keys)
        for arm in ("K100", "R1")
    }
    stable = {
        arm: {"all_four_hashes_identical": True}
        for arm in ("K100", "R1")
    }
    pass_quality = [
        {
            "arm": arm,
            "pass_index": pass_index,
            "boundary": {
                "mean_abs_start_error_normalized": 0.1,
                "mean_abs_end_error_normalized": 0.1,
                "short_action": {"recall_at_tiou_0.70": 0.5},
            },
        }
        for arm in ("K100", "R1")
        for pass_index in range(4)
    ]
    result = MODULE._classify_complete_replay(pass_summaries, arm_summaries, stable, pass_quality)
    assert result["decision"] == "ACCEPT_FOR_RESULT_TO_CLAIM_REVIEW"
    assert result["p50_reduction"] == pytest.approx(0.10)
    assert result["gross_energy_per_pass_reduction"] == pytest.approx(0.10)


def test_quality_stop_requires_complete_adverse_pass_range_separation():
    latency_keys = ["end_to_end_serial_ms"]
    pass_summaries = []
    for arm, p50, energy, accuracy in (
        ("K100", 100.0, 100.0, _reference_metrics("K100")),
        ("R1", 90.0, 90.0, _reference_metrics("R1")),
    ):
        for pass_index in range(4):
            pass_summaries.append(
                {
                    "arm": arm,
                    "pass_index": pass_index,
                    "latency_ms": {"end_to_end_serial_ms": {name: p50 for name in ("mean", "p50", "p95", "min", "max")}},
                    "throughput_windows_per_second": 1000.0 / p50,
                    "peak_gpu_allocated_mb": p50,
                    "peak_gpu_reserved_mb": p50,
                    "gross_gpu_energy_j": energy,
                    "gpu_energy_j_per_window": energy / MODULE.EXPECTED_WINDOW_COUNT,
                    "final_ema_metrics": accuracy,
                }
            )
    arm_summaries = {
        arm: MODULE._median_of_four_arm_summary(pass_summaries, arm, latency_keys)
        for arm in ("K100", "R1")
    }
    varied_hashes = {
        "K100": {"all_four_hashes_identical": False},
        "R1": {"all_four_hashes_identical": False},
    }
    pass_quality = []
    for arm, starts, recalls in (
        ("K100", (0.10, 0.10, 0.10, 0.10), (0.50, 0.50, 0.50, 0.50)),
        ("R1", (0.09, 0.11, 0.09, 0.11), (0.49, 0.51, 0.49, 0.51)),
    ):
        for pass_index, (start, recall) in enumerate(zip(starts, recalls)):
            pass_quality.append(
                {
                    "arm": arm,
                    "pass_index": pass_index,
                    "boundary": {
                        "mean_abs_start_error_normalized": start,
                        "mean_abs_end_error_normalized": start,
                        "short_action": {"recall_at_tiou_0.70": recall},
                    },
                }
            )
    mixed = MODULE._classify_complete_replay(
        pass_summaries, arm_summaries, varied_hashes, pass_quality
    )
    assert mixed["decision"] == "ACCEPT_FOR_RESULT_TO_CLAIM_REVIEW"
    assert mixed["quality_reversals"] == []
    assert mixed["prediction_hash_variation_diagnostic"] is True

    for row in pass_quality:
        if row["arm"] == "R1":
            row["boundary"]["mean_abs_start_error_normalized"] = 0.20
            row["boundary"]["mean_abs_end_error_normalized"] = 0.20
            row["boundary"]["short_action"]["recall_at_tiou_0.70"] = 0.40
    separated = MODULE._classify_complete_replay(
        pass_summaries, arm_summaries, varied_hashes, pass_quality
    )
    assert separated["decision"] == "TARGETED_CLAIM_REVISION_REQUIRED"
    assert set(separated["quality_reversals"]) == {
        "short_action_recall_at_tiou_0.70",
        "overall_boundary_mean_abs_start_error_normalized",
        "overall_boundary_mean_abs_end_error_normalized",
    }
