from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "bata" / "profile_zoomtoken_r1_depth_pareto_cost.py"
SPEC = importlib.util.spec_from_file_location("zoomtoken_r1_depth_pareto_cost", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _reference_metrics(arm: str = "A") -> dict[str, float]:
    return {
        key: value / 100.0
        for key, value in MODULE.EXPECTED_METRICS_PERCENT[arm].items()
    }


def _cost_summary(value: float) -> dict:
    return {
        "sample_count": MODULE.EXPECTED_WINDOW_COUNT,
        "latency_ms": {
            "end_to_end_serial_ms": {
                "mean": value,
                "p50": value,
                "p95": value,
                "min": value,
                "max": value,
            }
        },
        "throughput_windows_per_second": 1000.0 / value,
        "peak_gpu_allocated_mb": value,
        "peak_gpu_reserved_mb": value,
        "gross_gpu_energy_j": value,
        "gpu_energy_j_per_window": value / MODULE.EXPECTED_WINDOW_COUNT,
        "final_ema_metrics": _reference_metrics(),
    }


def _arm_summary(value: float, *, allocated: float | None = None, reserved: float | None = None) -> dict:
    return {
        "latency_ms": {"end_to_end_serial_ms": {"p50": value}},
        "gross_gpu_energy_j": value,
        "peak_gpu_allocated_mb": value if allocated is None else allocated,
        "peak_gpu_reserved_mb": value if reserved is None else reserved,
    }


def test_williams_order_and_four_frozen_arms():
    assert MODULE.PROFILE_ORDER == (
        "A", "B", "D", "C",
        "B", "C", "A", "D",
        "C", "D", "B", "A",
        "D", "A", "C", "B",
    )
    assert MODULE.ARM_KEYS == ("A", "B", "C", "D")
    assert all(MODULE.PROFILE_ORDER.count(arm) == 4 for arm in MODULE.ARM_KEYS)
    assert MODULE.ARM_SPECS["A"]["label"] == "R1/FULL64"
    assert MODULE.ARM_SPECS["B"]["label"] == "DSR6-KV"
    assert MODULE.ARM_SPECS["C"]["label"] == "MOD32-KV"
    assert MODULE.ARM_SPECS["D"]["label"] == "DROP32"
    assert MODULE.ARM_SPECS["D"]["selected_tokens_per_tubelet"] == 32
    assert MODULE.ARM_SPECS["D"]["executed_tokens_per_window"] == 12288
    assert all(len(MODULE.ARM_SPECS[arm]["checkpoint_sha256"]) == 64 for arm in MODULE.ARM_KEYS)
    assert MODULE.EXPECTED_VIDEO_COUNT == 211
    assert MODULE.EXPECTED_WINDOW_COUNT == 792


def test_runtime_audit_contract_distinguishes_depth_and_drop_modes():
    expected_a = MODULE._expected_runtime_audit("A")
    expected_b = MODULE._expected_runtime_audit("B")
    expected_c = MODULE._expected_runtime_audit("C")
    expected_d = MODULE._expected_runtime_audit("D")
    assert expected_a["strict_kv_context_tokens_per_tubelet"] == 64
    assert expected_b["strict_kv_context_tokens_per_tubelet"] == 64
    assert expected_c["strict_kv_context_tokens_per_tubelet"] == 64
    assert expected_d["strict_kv_context_tokens_per_tubelet"] == 32
    assert expected_b["full_update_block_count"] == 6
    assert expected_b["refresh_update_block_count"] == 6
    assert expected_d["executed_patch_tokens_per_window"] == 12288


def test_statistics_and_energy_are_numerically_defined():
    summary = MODULE.summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
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


def test_launcher_preserves_slurm_and_readonly_boundaries():
    launcher = (
        ROOT / "scripts" / "run_zoomtoken_r1_depth_pareto_cost_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert "SLURM_JOB_ID" in launcher
    assert "SLURM_CPUS_PER_TASK" in launcher
    assert "SLURM_JOB_NAME" in launcher
    assert "r1_depth_pareto" in launcher
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert "PRECHECK_ONLY" in launcher
    assert "profile_zoomtoken_r1_depth_pareto_cost.py precheck" in launcher
    assert all(f"--{arm.lower()}-checkpoint" in launcher for arm in MODULE.ARM_KEYS)
    assert launcher.count("epoch_59.pth") == 4
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


def test_historical_accuracy_is_nonblocking_diagnostic():
    metrics = _reference_metrics("A")
    metrics["mAP@0.7"] = 0.46546663
    receipt = MODULE._assert_metric_parity("A", metrics)
    row = receipt["metrics"]["mAP@0.7"]
    assert receipt["contract"]["reference"]["precision"] == "reported_2dp"
    assert row["display_pp_2dp_half_up"] == "46.55"
    assert row["diagnosis"] == "compatible"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_historical_accuracy_rejects_nonfinite(bad):
    metrics = _reference_metrics("B")
    metrics["average_mAP"] = bad
    with pytest.raises(RuntimeError, match="not finite"):
        MODULE._assert_metric_parity("B", metrics)


def test_historical_accuracy_rejects_missing_metric():
    metrics = _reference_metrics("C")
    del metrics["mAP@0.6"]
    with pytest.raises(RuntimeError, match="missing required metric"):
        MODULE._assert_metric_parity("C", metrics)


def test_power_coverage_applies_frozen_gap_and_validity_gates():
    summary = MODULE._power_coverage_summary(
        [(0.0, 100.0), (0.02, 101.0), (0.04, 99.0)],
        [(0.01, 0.03)],
    )
    assert summary["status"] == "COMPLETE"
    assert summary["invalid_sample_count"] == 0
    assert summary["max_trace_gap_ms"] == pytest.approx(20.0)
    assert summary["uncovered_fraction"] == 0.0
    with pytest.raises(ValueError, match="non-finite"):
        MODULE._power_coverage_summary(
            [(0.0, 100.0), (0.02, float("nan"))], [(0.0, 0.01)]
        )
    with pytest.raises(RuntimeError, match="maximum gap"):
        MODULE._power_coverage_summary(
            [(0.0, 100.0), (3.1, 100.0), (4.0, 100.0)], [(0.0, 4.0)]
        )
    with pytest.raises(RuntimeError, match="does not fully cover"):
        MODULE._power_coverage_summary(
            [(0.02, 100.0), (0.04, 100.0)], [(0.01, 0.03)]
        )


def test_missing_cost_row_invalidates_a_pass():
    rows = [
        {
            "arm": "A",
            "pass_index": 0,
            "end_to_end_serial_ms": 1.0,
            "gpu_energy_j": 1.0,
            "peak_gpu_allocated_mb": 1.0,
            "peak_gpu_reserved_mb": 1.0,
        }
    ]
    receipt = {
        "arm": "A",
        "pass_index": 0,
        "metrics": _reference_metrics("A"),
        "accuracy_parity": MODULE._assert_metric_parity("A", _reference_metrics("A")),
        "power_coverage": {"status": "COMPLETE"},
    }
    with pytest.raises(ValueError, match="expected 2"):
        MODULE._summarize_pass(
            rows, receipt, ["end_to_end_serial_ms"], expected_window_count=2
        )


def test_every_pass_persists_prediction_and_raw_evaluator_vector(tmp_path):
    receipts = []
    predictions = []
    for pass_index, arm in enumerate(MODULE.PROFILE_ORDER):
        receipts.append(
            {"pass_index": pass_index, "arm": arm, "metrics": _reference_metrics(arm)}
        )
        predictions.append({"results": {"v": [{"score": ord(arm)}]}})
    artifacts = MODULE._persist_pass_artifacts(tmp_path, receipts, predictions)
    assert len(artifacts) == 16
    assert len({row["prediction_path"] for row in artifacts}) == 16
    assert all(Path(row["prediction_path"]).is_file() for row in artifacts)
    assert all(len(row["prediction_sha256"]) == 64 for row in artifacts)
    assert all(Path(row["evaluator_raw_vector_path"]).is_file() for row in artifacts)
    for arm in MODULE.ARM_KEYS:
        assert len({row["prediction_sha256"] for row in artifacts if row["arm"] == arm}) == 1


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
        pytest.skip("the production evaluator check requires the N16R4 runtime")
    from mmengine.config import Config

    cfg = Config.fromfile(str(ROOT / MODULE.ARM_SPECS["A"]["config"]))
    result = MODULE._short_action_evaluator_known_answer(cfg, tmp_path)
    assert result["status"] == "KNOWN_ANSWER_PASS"
    assert result["registry_type"] == "mAP"
    assert result["scientific_result"] is False


def test_primary_estimator_is_median_of_four_complete_passes():
    passes = []
    for pass_index, value in enumerate((1.0, 2.0, 100.0, 4.0)):
        row = _cost_summary(value)
        row.update({"arm": "A", "pass_index": pass_index})
        passes.append(row)
    summary = MODULE._median_of_four_arm_summary(
        passes, "A", ["end_to_end_serial_ms"]
    )
    assert summary["primary_estimator"] == "median_of_four_pass_estimates"
    assert summary["latency_ms"]["end_to_end_serial_ms"]["p50"] == 3.0
    assert summary["gross_gpu_energy_j"] == 3.0


def _complete_receipts() -> list[dict]:
    hashes = {arm: arm.lower() * 64 for arm in MODULE.ARM_KEYS}
    receipts = []
    for pass_index, arm in enumerate(MODULE.PROFILE_ORDER):
        receipts.append(
            {
                "pass_index": pass_index,
                "arm": arm,
                "artifacts": {"prediction_sha256": hashes[arm]},
                "power_coverage": {
                    "status": "COMPLETE",
                    "invalid_sample_count": 0,
                    "max_trace_gap_ms": 20.0,
                    "uncovered_fraction": 0.0,
                },
                "cost_summary": _cost_summary(1.0),
            }
        )
    return receipts


def test_measurement_completeness_requires_sixteen_identity_bound_passes():
    result = MODULE._measurement_completeness(_complete_receipts())
    assert result["status"] == "COMPLETE"
    assert result["pass_count"] == 16
    assert result["population_items_total"] == 16 * 792
    assert result["final_video_nms_amortized_across_window_rows"] is True
    assert set(result["within_arm_prediction_sha_stability"]) == set(MODULE.ARM_KEYS)


def test_measurement_completeness_rejects_within_arm_prediction_drift():
    receipts = _complete_receipts()
    second_a = [index for index, arm in enumerate(MODULE.PROFILE_ORDER) if arm == "A"][1]
    receipts[second_a]["artifacts"]["prediction_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="not identical"):
        MODULE._measurement_completeness(receipts)


def test_pareto_classifier_records_each_survivor_independently():
    arm_summaries = {
        "A": _arm_summary(100.0),
        "B": _arm_summary(94.0, allocated=104.0, reserved=105.0),
        "C": _arm_summary(96.0),
        "D": _arm_summary(90.0, allocated=106.0, reserved=90.0),
    }
    stable = {
        arm: {"all_four_hashes_identical": True} for arm in MODULE.ARM_KEYS
    }
    result = MODULE._classify_complete_replay([], arm_summaries, stable, [])
    assert result["decision"] == "PARETO_SYSTEMS_SURVIVOR_PENDING_FRESH_PRO"
    assert result["survivor_arms"] == ["B"]
    assert result["candidate_ratios"]["B"]["passes_all_gates"] is True
    assert result["candidate_ratios"]["C"]["passes_all_gates"] is False
    assert result["candidate_ratios"]["D"]["passes_all_gates"] is False


def test_pareto_classifier_stops_family_when_no_arm_survives():
    arm_summaries = {
        "A": _arm_summary(100.0),
        "B": _arm_summary(96.0),
        "C": _arm_summary(100.0),
        "D": _arm_summary(110.0),
    }
    stable = {
        arm: {"all_four_hashes_identical": True} for arm in MODULE.ARM_KEYS
    }
    result = MODULE._classify_complete_replay([], arm_summaries, stable, [])
    assert result["decision"] == (
        "STOP_R1_FIXED_DEPTH_SPARSITY_FOURPOINT_AS_CURRENT_EFFICIENCY_ROUTE"
    )
    assert result["survivor_arms"] == []


def test_controlled_failure_writes_no_scientific_decision_receipt(tmp_path):
    args = argparse.Namespace(
        command="precheck",
        result_root=tmp_path / "formal_depth_pareto",
        expected_commit="a" * 40,
    )
    path = MODULE._write_failure_terminal_receipt(args, RuntimeError("identity mismatch"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKER_NO_SCIENTIFIC_COST_DECISION"
    assert payload["blocker_phase"] == "precheck"
    assert payload["error_message"] == "identity mismatch"
    assert payload["cost_acquisition_complete"] is False


def test_post_acquisition_failure_preserves_complete_raw_acquisition(tmp_path):
    result_root = tmp_path / "formal_depth_pareto"
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
    args = argparse.Namespace(
        command="profile", result_root=result_root, expected_commit="b" * 40
    )
    path = MODULE._write_failure_terminal_receipt(args, RuntimeError("diagnostic failed"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKER_NO_SCIENTIFIC_COST_DECISION"
    assert payload["blocker_phase"] == "post_acquisition_diagnostic"
    assert payload["cost_acquisition_complete"] is True
    assert payload["diagnostic_complete"] is False
    assert {"cost_samples.jsonl", "power_trace.jsonl"}.issubset(
        payload["produced_artifacts"]
    )
