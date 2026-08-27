from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "bata" / "profile_zoomtoken_bpns_r1_cost.py"
SPEC = importlib.util.spec_from_file_location("zoomtoken_bpns_cost", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


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
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert "PRECHECK_ONLY" in launcher
    assert "profile_zoomtoken_bpns_r1_cost.py precheck" in launcher
    assert "epoch_59.pth" in launcher
    assert "tools/train.py" not in launcher
    assert "--resume" not in launcher


def test_power_sidecar_starts_before_detector_affinity_is_narrowed():
    source = SOURCE.read_text(encoding="utf-8")
    assert source.index("sidecar.start()") < source.index(
        "os.sched_setaffinity(0, set(detector_cpus))"
    )
