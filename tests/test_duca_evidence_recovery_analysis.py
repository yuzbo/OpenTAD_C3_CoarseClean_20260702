from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "tools" / "bata" / "analyze_duca_evidence_recovery.py"
ARMS = ["C0", "F", "A1", "A2", "A3", "A4", "A5", "A6"]


def _write_metric_cell(root: Path, arm: str, seed: int, avg_map: float, *, video_count: int = 5) -> None:
    eval_dir = root / arm / f"seed_{seed}" / "official_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    video_metrics = {
        f"video_{idx:03d}": {
            "average_mAP": avg_map + idx * 0.01,
            "mAP@0.70": avg_map - 18.0 + idx * 0.01,
        }
        for idx in range(video_count)
    }
    payload = {
        "metrics": {
            "average_mAP": avg_map,
            "mAP@0.70": avg_map - 18.0,
        },
        "video_metrics": video_metrics,
        "video_mAP": {key: row["average_mAP"] for key, row in video_metrics.items()},
        "video_mAP@0.70": {key: row["mAP@0.70"] for key, row in video_metrics.items()},
    }
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_profiles(root: Path) -> None:
    profile_dir = root / "cost_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        "C0": (10.0, 12.0, 1000.0),
        "F": (8.5, 12.3, 950.0),
        "A4": (10.0, 12.4, 980.0),
        "A5": (8.8, 12.1, 940.0),
    }
    for arm, (p50, p95, mem) in profiles.items():
        with (profile_dir / f"profile_{arm}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "duca_evidence_recovery_profile_v1",
                    "profile_complete": True,
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "peak_memory_allocated_mb": mem,
                },
                f,
            )


def _write_matrix(root: Path, seed: int = 8261) -> None:
    base_maps = {
        "C0": 65.13,
        "F": 67.25,
        "A1": 65.80,
        "A2": 66.10,
        "A3": 66.40,
        "A4": 66.90,
        "A5": 65.95,
        "A6": 65.50,
    }
    for arm in ARMS:
        _write_metric_cell(root, arm, seed, base_maps[arm])


def test_analyzer_fails_when_cost_profile_is_missing(tmp_path):
    run_root = tmp_path / "run"
    _write_matrix(run_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--run-root",
            str(run_root),
            "--output",
            str(tmp_path / "analysis.json"),
            "--seeds",
            "8261",
            "--expected-video-count",
            "5",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "cost_profile" in result.stderr


def test_analyzer_uses_video_metrics_and_required_profiles(tmp_path):
    run_root = tmp_path / "run"
    out_path = tmp_path / "analysis.json"
    _write_matrix(run_root)
    _write_profiles(run_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--run-root",
            str(run_root),
            "--output",
            str(out_path),
            "--seeds",
            "8261",
            "--expected-video-count",
            "5",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    with out_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)
    assert analysis["comparisons"]["FULL_vs_C0"]["bootstrap_mode"] == "hierarchical_seeds_and_videos"
    assert analysis["decision_gates"]["gate8_matrix_completeness"]["video_identity_count"] == 5
    assert analysis["all_gates_passed"] is True
