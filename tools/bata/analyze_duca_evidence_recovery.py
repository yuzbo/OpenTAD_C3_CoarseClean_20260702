"""Hierarchical paired bootstrap analysis and decision gates for DUCA Evidence Recovery."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARM_NAMES = ["C0", "F", "A1", "A2", "A3", "A4", "A5", "A6"]
SEEDS = [8261, 19237, 31153]
EXPECTED_VIDEO_COUNT = 211
REQUIRED_PROFILE_ARMS = ["C0", "F", "A4", "A5"]

COMPARISONS = {
    "FULL_vs_C0": ("F", "C0", "Primary comparison: FULL vs MATCHED_H65_60"),
    "FULL_vs_NO_COVERAGE": ("F", "A1", "Coverage necessity: FULL vs NO_COVERAGE"),
    "FULL_vs_NO_TIME": ("F", "A2", "Time conditioning efficacy: FULL vs NO_TIME"),
    "FULL_vs_NO_ROBUST": ("F", "A3", "Robust training efficacy: FULL vs NO_ROBUST"),
    "FULL_vs_NO_MERGE": ("F", "A4", "Token merge efficacy: FULL vs NO_MERGE"),
    "FULL_vs_NO_RECOVERY": ("F", "A5", "Recovery layer efficacy: FULL vs NO_RECOVERY"),
    "FULL_vs_H65_SELECTION": ("F", "A6", "Semantic allocation vs H65: FULL vs H65_SELECTION"),
}

DENSE_REFERENCE_AVG_MAP = 68.73


def _as_percent(value: Any) -> float:
    result = float(value)
    if abs(result) <= 1.0:
        result *= 100.0
    return result


def _lookup_metric(metrics: Dict[str, Any], data: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key in metrics:
            return _as_percent(metrics[key])
        if key in data:
            return _as_percent(data[key])
    return None


def _video_metric_map(data: Dict[str, Any], metric_keys: Tuple[str, ...], direct_keys: Tuple[str, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    video_metrics = data.get("video_metrics", data.get("per_video_metrics", {}))
    if isinstance(video_metrics, dict):
        for video_id, row in video_metrics.items():
            value = None
            if isinstance(row, dict):
                for key in metric_keys:
                    if key in row:
                        value = row[key]
                        break
            elif "average_mAP" in metric_keys or "mAP" in metric_keys:
                value = row
            if value is not None:
                out[str(video_id)] = _as_percent(value)

    for direct_key in direct_keys:
        direct = data.get(direct_key)
        if isinstance(direct, dict):
            for video_id, value in direct.items():
                if isinstance(value, dict):
                    nested = None
                    for key in metric_keys:
                        if key in value:
                            nested = value[key]
                            break
                    if nested is not None:
                        out[str(video_id)] = _as_percent(nested)
                else:
                    out[str(video_id)] = _as_percent(value)
            break
    return out


def parse_eval_results(eval_dir: Path) -> Dict[str, Any]:
    """Parse one official evaluation directory with mandatory structured metrics."""
    target_files = []
    metrics_json = eval_dir / "metrics.json"
    if metrics_json.is_file():
        target_files.append(metrics_json)
    target_files.extend(path for path in sorted(eval_dir.glob("*.json")) if path not in target_files)

    for result_file in target_files:
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        metrics = data.get("metrics", data)
        if not isinstance(metrics, dict):
            continue
        avg_map = _lookup_metric(metrics, data, ("average_mAP", "mAP_avg", "Average_mAP", "mAP"))
        if avg_map is None:
            continue
        map_70 = _lookup_metric(metrics, data, ("mAP@0.70", "mAP@0.7"))
        video_map = _video_metric_map(
            data,
            ("average_mAP", "mAP_avg", "Average_mAP", "mAP"),
            ("video_mAP", "per_video_mAP"),
        )
        video_map_70 = _video_metric_map(
            data,
            ("mAP@0.70", "mAP@0.7"),
            ("video_mAP@0.70", "video_mAP@0.7", "per_video_mAP@0.70"),
        )
        return {
            "average_mAP": avg_map,
            "mAP@0.70": 0.0 if map_70 is None else map_70,
            "video_mAP": video_map,
            "video_mAP@0.70": video_map_70,
            "metrics": metrics,
            "source_file": str(result_file),
        }
    return {}


def paired_bootstrap_analysis(
    arm_a_seed_values: Dict[int, float],
    arm_b_seed_values: Dict[int, float],
    arm_a_video_maps: Optional[Dict[int, Dict[str, float]]] = None,
    arm_b_video_maps: Optional[Dict[int, Dict[str, float]]] = None,
    seeds: List[int] = SEEDS,
    num_replicates: int = 10000,
    rng_seed: int = 20260901,
) -> Dict[str, Any]:
    """Hierarchical paired bootstrap over seeds and video identities."""
    rng = np.random.default_rng(rng_seed)
    has_full_video_maps = False
    if arm_a_video_maps and arm_b_video_maps:
        has_full_video_maps = all(
            bool(arm_a_video_maps.get(seed)) and bool(arm_b_video_maps.get(seed))
            for seed in seeds
        )

    if has_full_video_maps:
        seed_diff_arrays = {}
        for seed in seeds:
            v_a = arm_a_video_maps[seed]
            v_b = arm_b_video_maps[seed]
            common_vids = sorted(set(v_a.keys()) & set(v_b.keys()))
            if not common_vids:
                raise RuntimeError(f"no common video identities for seed {seed}")
            seed_diff_arrays[seed] = np.array([v_a[vid] - v_b[vid] for vid in common_vids], dtype=np.float64)

        replicate_diffs = []
        for _ in range(num_replicates):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_means = []
            for seed in sampled_seeds:
                diffs = seed_diff_arrays[int(seed)]
                idx = rng.integers(0, len(diffs), size=len(diffs))
                seed_means.append(float(np.mean(diffs[idx])))
            replicate_diffs.append(float(np.mean(seed_means)))
        bootstrap_mode = "hierarchical_seeds_and_videos"
    else:
        paired_diffs = np.array([arm_a_seed_values[seed] - arm_b_seed_values[seed] for seed in seeds], dtype=np.float64)
        replicate_diffs = []
        for _ in range(num_replicates):
            idx = rng.choice(len(paired_diffs), size=len(paired_diffs), replace=True)
            replicate_diffs.append(float(np.mean(paired_diffs[idx])))
        bootstrap_mode = "paired_seeds"

    arr = np.array(replicate_diffs, dtype=np.float64)
    return {
        "mean_diff": round(float(np.mean(arr)), 4),
        "ci_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        "p_value": round(float(np.mean(arr <= 0.0)), 5),
        "bootstrap_mode": bootstrap_mode,
        "num_replicates": int(num_replicates),
    }


def compute_dense_gap_closure(
    full_map: float,
    baseline_map: float,
    dense_ref: float = DENSE_REFERENCE_AVG_MAP,
) -> float:
    """Compute relative closure percentage of the dense-reference gap."""
    gap = dense_ref - baseline_map
    if gap <= 1e-4:
        return 100.0
    return round(float((full_map - baseline_map) / gap * 100.0), 2)


def _load_optional_profiles(run_root: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    profile_dir = run_root / "cost_profile"
    if not profile_dir.is_dir():
        return {}, [f"optional cost_profile directory not present: {profile_dir}"]
    profiles: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    for arm in REQUIRED_PROFILE_ARMS:
        profile_path = profile_dir / f"profile_{arm}.json"
        if not profile_path.is_file():
            issues.append(f"optional profile missing: {profile_path}")
            continue
        try:
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"optional profile unreadable: {profile_path}: {exc}")
            continue
        arm_issues = []
        if data.get("profile_complete") is not True:
            arm_issues.append(f"{profile_path} missing profile_complete=true")
        for key in ("p50_latency_ms", "p95_latency_ms", "peak_memory_allocated_mb"):
            if key not in data:
                arm_issues.append(f"{profile_path} missing {key}")
                continue
            try:
                value = float(data[key])
            except (TypeError, ValueError):
                arm_issues.append(f"{profile_path} has non-numeric {key}={data[key]!r}")
                continue
            if key.endswith("latency_ms") and value <= 0.0:
                arm_issues.append(f"{profile_path} has non-positive {key}={value}")
            if key == "peak_memory_allocated_mb" and value < 0.0:
                arm_issues.append(f"{profile_path} has negative peak memory")
        if arm_issues:
            issues.extend(arm_issues)
        else:
            profiles[arm] = data
    return profiles, issues


def _validate_video_maps(
    *,
    arm: str,
    seed: int,
    video_map: Dict[str, float],
    video_map70: Dict[str, float],
    expected_video_count: int,
) -> set[str]:
    problems = []
    if not video_map:
        problems.append("missing video_mAP")
    if not video_map70:
        problems.append("missing video_mAP@0.70")
    if expected_video_count > 0 and len(video_map) != expected_video_count:
        problems.append(f"video_mAP count {len(video_map)} != {expected_video_count}")
    if expected_video_count > 0 and len(video_map70) != expected_video_count:
        problems.append(f"video_mAP@0.70 count {len(video_map70)} != {expected_video_count}")
    if set(video_map.keys()) != set(video_map70.keys()):
        problems.append("video_mAP and video_mAP@0.70 identities differ")
    if problems:
        raise RuntimeError(f"{arm}/seed_{seed}: " + "; ".join(problems))
    return set(video_map.keys())


def main():
    parser = argparse.ArgumentParser(description="Analyze DUCA Evidence Recovery results.")
    parser.add_argument("--run-root", type=str, required=True, help="Path to matrix run root.")
    parser.add_argument("--output", type=str, default="statistical_analysis.json", help="Output path.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds to analyze.")
    parser.add_argument(
        "--expected-video-count",
        type=int,
        default=int(os.environ.get("DUCA_EXPECTED_VIDEO_COUNT", EXPECTED_VIDEO_COUNT)),
        help="Expected validation video identities in each official metric file.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root directory does not exist: {run_root}")

    if args.seeds is not None:
        target_seeds = [int(seed) for seed in args.seeds]
    elif "DUCA_SEEDS" in os.environ:
        target_seeds = [int(seed) for seed in os.environ["DUCA_SEEDS"].split()]
    else:
        target_seeds = SEEDS

    print(f"[INFO] Analyzing matrix results from {run_root} for seeds {target_seeds}...")

    arm_seed_avg_map: Dict[str, Dict[int, float]] = {arm: {} for arm in ARM_NAMES}
    arm_seed_map70: Dict[str, Dict[int, float]] = {arm: {} for arm in ARM_NAMES}
    arm_video_maps: Dict[str, Dict[int, Dict[str, float]]] = {arm: {} for arm in ARM_NAMES}
    arm_video_map70: Dict[str, Dict[int, Dict[str, float]]] = {arm: {} for arm in ARM_NAMES}
    arm_mean_map: Dict[str, float] = {}
    missing_cells = []
    reference_video_ids: Optional[set[str]] = None

    for arm in ARM_NAMES:
        seed_means = []
        for seed in target_seeds:
            eval_dir = run_root / arm / f"seed_{seed}" / "official_eval"
            if not eval_dir.exists():
                missing_cells.append(f"{arm}/seed_{seed} (eval_dir missing: {eval_dir})")
                continue
            res = parse_eval_results(eval_dir)
            avg_m = res.get("average_mAP")
            if avg_m is None:
                missing_cells.append(f"{arm}/seed_{seed} (no average_mAP in eval output)")
                continue
            video_ids = _validate_video_maps(
                arm=arm,
                seed=seed,
                video_map=res.get("video_mAP", {}),
                video_map70=res.get("video_mAP@0.70", {}),
                expected_video_count=int(args.expected_video_count),
            )
            if reference_video_ids is None:
                reference_video_ids = video_ids
            elif video_ids != reference_video_ids:
                missing_cells.append(f"{arm}/seed_{seed} video identities differ from matrix reference")
                continue

            arm_seed_avg_map[arm][seed] = avg_m
            arm_seed_map70[arm][seed] = res.get("mAP@0.70", 0.0)
            arm_video_maps[arm][seed] = res["video_mAP"]
            arm_video_map70[arm][seed] = res["video_mAP@0.70"]
            seed_means.append(avg_m)

        if len(seed_means) < len(target_seeds):
            missing_for_arm = [seed for seed in target_seeds if seed not in arm_seed_avg_map[arm]]
            missing_cells.append(f"{arm} missing seeds {missing_for_arm}")
        else:
            arm_mean_map[arm] = round(float(np.mean(seed_means)), 4)

    if missing_cells:
        raise RuntimeError(
            f"Formal statistical analysis requires all cells for target seeds {target_seeds}. "
            f"Missing or invalid cells: {missing_cells}"
        )

    profile_data, profile_issues = _load_optional_profiles(run_root)
    if not profile_data:
        profile_status = "not_collected"
    elif profile_issues:
        profile_status = "partial_or_invalid"
    else:
        profile_status = "available"

    analysis_results: Dict[str, Any] = {
        "arm_mean_map": arm_mean_map,
        "dense_reference": DENSE_REFERENCE_AVG_MAP,
        "comparisons": {},
        "decision_gates": {},
        "optional_system_metrics": {
            "required_for_scientific_acceptance": False,
            "status": profile_status,
            "profiles": profile_data,
            "issues": profile_issues,
        },
        "expected_video_count": int(args.expected_video_count),
        "video_identity_count": 0 if reference_video_ids is None else len(reference_video_ids),
    }

    for comp_key, (arm_a, arm_b, desc) in COMPARISONS.items():
        boot_res = paired_bootstrap_analysis(
            arm_seed_avg_map[arm_a],
            arm_seed_avg_map[arm_b],
            arm_video_maps[arm_a],
            arm_video_maps[arm_b],
            seeds=target_seeds,
        )
        boot_res["description"] = desc
        boot_res["arm_a_mean"] = arm_mean_map[arm_a]
        boot_res["arm_b_mean"] = arm_mean_map[arm_b]
        analysis_results["comparisons"][comp_key] = boot_res

    gate1_boot = analysis_results["comparisons"]["FULL_vs_C0"]
    analysis_results["decision_gates"]["gate1_full_vs_c0_ci_low_positive"] = {
        "passed": gate1_boot["ci_low"] > 0.00,
        "ci_low": gate1_boot["ci_low"],
        "criterion": "FULL-C0 average mAP 95% CI lower bound > 0.00",
    }

    closure_pct = compute_dense_gap_closure(arm_mean_map["F"], arm_mean_map["C0"])
    analysis_results["decision_gates"]["gate2_dense_gap_closure"] = {
        "passed": closure_pct >= 10.0,
        "closure_percentage": closure_pct,
        "criterion": "dense-gap closure >= 10.0%",
    }

    boot_70 = paired_bootstrap_analysis(
        arm_seed_map70["F"],
        arm_seed_map70["C0"],
        arm_video_map70["F"],
        arm_video_map70["C0"],
        seeds=target_seeds,
    )
    analysis_results["decision_gates"]["gate3_high_iou_retention"] = {
        "passed": boot_70["ci_low"] > -0.20,
        "map70_ci_low": boot_70["ci_low"],
        "map70_mean_diff": boot_70["mean_diff"],
        "bootstrap_mode": boot_70["bootstrap_mode"],
        "criterion": "mAP@0.70 95% CI lower bound > -0.20",
    }

    analysis_results["decision_gates"]["gate8_matrix_completeness"] = {
        "passed": True,
        "completed_cells": len(ARM_NAMES) * len(target_seeds),
        "target_cells": len(ARM_NAMES) * len(target_seeds),
        "video_identity_count": 0 if reference_video_ids is None else len(reference_video_ids),
        "criterion": "all 8x3 cells complete with matching video identities",
    }

    analysis_results["all_gates_passed"] = all(
        gate.get("passed", False) for gate in analysis_results["decision_gates"].values()
    )

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"[INFO] Analysis completed successfully. Output saved to {out_path}")
    print(f"[INFO] All Gates Passed: {analysis_results['all_gates_passed']}")


if __name__ == "__main__":
    main()
