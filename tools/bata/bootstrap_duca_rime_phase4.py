from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.duca_p0_evaluation import (
    bootstrap_official_metric_differences,
)


SCHEMA = "duca_rime_phase4_comparisons_v1"


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_metrics(path: str | Path, expected_variant: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if (
        payload.get("schema_version") != "duca_rime_localization_metrics_v1"
        or embedded != _canonical_sha256(unsigned)
        or int(payload.get("phase", -1)) != 4
        or payload.get("variant") != expected_variant
        or payload.get("split_role") != "official_final_evaluation"
        or payload.get("uses_official_final") is not True
        or payload.get("official_final_used_for_training_or_selection") is not False
        or payload.get("padded_to_kmax") is not False
    ):
        raise ValueError(f"invalid Phase-4 localization metrics: {resolved}")
    terminal_path = Path(payload["terminal_evaluation_path"]).resolve()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        not terminal_path.is_file()
        or _sha256_file(terminal_path) != payload["terminal_evaluation_sha256"]
        or terminal.get("schema_version") != "duca_rime_terminal_evaluation_v1"
        or terminal.get("variant") != expected_variant
        or terminal.get("padded_to_kmax") is not False
    ):
        raise ValueError("Phase-4 terminal evaluation artifact drifted")
    return resolved, payload, terminal


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(quantile)
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _auxiliary_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    draws: Sequence[tuple[str, ...]],
) -> dict[str, float]:
    if set(left) != set(right) or len(left) < 3:
        raise ValueError("auxiliary bootstrap requires aligned per-video metrics")
    differences = [
        mean(float(left[video]) - float(right[video]) for video in draw)
        for draw in draws
    ]
    return {
        "mean": mean(
            float(left[video]) - float(right[video]) for video in sorted(left)
        ),
        "ci95_low": _percentile(differences, 0.025),
        "ci95_high": _percentile(differences, 0.975),
    }


def build_comparisons(
    *,
    rime_metrics: str | Path,
    fixed_metrics: str | Path,
    same_k_metrics: str | Path,
    output: str | Path,
    bootstrap_samples: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if int(bootstrap_samples) < 1000:
        raise ValueError("Phase-4 comparisons require at least 1000 bootstrap samples")
    rime_variant = str(
        json.loads(
            Path(rime_metrics).expanduser().resolve().read_text(encoding="utf-8")
        ).get("variant", "")
    )
    if rime_variant not in {"RIME-full", "RIME-full-TriDet"}:
        raise ValueError("Phase-4 RIME comparison has an invalid full-model variant")
    rime_path, rime, rime_terminal = _load_metrics(rime_metrics, rime_variant)
    backend = str(rime["detector_backend"])
    same_variant = "U-same-K-TriDet" if backend == "TriDet" else "U-same-K"
    fixed_variant = "U-fixed-TriDet" if backend == "TriDet" else "U-fixed"
    fixed_path, fixed, fixed_terminal = _load_metrics(fixed_metrics, fixed_variant)
    same_path, same, same_terminal = _load_metrics(same_k_metrics, same_variant)
    common = (
        "git_commit",
        "seed",
        "detector_backend",
        "target_mean_cost",
        "split_assignment_sha256",
        "evaluation_video_ids",
        "annotation_sha256",
        "duration_thresholds_seconds",
    )
    for key in common:
        if rime.get(key) != fixed.get(key) or rime.get(key) != same.get(key):
            raise ValueError(f"Phase-4 comparison arms differ on {key}")
    if (
        rime_terminal.get("evaluation_config")
        != fixed_terminal.get("evaluation_config")
        or rime_terminal.get("evaluation_config")
        != same_terminal.get("evaluation_config")
    ):
        raise ValueError("Phase-4 comparison arms use different official evaluators")
    videos = tuple(str(value) for value in rime["evaluation_video_ids"])
    rng = random.Random(int(seed))
    draws = [
        tuple(videos[rng.randrange(len(videos))] for _ in videos)
        for _ in range(int(bootstrap_samples))
    ]

    prediction_paths = {
        "rime": rime_terminal["prediction_path"],
        "fixed": fixed_terminal["prediction_path"],
        "same_k": same_terminal["prediction_path"],
    }
    official_fixed = bootstrap_official_metric_differences(
        {"fixed": prediction_paths["fixed"], "rime": prediction_paths["rime"]},
        rime_terminal["evaluation_config"],
        baseline_family="fixed",
        expected_video_ids=videos,
        expected_subset="validation",
        samples=bootstrap_samples,
        seed=seed,
        workers=workers,
    )
    official_same = bootstrap_official_metric_differences(
        {"same_k": prediction_paths["same_k"], "rime": prediction_paths["rime"]},
        rime_terminal["evaluation_config"],
        baseline_family="same_k",
        expected_video_ids=videos,
        expected_subset="validation",
        samples=bootstrap_samples,
        seed=seed,
        workers=workers,
    )

    def comparison(
        baseline: Mapping[str, Any],
        official: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "official_map_bootstrap": {
                "schema_version": official["schema_version"],
                "official_evaluator_reexecuted_per_resample": True,
                "paired_video_cluster_bootstrap": True,
                "bootstrap_samples": int(official["bootstrap_samples"]),
                "evaluation_config_sha256": official["evaluation_config_sha256"],
            },
            "auxiliary_video_bootstrap": {
                "official_evaluator_reexecuted_per_resample": False,
                "paired_video_cluster_bootstrap": True,
                "bootstrap_samples": int(bootstrap_samples),
                "semantics": "predeclared_per_video_duration_and_pair_metrics",
            },
            "avg_map": dict(official["comparisons"]["rime"]["avg_map"]),
            "map_0.7": dict(official["comparisons"]["rime"]["map_0.7"]),
            "short_map": _auxiliary_bootstrap(
                rime["video_metrics"]["short_map"],
                baseline["video_metrics"]["short_map"],
                draws=draws,
            ),
            "pair_support": _auxiliary_bootstrap(
                rime["video_metrics"]["pair_support"],
                baseline["video_metrics"]["pair_support"],
                draws=draws,
            ),
        }

    payload = {
        "schema_version": SCHEMA,
        "git_commit": rime["git_commit"],
        "detector_backend": backend,
        "target_mean_cost": float(rime["target_mean_cost"]),
        "seed": int(rime["seed"]),
        "evaluation_video_ids": list(videos),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "comparisons": {
            "rime_minus_best_fixed": comparison(fixed, official_fixed),
            "rime_minus_uniform_same_k": comparison(same, official_same),
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in (rime_path, fixed_path, same_path)
        ],
        "official_final_used_for_training_or_selection": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    target = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite different Phase-4 comparisons: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {"path": str(target), "sha256": _sha256_file(target), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build official and auxiliary paired Phase-4 RIME comparisons."
    )
    parser.add_argument("--rime-metrics", required=True)
    parser.add_argument("--fixed-metrics", required=True)
    parser.add_argument("--same-k-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = build_comparisons(
        rime_metrics=args.rime_metrics,
        fixed_metrics=args.fixed_metrics,
        same_k_metrics=args.same_k_metrics,
        output=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
