from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import TRAIN_ROLES, validate_rime_splits


OBSERVATION_SCHEMA = "duca_rime_budget_target_observation_v1"
HARD_UTILITY_SCHEMA = "duca_rime_hard_frame_target_v1"
TARGET_SCHEMA = "duca_rime_training_target_v1"
SUMMARY_SCHEMA = "duca_rime_training_target_summary_v1"
DEFAULT_FIT_ROLES = (
    "hard_label_generation",
    "utility_risk_fit",
    "dual_risk_calibration",
)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _read_jsonl(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain nonempty object records: {source}")
    return source, rows


def _validate_budgets(candidate_budgets: Sequence[int]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in candidate_budgets)
    if (
        len(budgets) < 2
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] <= 0
    ):
        raise ValueError("candidate budgets must be positive, unique, and increasing")
    return budgets


def _window_key(row: Mapping[str, Any]) -> tuple[str, int]:
    video = str(row.get("video_id") or row.get("video_name") or "")
    start = int(row.get("window_start_frame", -1))
    if not video or start < 0:
        raise ValueError("RIME target source requires video_id and window_start_frame")
    return video, start


def _validate_cross_fit_provenance(
    row: Mapping[str, Any],
    *,
    video: str,
    fit_universe: set[str],
    final_videos: set[str],
) -> set[str]:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("RIME target source provenance must be an object")
    fit_ids = {str(value) for value in provenance.get("fit_video_ids", ())}
    eval_ids = {str(value) for value in provenance.get("eval_video_ids", ())}
    if (
        provenance.get("fit_split") not in {"train", "training", "train_only"}
        or provenance.get("cross_fitted") is not True
        or provenance.get("uses_validation_or_test") is not False
        or not fit_ids
        or video not in eval_ids
        or video in fit_ids
        or not fit_ids <= fit_universe
        or bool(fit_ids & final_videos)
        or bool(eval_ids & final_videos)
    ):
        raise ValueError(
            "RIME target source violates train-only cross-fit or official-final isolation"
        )
    return fit_ids


def _load_split_contract(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    target_role: str,
    fit_roles: Sequence[str],
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    actual_sha = _sha256_file(path)
    if not expected_sha256 or actual_sha != str(expected_sha256).lower():
        raise ValueError("RIME split manifest SHA-256 is required and must match")
    validate_rime_splits(path, expected_sha256=expected_sha256)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if target_role not in TRAIN_ROLES:
        raise ValueError(f"unknown RIME target role: {target_role}")
    selected_fit_roles = tuple(str(value) for value in fit_roles)
    if (
        not selected_fit_roles
        or len(selected_fit_roles) != len(set(selected_fit_roles))
        or target_role in selected_fit_roles
        or any(role not in TRAIN_ROLES for role in selected_fit_roles)
    ):
        raise ValueError("fit roles must be unique train roles disjoint from target_role")
    role_rows = manifest["train_roles"]
    target_videos = set(str(value) for value in role_rows[target_role]["videos"])
    fit_universe = set().union(
        *(set(str(value) for value in role_rows[role]["videos"]) for role in selected_fit_roles)
    )
    final_videos = set(
        str(value) for value in manifest["official_final_evaluation"]["videos"]
    )
    if target_videos & fit_universe or target_videos & final_videos or fit_universe & final_videos:
        raise ValueError("RIME split roles are not disjoint")
    return {
        "path": path,
        "sha256": actual_sha,
        "assignment_sha256": str(manifest["assignment_sha256"]),
        "target_role": str(target_role),
        "fit_roles": selected_fit_roles,
        "target_videos": target_videos,
        "fit_universe": fit_universe,
        "final_videos": final_videos,
    }


def _immutable_write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    text = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different RIME targets: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": len(rows),
    }


def build_training_targets(
    *,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    observations_jsonl: str | Path,
    hard_utility_jsonl: str | Path,
    output_jsonl: str | Path,
    candidate_budgets: Sequence[int],
    target_role: str = "detector_selector_train",
    fit_roles: Sequence[str] = DEFAULT_FIT_ROLES,
) -> dict[str, Any]:
    budgets = _validate_budgets(candidate_budgets)
    split = _load_split_contract(
        split_manifest,
        expected_sha256=split_manifest_sha256,
        target_role=target_role,
        fit_roles=fit_roles,
    )
    observation_path, observations = _read_jsonl(observations_jsonl)
    hard_path, hard_rows = _read_jsonl(hard_utility_jsonl)

    budget_values: dict[tuple[str, int, int], list[tuple[float, int]]] = defaultdict(list)
    fit_ids_by_window: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in observations:
        if row.get("schema_version") != OBSERVATION_SCHEMA:
            raise ValueError("unsupported RIME budget-target observation schema")
        video, start = _window_key(row)
        if video not in split["target_videos"]:
            raise ValueError("budget-target observation is outside target_role")
        fit_ids = _validate_cross_fit_provenance(
            row,
            video=video,
            fit_universe=split["fit_universe"],
            final_videos=split["final_videos"],
        )
        budget = int(row.get("budget", -1))
        utility = float(row.get("utility", float("nan")))
        failure = int(row.get("observed_pair_failure", -1))
        if budget not in budgets or not math.isfinite(utility) or failure not in {0, 1}:
            raise ValueError("invalid RIME budget utility/risk observation")
        budget_values[(video, start, budget)].append((utility, failure))
        fit_ids_by_window[(video, start)].update(fit_ids)

    hard_values: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    for row in hard_rows:
        if row.get("schema_version") != HARD_UTILITY_SCHEMA:
            raise ValueError("unsupported RIME hard-frame utility schema")
        video, start = _window_key(row)
        if video not in split["target_videos"]:
            raise ValueError("hard-frame target is outside target_role")
        fit_ids = _validate_cross_fit_provenance(
            row,
            video=video,
            fit_universe=split["fit_universe"],
            final_videos=split["final_videos"],
        )
        values = [float(value) for value in row.get("hard_frame_utility", ())]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("hard-frame utility must be a nonempty finite vector")
        hard_values[(video, start)].append(values)
        fit_ids_by_window[(video, start)].update(fit_ids)

    windows = sorted({(video, start) for video, start, _budget in budget_values})
    if not windows or set(windows) != set(hard_values):
        raise ValueError("budget and hard-frame target windows must be nonempty and identical")
    output_rows = []
    for video, start in windows:
        missing = [
            budget for budget in budgets if (video, start, budget) not in budget_values
        ]
        if missing:
            raise ValueError(f"RIME budget target panel is incomplete for {(video, start)}")
        hard_replicates = hard_values[(video, start)]
        hard_lengths = {len(values) for values in hard_replicates}
        if len(hard_lengths) != 1:
            raise ValueError("hard-frame utility replicates disagree on temporal length")
        hard_length = hard_lengths.pop()
        hard_mean = [
            mean(values[index] for values in hard_replicates)
            for index in range(hard_length)
        ]
        utility_target = []
        risk_target = []
        replicate_counts = []
        for budget in budgets:
            values = budget_values[(video, start, budget)]
            utility_target.append(mean(item[0] for item in values))
            risk_target.append(mean(item[1] for item in values))
            replicate_counts.append(len(values))
        output_rows.append(
            {
                "schema_version": TARGET_SCHEMA,
                "video_id": video,
                "window_start_frame": start,
                "candidate_budgets": list(budgets),
                "utility_target": utility_target,
                "risk_target": risk_target,
                "target_mask": [True] * len(budgets),
                "hard_frame_utility": hard_mean,
                "provenance": {
                    "fit_split": "train_only",
                    "cross_fitted": True,
                    "uses_validation_or_test": False,
                    "target_role": split["target_role"],
                    "fit_roles": list(split["fit_roles"]),
                    "fit_video_ids": sorted(fit_ids_by_window[(video, start)]),
                    "eval_video_ids": [video],
                    "split_manifest_sha256": split["sha256"],
                    "split_assignment_sha256": split["assignment_sha256"],
                    "budget_observations_sha256": _sha256_file(observation_path),
                    "hard_utility_sha256": _sha256_file(hard_path),
                    "budget_replicate_counts": replicate_counts,
                    "hard_utility_replicate_count": len(hard_replicates),
                    "uses_gt_at_deployment": False,
                    "uses_teacher_at_deployment": False,
                    "uses_prediction_cache_at_deployment": False,
                },
            }
        )

    artifact = _immutable_write_jsonl(output_jsonl, output_rows)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "built",
        "target_artifact": artifact,
        "candidate_budgets": list(budgets),
        "target_role": split["target_role"],
        "fit_roles": list(split["fit_roles"]),
        "target_video_count": len({row["video_id"] for row in output_rows}),
        "window_count": len(output_rows),
        "split_manifest_path": str(split["path"]),
        "split_manifest_sha256": split["sha256"],
        "split_assignment_sha256": split["assignment_sha256"],
        "source_artifacts": {
            "budget_observations": {
                "path": str(observation_path),
                "sha256": _sha256_file(observation_path),
            },
            "hard_utility": {
                "path": str(hard_path),
                "sha256": _sha256_file(hard_path),
            },
        },
        "official_final_subset_consumed": False,
        "claim_scope": "training_supervision_asset_only_no_model_result",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build immutable train-only cross-fitted DUCA-RIME targets."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--observations-jsonl", required=True)
    parser.add_argument("--hard-utility-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--candidate-budgets", nargs="+", type=int, required=True)
    parser.add_argument("--target-role", default="detector_selector_train")
    parser.add_argument("--fit-role", action="append", dest="fit_roles")
    args = parser.parse_args(argv)
    result = build_training_targets(
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        observations_jsonl=args.observations_jsonl,
        hard_utility_jsonl=args.hard_utility_jsonl,
        output_jsonl=args.output_jsonl,
        candidate_budgets=args.candidate_budgets,
        target_role=args.target_role,
        fit_roles=args.fit_roles or DEFAULT_FIT_ROLES,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
