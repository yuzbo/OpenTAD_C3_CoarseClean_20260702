from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import TRAIN_ROLES, validate_rime_splits


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain one JSON object: {resolved}")
    return resolved, payload


def _read_jsonl(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = Path(path).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain nonempty object records: {resolved}")
    return resolved, rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    text = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different RIME records: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": len(rows),
    }


def _metrics(path: str | Path, sha256: str) -> tuple[Path, dict[str, Any]]:
    resolved, payload = _load_json(path)
    if (
        _sha256_file(resolved) != str(sha256)
        or payload.get("schema_version") != "duca_rime_localization_metrics_v1"
        or int(payload.get("phase", -1)) not in {2, 3}
        or payload.get("uses_official_final") is not False
        or payload.get("official_final_used_for_training_or_selection") is not False
        or payload.get("padded_to_kmax") is not False
    ):
        raise ValueError(f"invalid train-only localization metrics: {resolved}")
    return resolved, payload


def phase0_records(
    *,
    source_manifest: str | Path,
    output: str | Path,
    primary_metric: str,
) -> dict[str, Any]:
    manifest_path, manifest = _load_json(source_manifest)
    if (
        manifest.get("schema_version") != "duca_rime_phase0_source_manifest_v1"
        or manifest.get("uses_official_final") is not False
    ):
        raise ValueError("invalid Phase-0 source manifest")
    rows = []
    common_videos = None
    split_hash = None
    seen_replicates = set()
    for entry in manifest.get("replicates", ()):
        replicate = str(entry.get("replicate_id", ""))
        replicate_kind = str(entry.get("replicate_kind", "unspecified_legacy"))
        if not replicate or replicate in seen_replicates:
            raise ValueError("Phase-0 replicate identities must be nonempty and unique")
        seen_replicates.add(replicate)
        metrics_path, metrics = _metrics(entry["path"], entry["sha256"])
        values = metrics.get("video_metrics", {}).get(primary_metric)
        if not isinstance(values, Mapping) or len(values) < 3:
            raise ValueError("Phase-0 replicate lacks per-video primary metrics")
        videos = tuple(sorted(str(value) for value in values))
        if common_videos is None:
            common_videos = videos
            split_hash = metrics["split_assignment_sha256"]
        elif videos != common_videos or metrics["split_assignment_sha256"] != split_hash:
            raise ValueError("Phase-0 replicate video/split identity drift")
        for video in videos:
            value = float(values[video])
            if not math.isfinite(value):
                raise ValueError("Phase-0 primary metric must be finite")
            rows.append(
                {
                    "schema_version": "duca_rime_phase0_measurement_v1",
                    "video_id": video,
                    "replicate_id": replicate,
                    "replicate_kind": replicate_kind,
                    "metric_name": str(primary_metric),
                    "value": value,
                    "source_path": str(metrics_path),
                    "source_sha256": _sha256_file(metrics_path),
                    "uses_official_final": False,
                    "source_manifest_claim_scope": str(
                        manifest.get("claim_scope", "unspecified")
                    ),
                }
            )
    if len(seen_replicates) < 2:
        raise ValueError("Phase-0 variance requires at least two real replicates")
    artifact = _write_jsonl(output, rows)
    return {
        "schema_version": "duca_rime_gate_record_build_v1",
        "stage": "phase0",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "split_assignment_sha256": split_hash,
        "replicate_count": len(seen_replicates),
        "output": artifact,
        "official_final_subset_consumed": False,
    }


def o1_records(
    *,
    source_manifest: str | Path,
    output: str | Path,
    score_metric: str,
) -> dict[str, Any]:
    manifest_path, manifest = _load_json(source_manifest)
    if (
        manifest.get("schema_version") != "duca_rime_o1_source_manifest_v1"
        or manifest.get("uses_official_final") is not False
        or manifest.get("position_policy") != "exact_uniform"
        or manifest.get("detector_training_exposure")
        != "mixed_k_registered_panel"
        or not str(manifest.get("mixed_k_detector_identity_sha256", ""))
    ):
        raise ValueError(
            "invalid formal O1 source manifest; a registered mixed-K-trained "
            "detector is required"
        )
    panel = {}
    common_videos = None
    split_hash = None
    split_role = None
    for entry in manifest.get("budget_evaluations", ()):
        budget = int(entry["budget"])
        cost = float(entry["measured_heavy_frame_cost"])
        metrics_path, metrics = _metrics(entry["path"], entry["sha256"])
        values = metrics.get("video_metrics", {}).get(score_metric)
        if (
            budget <= 0
            or not math.isfinite(cost)
            or cost <= 0.0
            or not isinstance(values, Mapping)
        ):
            raise ValueError("invalid O1 budget evaluation")
        videos = tuple(sorted(str(value) for value in values))
        if common_videos is None:
            common_videos = videos
            split_hash = metrics["split_assignment_sha256"]
            split_role = metrics["split_role"]
        elif (
            videos != common_videos
            or metrics["split_assignment_sha256"] != split_hash
            or metrics["split_role"] != split_role
        ):
            raise ValueError("O1 budget panel has video/split drift")
        for video in videos:
            key = (video, budget)
            if key in panel:
                raise ValueError("duplicate O1 video/budget measurement")
            score = float(values[video])
            if not math.isfinite(score):
                raise ValueError("O1 score must be finite")
            panel[key] = {
                "schema_version": "duca_rime_o1_budget_panel_v1",
                "video_id": video,
                "budget": budget,
                "cost": cost,
                "score": score,
                "score_metric": score_metric,
                "position_policy": "exact_uniform",
                "detector_training_exposure": "mixed_k_registered_panel",
                "mixed_k_detector_identity_sha256": manifest[
                    "mixed_k_detector_identity_sha256"
                ],
                "source_path": str(metrics_path),
                "source_sha256": _sha256_file(metrics_path),
                "split_role": split_role,
                "uses_official_final": False,
            }
    budgets = sorted({budget for _video, budget in panel})
    if len(budgets) < 2 or set(panel) != {
        (video, budget) for video in (common_videos or ()) for budget in budgets
    }:
        raise ValueError("O1 budget evidence is not a rectangular panel")
    artifact = _write_jsonl(output, [panel[key] for key in sorted(panel)])
    return {
        "schema_version": "duca_rime_gate_record_build_v1",
        "stage": "o1",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "split_assignment_sha256": split_hash,
        "split_role": split_role,
        "budgets": budgets,
        "output": artifact,
        "official_final_subset_consumed": False,
    }


def _ledger_by_video(
    path: str | Path,
    sha256: str,
    *,
    budget: int,
) -> dict[str, list[str]]:
    resolved, rows = _read_jsonl(path)
    if _sha256_file(resolved) != str(sha256):
        raise ValueError("O2 inference ledger SHA-256 drift")
    grouped: dict[str, list[str]] = {}
    seen_windows = set()
    for row in rows:
        video = str(row.get("video_id", ""))
        start = int(row.get("window_start_frame", -1))
        positions = [int(value) for value in row.get("selected_dense_indices", ())]
        if (
            row.get("schema_version") != "duca_rime_inference_ledger_v1"
            or not video
            or start < 0
            or (video, start) in seen_windows
            or int(row.get("requested_k", -1)) != int(budget)
            or not int(row.get("effective_k", -1))
            == int(row.get("unique_k", -1))
            == int(row.get("backbone_input_k", -1))
            == int(row.get("padded_k", -1))
            == int(budget)
            or positions != sorted(set(positions))
            or len(positions) != int(budget)
            or float(row.get("observed_max_gap_seconds", math.inf))
            > float(row.get("max_gap_seconds_cap", -math.inf)) + 1.0e-8
        ):
            raise ValueError("O2 inference ledger violates exact-K/physical-gap execution")
        seen_windows.add((video, start))
        grouped.setdefault(video, []).extend(
            f"{start:012d}:{position:06d}" for position in positions
        )
    return {
        video: sorted(set(keys))
        for video, keys in grouped.items()
    }


def o2_records(
    *,
    source_manifest: str | Path,
    output: str | Path,
    score_metric: str,
) -> dict[str, Any]:
    manifest_path, manifest = _load_json(source_manifest)
    if (
        manifest.get("schema_version") != "duca_rime_o2_source_manifest_v1"
        or manifest.get("uses_official_final") is not False
        or not str(manifest.get("mixed_k_detector_identity_sha256", ""))
    ):
        raise ValueError("invalid O2 source manifest")
    panel = {}
    common_videos = None
    split_hash = None
    split_role = None
    for entry in manifest.get("decoder_evaluations", ()):
        budget = int(entry["budget"])
        family = str(entry["family"])
        metrics_path, metrics = _metrics(entry["metrics_path"], entry["metrics_sha256"])
        values = metrics.get("video_metrics", {}).get(score_metric)
        selections = _ledger_by_video(
            entry["ledger_path"],
            entry["ledger_sha256"],
            budget=budget,
        )
        if not family or not isinstance(values, Mapping):
            raise ValueError("invalid O2 decoder entry")
        videos = tuple(sorted(str(value) for value in values))
        if set(videos) != set(selections):
            raise ValueError("O2 metrics and selection ledger cover different videos")
        if common_videos is None:
            common_videos = videos
            split_hash = metrics["split_assignment_sha256"]
            split_role = metrics["split_role"]
        elif (
            videos != common_videos
            or metrics["split_assignment_sha256"] != split_hash
            or metrics["split_role"] != split_role
        ):
            raise ValueError("O2 decoder panel has video/split drift")
        for video in videos:
            key = (video, budget, family)
            if key in panel:
                raise ValueError("duplicate O2 video/budget/family measurement")
            panel[key] = {
                "schema_version": "duca_rime_o2_decoder_panel_v1",
                "video_id": video,
                "budget": budget,
                "family": family,
                "score": float(values[video]),
                "score_metric": score_metric,
                "selection_keys": selections[video],
                "exact_k_all_windows": True,
                "max_gap_violation": False,
                "mixed_k_detector_identity_sha256": manifest[
                    "mixed_k_detector_identity_sha256"
                ],
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "ledger_path": str(Path(entry["ledger_path"]).resolve()),
                "ledger_sha256": str(entry["ledger_sha256"]),
                "split_role": split_role,
                "uses_official_final": False,
            }
    videos = tuple(common_videos or ())
    budgets = sorted({key[1] for key in panel})
    families = sorted({key[2] for key in panel})
    required = {
        (video, budget, family)
        for video in videos
        for budget in budgets
        for family in families
    }
    if (
        len(budgets) < 2
        or "independent" not in families
        or set(panel) != required
    ):
        raise ValueError("O2 evidence is not a complete decoder-family panel")
    artifact = _write_jsonl(output, [panel[key] for key in sorted(panel)])
    return {
        "schema_version": "duca_rime_gate_record_build_v1",
        "stage": "o2",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "split_assignment_sha256": split_hash,
        "split_role": split_role,
        "budgets": budgets,
        "families": families,
        "output": artifact,
        "official_final_subset_consumed": False,
    }


def supervised_records(
    *,
    source_jsonl: str | Path,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    output: str | Path,
    kind: str,
) -> dict[str, Any]:
    split_validation = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    final_videos = set(split["official_final_evaluation"]["videos"])
    source_path, source_rows = _read_jsonl(source_jsonl)
    schemas = {
        "o3": (
            "duca_rime_o3_crossfit_prediction_v1",
            "duca_rime_o3_rank_record_v1",
        ),
        "o4": (
            "duca_rime_o4_calibrated_risk_prediction_v1",
            "duca_rime_o4_risk_record_v1",
        ),
        "price": (
            "duca_rime_price_prediction_v1",
            "duca_rime_price_calibration_v1",
        ),
    }
    source_schema, output_schema = schemas[kind]
    rows = []
    for row in source_rows:
        provenance = row.get("provenance")
        video = str(row.get("video_id", ""))
        if (
            row.get("schema_version") != source_schema
            or not isinstance(provenance, Mapping)
            or provenance.get("cross_fitted") is not True
            or provenance.get("fit_split") not in {"train", "training", "train_only"}
            or provenance.get("uses_validation_or_test") is not False
            or video in final_videos
            or bool(set(map(str, provenance.get("fit_video_ids", ()))) & final_videos)
            or bool(set(map(str, provenance.get("eval_video_ids", ()))) & final_videos)
            or provenance.get("split_assignment_sha256")
            != split_validation["assignment_sha256"]
        ):
            raise ValueError(f"{kind.upper()} source violates train-only cross-fit")
        converted = dict(row)
        converted["schema_version"] = output_schema
        rows.append(converted)
    artifact = _write_jsonl(output, rows)
    return {
        "schema_version": "duca_rime_gate_record_build_v1",
        "stage": kind,
        "source_artifact": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        },
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "output": artifact,
        "official_final_subset_consumed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build hash-bound real-measurement records for DUCA-RIME gates."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    phase0 = sub.add_parser("phase0")
    phase0.add_argument("--source-manifest", required=True)
    phase0.add_argument("--output", required=True)
    phase0.add_argument("--primary-metric", default="avg_map")
    o1 = sub.add_parser("o1")
    o1.add_argument("--source-manifest", required=True)
    o1.add_argument("--output", required=True)
    o1.add_argument("--score-metric", default="avg_map")
    o2 = sub.add_parser("o2")
    o2.add_argument("--source-manifest", required=True)
    o2.add_argument("--output", required=True)
    o2.add_argument("--score-metric", default="avg_map")
    for name in ("o3", "o4", "price"):
        item = sub.add_parser(name)
        item.add_argument("--source-jsonl", required=True)
        item.add_argument("--split-manifest", required=True)
        item.add_argument("--split-manifest-sha256", required=True)
        item.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "phase0":
        result = phase0_records(
            source_manifest=args.source_manifest,
            output=args.output,
            primary_metric=args.primary_metric,
        )
    elif args.command == "o1":
        result = o1_records(
            source_manifest=args.source_manifest,
            output=args.output,
            score_metric=args.score_metric,
        )
    elif args.command == "o2":
        result = o2_records(
            source_manifest=args.source_manifest,
            output=args.output,
            score_metric=args.score_metric,
        )
    else:
        result = supervised_records(
            source_jsonl=args.source_jsonl,
            split_manifest=args.split_manifest,
            split_manifest_sha256=args.split_manifest_sha256,
            output=args.output,
            kind=args.command,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
