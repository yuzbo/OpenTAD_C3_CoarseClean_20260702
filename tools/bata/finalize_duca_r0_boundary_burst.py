from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tools.bata.create_duca_frontend_split import validate_split_manifest
from tools.bata.duca_p0_evaluation import (
    bootstrap_official_map_differences,
    canonical_sha256,
    evaluation_video_ids,
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
    sha256_file,
)


SCHEMA = "duca_r0_selected_axis_boundary_burst_map_v3"
EVALUATION_SCHEMA = "duca_r0_selected_axis_evaluation_v1"
BOOTSTRAP_SCHEMA = "duca_r0_official_video_bootstrap_v1"
FAMILY_ORDER = (
    "A_exact_uniform",
    "R2Q3_privileged_boundary_burst",
    "R4Q5_privileged_boundary_burst",
    "Z_unrestricted_gt_oracle",
)
PROJECTED_FAMILIES = FAMILY_ORDER[1:3]
UNRESTRICTED_FAMILY = FAMILY_ORDER[3]
FAMILY_SUMMARY_SCHEMA = "duca_r0_boundary_burst_oracle_summary_v1"
FAMILY_RECORD_SCHEMA = "duca_allocation_family_ceiling_record_v1"
FROZEN_HEADROOM_PERCENTAGE_POINTS = 0.20
FROZEN_BOOTSTRAP_SEED = 3407
FROZEN_BOOTSTRAP_CONFIDENCE = 0.95


def _require_file(path: str | Path, sha256: str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if len(str(sha256)) != 64 or not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    if sha256_file(resolved) != str(sha256):
        raise RuntimeError(f"{label} path/hash drift: {resolved}")
    return resolved


def _self_hash(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    recorded = payload.get(key)
    unhashed = dict(payload)
    unhashed.pop(key, None)
    actual = canonical_sha256(unhashed)
    if recorded != actual:
        raise RuntimeError(f"{label} self-hash mismatch")
    return actual


def _is_sha256(value: Any) -> bool:
    try:
        return len(str(value)) == 64 and int(str(value), 16) >= 0
    except ValueError:
        return False


def _same_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1.0e-12)
        for key in left
    )


def _parse_family_evaluations(
    values: Mapping[str, str | Path] | Sequence[str],
) -> dict[str, Path]:
    if isinstance(values, Mapping):
        parsed = {str(key): Path(path).expanduser().resolve() for key, path in values.items()}
    else:
        parsed = {}
        for raw in values:
            family, separator, path = str(raw).partition("=")
            if not separator or not family or family in parsed:
                raise ValueError(f"invalid duplicate --family-evaluation: {raw}")
            parsed[family] = Path(path).expanduser().resolve()
    if tuple(parsed) != FAMILY_ORDER:
        raise ValueError(
            "R0 family evaluation order mismatch: "
            f"expected {FAMILY_ORDER}, got {tuple(parsed)}"
        )
    return parsed


def _validate_family_artifact_chain(
    summary_path: Path,
    artifact_path: Path,
    *,
    holdout_videos: tuple[str, ...],
) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FAMILY_SUMMARY_SCHEMA
        or payload.get("ok") is not True
        or payload.get("output_jsonl") != str(artifact_path)
        or payload.get("output_jsonl_sha256") != sha256_file(artifact_path)
        or payload.get("crop_cut_endpoints_excluded") is not True
        or payload.get("diagnostic_only") is not True
    ):
        raise RuntimeError("R0 family-summary identity mismatch")
    expected_sample_count = int(payload.get("sample_count", -1))
    expected_counts = {
        family: expected_sample_count for family in FAMILY_ORDER
    }
    if payload.get("families") != expected_counts or expected_sample_count < 1:
        raise RuntimeError("R0 family-summary count/order mismatch")

    rows: list[dict[str, Any]] = []
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("schema_version") != FAMILY_RECORD_SCHEMA:
                raise RuntimeError(f"R0 family record schema mismatch at line {line_number}")
            recorded_hash = row.get("record_sha256")
            unhashed = dict(row)
            unhashed.pop("record_sha256", None)
            if recorded_hash != canonical_sha256(unhashed):
                raise RuntimeError(f"R0 family record hash mismatch at line {line_number}")
            families = row.get("families")
            if not isinstance(families, list) or tuple(
                item.get("family_key") for item in families if isinstance(item, Mapping)
            ) != FAMILY_ORDER:
                raise RuntimeError(f"R0 family order mismatch at line {line_number}")
            by_family = {item["family_key"]: item for item in families}
            for family in FAMILY_ORDER:
                contract = by_family[family].get("r0_contract")
                if not isinstance(contract, Mapping) or contract.get("exact_k") is not True:
                    raise RuntimeError(f"R0 exact-K contract mismatch for {family}")
            requested_g = int(payload.get("max_unselected_hole", -1))
            for family in PROJECTED_FAMILIES:
                contract = by_family[family]["r0_contract"]
                if (
                    contract.get("global_coverage_enforced") is not True
                    or int(contract.get("max_unselected_hole", requested_g + 1))
                    > requested_g
                ):
                    raise RuntimeError(f"R0 projected coverage contract mismatch for {family}")
            unrestricted = by_family[UNRESTRICTED_FAMILY]["r0_contract"]
            if (
                unrestricted.get("global_coverage_enforced") is not False
                or unrestricted.get("coverage_cap") != "unrestricted"
                or unrestricted.get("projected_into_deployment_feasible_set") is not False
            ):
                raise RuntimeError("R0 unrestricted Oracle is still coverage-constrained")
            rows.append(row)
    if len(rows) != expected_sample_count:
        raise RuntimeError("R0 family artifact sample count mismatch")
    if {str(row.get("video_id")) for row in rows} != set(holdout_videos):
        raise RuntimeError("R0 family artifact video set differs from the sealed holdout")
    return payload


def _validate_evaluation(
    path: Path,
    *,
    family: str,
    expected_commit: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    config: Path,
    config_sha256: str,
    allocation_artifact: Path,
    allocation_artifact_sha256: str,
    annotation: Path,
    annotation_sha256: str,
    blocked_videos: Path,
    blocked_videos_sha256: str,
    holdout_videos: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"R0 evaluation is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVALUATION_SCHEMA:
        raise RuntimeError(f"R0 evaluation schema mismatch for {family}")
    _self_hash(payload, "evaluation_sha256", label=f"R0 evaluation {family}")
    identity_checks = {
        "git_commit": expected_commit,
        "task": "offline_temporal_action_detection",
        "family": family,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_state_key": "state_dict_ema",
        "config_path": str(config),
        "config_sha256": config_sha256,
        "allocation_artifact_path": str(allocation_artifact),
        "allocation_artifact_sha256": allocation_artifact_sha256,
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_blocked_videos_path": str(blocked_videos),
        "evaluation_blocked_videos_sha256": blocked_videos_sha256,
        "source_subset": "training_internal_holdout",
        "test_subset_consumed": False,
        "runtime_gt_input_to_selector": False,
    }
    for field, expected in identity_checks.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"R0 {family} identity mismatch: {field}")
    if payload.get("evaluator") != official_evaluator_identity():
        raise RuntimeError(f"R0 {family} evaluator identity mismatch")
    class_map = _require_file(
        payload.get("evaluation_class_map_path", ""),
        str(payload.get("evaluation_class_map_sha256", "")),
        label=f"R0 class map {family}",
    )
    prediction = _require_file(
        payload.get("prediction_path", ""),
        str(payload.get("prediction_sha256", "")),
        label=f"R0 prediction {family}",
    )
    evaluation_config = normalize_evaluation_config(
        payload.get("evaluation_config"),
        expected_subset="training",
    )
    if (
        evaluation_config["ground_truth_filename"] != str(annotation)
        or evaluation_config["blocked_videos"] != str(blocked_videos)
        or payload.get("evaluation_config_sha256")
        != canonical_sha256(evaluation_config)
        or evaluation_video_ids(
            evaluation_config,
            expected_subset="training",
        )
        != tuple(sorted(holdout_videos))
    ):
        raise RuntimeError(f"R0 {family} evaluator subset/binding mismatch")
    recomputed = recompute_official_map(
        prediction,
        evaluation_config,
        expected_subset="training",
    )
    metrics = payload.get("metrics")
    if not _is_sha256(payload.get("resolved_config_sha256")) or not _is_sha256(
        payload.get("runtime_config_sha256")
    ):
        raise RuntimeError(f"R0 {family} resolved/runtime config hash is invalid")
    if (
        not isinstance(metrics, Mapping)
        or not _same_metrics(metrics, recomputed["metrics"])
        or int(payload.get("result_count", -1)) != recomputed["result_count"]
        or int(payload.get("video_count", -1)) != recomputed["video_count"]
    ):
        raise RuntimeError(f"R0 {family} official metric recomputation mismatch")
    return {
        "family": family,
        "evaluation_path": str(path),
        "evaluation_file_sha256": sha256_file(path),
        "evaluation_self_sha256": payload["evaluation_sha256"],
        "prediction_path": str(prediction),
        "prediction_sha256": payload["prediction_sha256"],
        "class_map_path": str(class_map),
        "class_map_sha256": payload["evaluation_class_map_sha256"],
        "resolved_config_sha256": payload.get("resolved_config_sha256"),
        "runtime_config_sha256": payload.get("runtime_config_sha256"),
        "evaluation_config_sha256": payload.get("evaluation_config_sha256"),
        "metrics": dict(metrics),
        "average_mAP": float(metrics["average_mAP"]),
    }


def finalize_r0(
    *,
    expected_commit: str,
    family_evaluations: Mapping[str, str | Path] | Sequence[str],
    split_manifest: str | Path,
    split_manifest_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    config_path: str | Path,
    config_sha256: str,
    allocation_artifact_path: str | Path,
    allocation_artifact_sha256: str,
    family_summary_path: str | Path,
    family_summary_sha256: str,
    pretrain_path: str | Path,
    pretrain_sha256: str,
    blocked_videos_path: str | Path,
    blocked_videos_sha256: str,
    bootstrap_output_path: str | Path,
    summary_output_path: str | Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 3407,
    bootstrap_confidence: float = 0.95,
    bootstrap_workers: int = 1,
    required_headroom_percentage_points: float = 0.20,
) -> dict[str, Any]:
    if len(expected_commit) != 40:
        raise ValueError("R0 expected commit must be exact")
    if not math.isclose(
        float(required_headroom_percentage_points),
        FROZEN_HEADROOM_PERCENTAGE_POINTS,
        abs_tol=0.0,
    ):
        raise ValueError("R0 headroom threshold is frozen at 0.20 percentage points")
    if int(bootstrap_seed) != FROZEN_BOOTSTRAP_SEED or not math.isclose(
        float(bootstrap_confidence), FROZEN_BOOTSTRAP_CONFIDENCE, abs_tol=0.0
    ):
        raise ValueError("R0 bootstrap seed/confidence protocol drift")
    evaluations = _parse_family_evaluations(family_evaluations)
    split = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
    )
    split_payload = json.loads(Path(split["manifest_path"]).read_text(encoding="utf-8"))
    holdout_videos = tuple(sorted(str(value) for value in split_payload["holdout_videos"]))
    train_videos = tuple(sorted(str(value) for value in split_payload["train_videos"]))
    checkpoint = _require_file(checkpoint_path, checkpoint_sha256, label="R0 checkpoint")
    config = _require_file(config_path, config_sha256, label="R0 replay config")
    artifact = _require_file(
        allocation_artifact_path,
        allocation_artifact_sha256,
        label="R0 allocation artifact",
    )
    family_summary = _require_file(
        family_summary_path,
        family_summary_sha256,
        label="R0 family summary",
    )
    _validate_family_artifact_chain(
        family_summary,
        artifact,
        holdout_videos=holdout_videos,
    )
    pretrain = _require_file(pretrain_path, pretrain_sha256, label="AdaTAD pretrain")
    blocked = _require_file(
        blocked_videos_path,
        blocked_videos_sha256,
        label="R0 blocked-video artifact",
    )
    blocked_values = json.loads(blocked.read_text(encoding="utf-8"))
    if not isinstance(blocked_values, list) or tuple(sorted(blocked_values)) != train_videos:
        raise RuntimeError("R0 evaluator blocked-video set differs from the train assignment")
    annotation = _require_file(
        split["annotation_path"],
        split["annotation_sha256"],
        label="R0 annotation",
    )

    rows = [
        _validate_evaluation(
            path,
            family=family,
            expected_commit=expected_commit,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=checkpoint_epoch,
            config=config,
            config_sha256=config_sha256,
            allocation_artifact=artifact,
            allocation_artifact_sha256=allocation_artifact_sha256,
            annotation=annotation,
            annotation_sha256=split["annotation_sha256"],
            blocked_videos=blocked,
            blocked_videos_sha256=blocked_videos_sha256,
            holdout_videos=holdout_videos,
            seed=bootstrap_seed,
        )
        for family, path in evaluations.items()
    ]
    class_map_hashes = {row["class_map_sha256"] for row in rows}
    class_map_paths = {row["class_map_path"] for row in rows}
    if len(class_map_hashes) != 1 or len(class_map_paths) != 1:
        raise RuntimeError("R0 family evaluations disagree on the class map")
    if len({row["evaluation_config_sha256"] for row in rows}) != 1:
        raise RuntimeError("R0 family evaluations disagree on evaluator config")

    evaluation_config = json.loads(
        evaluations[FAMILY_ORDER[0]].read_text(encoding="utf-8")
    )["evaluation_config"]
    bootstrap = bootstrap_official_map_differences(
        {row["family"]: row["prediction_path"] for row in rows},
        evaluation_config,
        baseline_family=FAMILY_ORDER[0],
        expected_video_ids=holdout_videos,
        expected_subset="training",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=bootstrap_confidence,
        workers=bootstrap_workers,
    )
    bootstrap.update(
        {
            "git_commit": expected_commit,
            "split_manifest_path": split["manifest_path"],
            "split_manifest_sha256": split_manifest_sha256,
            "assignment_sha256": split["assignment_sha256"],
            "prediction_sha256_by_family": {
                row["family"]: row["prediction_sha256"] for row in rows
            },
            "test_subset_consumed": False,
        }
    )
    bootstrap["bootstrap_sha256"] = canonical_sha256(bootstrap)
    bootstrap_path = Path(bootstrap_output_path).expanduser().resolve()
    if bootstrap_path.exists():
        raise FileExistsError(bootstrap_path)
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    by_family = {row["family"]: row for row in rows}
    uniform = by_family[FAMILY_ORDER[0]]["average_mAP"]
    required_fraction = float(required_headroom_percentage_points) / 100.0
    if required_fraction <= 0.0:
        raise ValueError("R0 required headroom must be positive")
    eligible: list[str] = []
    for family in FAMILY_ORDER[1:]:
        comparison = bootstrap["comparisons"][family]
        by_family[family]["headroom_vs_uniform_average_mAP"] = (
            by_family[family]["average_mAP"] - uniform
        )
        by_family[family]["headroom_bootstrap_ci_lower"] = comparison[
            "headroom_ci_lower"
        ]
        by_family[family]["headroom_bootstrap_ci_upper"] = comparison[
            "headroom_ci_upper"
        ]
        if (
            family in PROJECTED_FAMILIES
            and comparison["headroom_ci_lower"] > required_fraction
        ):
            eligible.append(family)
    selected = next((family for family in PROJECTED_FAMILIES if family in eligible), None)
    summary = {
        "schema": SCHEMA,
        "ok": selected is not None,
        "status": "GO_TO_P0" if selected is not None else "KILL_PROJECTED_FEASIBLE_SET",
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "source_subset": "training_internal_holdout",
        "test_subset_consumed": False,
        "family_order": list(FAMILY_ORDER),
        "projected_family_order_weak_to_strong": list(PROJECTED_FAMILIES),
        "unrestricted_family": UNRESTRICTED_FAMILY,
        "selected_weakest_projected_family": selected,
        "eligible_projected_families": eligible,
        "required_headroom_percentage_points": float(required_headroom_percentage_points),
        "required_headroom_average_mAP_fraction": required_fraction,
        "decision_rule": "paired video-cluster bootstrap lower CI strictly exceeds the frozen headroom; choose the first weak-to-strong projected family",
        "split_manifest_path": split["manifest_path"],
        "split_manifest_sha256": split_manifest_sha256,
        "assignment_sha256": split["assignment_sha256"],
        "annotation_path": str(annotation),
        "annotation_sha256": split["annotation_sha256"],
        "train_block_list_path": split["train_block_list"],
        "train_block_list_sha256": split["train_block_list_sha256"],
        "holdout_block_list_path": split["holdout_block_list"],
        "holdout_block_list_sha256": split["holdout_block_list_sha256"],
        "evaluation_blocked_videos_path": str(blocked),
        "evaluation_blocked_videos_sha256": blocked_videos_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_state_key": "state_dict_ema",
        "config_path": str(config),
        "config_sha256": config_sha256,
        "allocation_artifact_path": str(artifact),
        "allocation_artifact_sha256": allocation_artifact_sha256,
        "family_summary_path": str(family_summary),
        "family_summary_sha256": family_summary_sha256,
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha256,
        "class_map_path": rows[0]["class_map_path"],
        "class_map_sha256": rows[0]["class_map_sha256"],
        "bootstrap_path": str(bootstrap_path),
        "bootstrap_file_sha256": sha256_file(bootstrap_path),
        "bootstrap_self_sha256": bootstrap["bootstrap_sha256"],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_confidence": float(bootstrap_confidence),
        "rows": rows,
        "runtime_gt_input_to_selector": False,
        "diagnostic_only": True,
        "absolute_map_paper_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = Path(summary_output_path).expanduser().resolve()
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def revalidate_r0_summary(
    *,
    summary_path: str | Path,
    summary_file_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Fail-closed P0 consumer for the sealed R0 producer artifact.

    The producer already executes the official evaluator for every bootstrap
    resample.  Repeating all 4,000 evaluator calls in every downstream
    consumer does not strengthen the sealed evidence and used to delay P0 by
    hours.  Consumers instead reopen the source predictions/evaluations,
    recompute each family's official point estimate, and recompute all
    bootstrap differences and quantiles from the hashed producer samples.
    """

    source = _require_file(summary_path, summary_file_sha256, label="R0 summary")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != SCHEMA
        or summary.get("ok") is not True
        or summary.get("git_commit") != expected_commit
        or summary.get("source_subset") != "training_internal_holdout"
        or summary.get("test_subset_consumed") is not False
        or tuple(summary.get("family_order", ())) != FAMILY_ORDER
        or tuple(summary.get("projected_family_order_weak_to_strong", ()))
        != PROJECTED_FAMILIES
        or summary.get("unrestricted_family") != UNRESTRICTED_FAMILY
    ):
        raise RuntimeError("R0 summary identity/family contract mismatch")
    _self_hash(summary, "summary_sha256", label="R0 summary")
    split = validate_split_manifest(
        summary["split_manifest_path"],
        expected_manifest_sha256=summary["split_manifest_sha256"],
        annotation_path=summary["annotation_path"],
        train_block_list=summary["train_block_list_path"],
        holdout_block_list=summary["holdout_block_list_path"],
    )
    if (
        split["assignment_sha256"] != summary.get("assignment_sha256")
        or split["annotation_sha256"] != summary.get("annotation_sha256")
        or split["train_block_list_sha256"]
        != summary.get("train_block_list_sha256")
        or split["holdout_block_list_sha256"]
        != summary.get("holdout_block_list_sha256")
    ):
        raise RuntimeError("R0 split reference hash mismatch")
    split_payload = json.loads(Path(split["manifest_path"]).read_text(encoding="utf-8"))
    holdout_videos = tuple(sorted(str(value) for value in split_payload["holdout_videos"]))
    train_videos = tuple(sorted(str(value) for value in split_payload["train_videos"]))
    checkpoint = _require_file(
        summary["checkpoint_path"], summary["checkpoint_sha256"], label="R0 checkpoint"
    )
    config = _require_file(
        summary["config_path"], summary["config_sha256"], label="R0 replay config"
    )
    artifact = _require_file(
        summary["allocation_artifact_path"],
        summary["allocation_artifact_sha256"],
        label="R0 allocation artifact",
    )
    family_summary = _require_file(
        summary["family_summary_path"],
        summary["family_summary_sha256"],
        label="R0 family summary",
    )
    _validate_family_artifact_chain(
        family_summary,
        artifact,
        holdout_videos=holdout_videos,
    )
    _require_file(
        summary["pretrain_path"], summary["pretrain_sha256"], label="AdaTAD pretrain"
    )
    blocked = _require_file(
        summary["evaluation_blocked_videos_path"],
        summary["evaluation_blocked_videos_sha256"],
        label="R0 blocked-video artifact",
    )
    blocked_values = json.loads(blocked.read_text(encoding="utf-8"))
    if not isinstance(blocked_values, list) or tuple(sorted(blocked_values)) != train_videos:
        raise RuntimeError("R0 blocked-video set differs from the train assignment")
    annotation = _require_file(
        summary["annotation_path"], summary["annotation_sha256"], label="R0 annotation"
    )
    raw_rows = summary.get("rows")
    if not isinstance(raw_rows, list) or tuple(
        str(row.get("family")) for row in raw_rows if isinstance(row, Mapping)
    ) != FAMILY_ORDER:
        raise RuntimeError("R0 row order mismatch")
    reopened = []
    for raw in raw_rows:
        family = str(raw["family"])
        evaluation = _require_file(
            raw["evaluation_path"],
            raw["evaluation_file_sha256"],
            label=f"R0 evaluation {family}",
        )
        row = _validate_evaluation(
            evaluation,
            family=family,
            expected_commit=expected_commit,
            checkpoint=checkpoint,
            checkpoint_sha256=summary["checkpoint_sha256"],
            checkpoint_epoch=int(summary["checkpoint_epoch"]),
            config=config,
            config_sha256=summary["config_sha256"],
            allocation_artifact=artifact,
            allocation_artifact_sha256=summary["allocation_artifact_sha256"],
            annotation=annotation,
            annotation_sha256=summary["annotation_sha256"],
            blocked_videos=blocked,
            blocked_videos_sha256=summary["evaluation_blocked_videos_sha256"],
            holdout_videos=holdout_videos,
            seed=int(summary.get("bootstrap_seed", 3407)),
        )
        for field in (
            "headroom_vs_uniform_average_mAP",
            "headroom_bootstrap_ci_lower",
            "headroom_bootstrap_ci_upper",
        ):
            if field in raw:
                row[field] = float(raw[field])
        for field, value in row.items():
            if raw.get(field) != value:
                raise RuntimeError(f"R0 copied row mismatch for {family}: {field}")
        reopened.append(row)
    if any(
        row["class_map_path"] != summary.get("class_map_path")
        or row["class_map_sha256"] != summary.get("class_map_sha256")
        for row in reopened
    ):
        raise RuntimeError("R0 class-map binding mismatch")
    if len({row["evaluation_config_sha256"] for row in reopened}) != 1:
        raise RuntimeError("R0 family evaluator configs disagree")

    bootstrap_path = _require_file(
        summary["bootstrap_path"],
        summary["bootstrap_file_sha256"],
        label="R0 bootstrap",
    )
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    if bootstrap.get("schema") != BOOTSTRAP_SCHEMA:
        raise RuntimeError("R0 bootstrap schema mismatch")
    _self_hash(bootstrap, "bootstrap_sha256", label="R0 bootstrap")
    if bootstrap.get("bootstrap_sha256") != summary.get("bootstrap_self_sha256"):
        raise RuntimeError("R0 bootstrap self-hash copy mismatch")
    if (
        int(bootstrap.get("samples", -1)) != int(summary.get("bootstrap_samples", -2))
        or int(bootstrap.get("seed", -1)) != int(summary.get("bootstrap_seed", -2))
        or not math.isclose(
            float(bootstrap.get("confidence", -1.0)),
            float(summary.get("bootstrap_confidence", -2.0)),
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("R0 bootstrap protocol copy mismatch")
    if int(bootstrap["seed"]) != FROZEN_BOOTSTRAP_SEED or not math.isclose(
        float(bootstrap["confidence"]),
        FROZEN_BOOTSTRAP_CONFIDENCE,
        abs_tol=0.0,
    ):
        raise RuntimeError("R0 frozen bootstrap protocol drift")
    evaluation_config = json.loads(
        Path(reopened[0]["evaluation_path"]).read_text(encoding="utf-8")
    )["evaluation_config"]
    if (
        bootstrap.get("official_evaluator_reexecuted_per_resample") is not True
        or bootstrap.get("paired_video_cluster_bootstrap") is not True
        or bootstrap.get("baseline_family") != FAMILY_ORDER[0]
        or tuple(bootstrap.get("family_order", ())) != FAMILY_ORDER
        or tuple(bootstrap.get("video_ids", ())) != holdout_videos
        or bootstrap.get("evaluation_config") != evaluation_config
        or bootstrap.get("evaluation_config_sha256")
        != canonical_sha256(evaluation_config)
        or bootstrap.get("git_commit") != expected_commit
        or bootstrap.get("split_manifest_path") != split["manifest_path"]
        or bootstrap.get("split_manifest_sha256")
        != summary["split_manifest_sha256"]
        or bootstrap.get("assignment_sha256") != split["assignment_sha256"]
        or bootstrap.get("prediction_sha256_by_family")
        != {row["family"]: row["prediction_sha256"] for row in reopened}
        or bootstrap.get("test_subset_consumed") is not False
    ):
        raise RuntimeError("R0 sealed bootstrap identity mismatch")

    import numpy as np

    samples = int(bootstrap["samples"])
    sampled = bootstrap.get("sampled_average_mAP")
    comparisons = bootstrap.get("comparisons")
    if (
        not isinstance(sampled, Mapping)
        or tuple(sampled) != FAMILY_ORDER
        or not isinstance(comparisons, Mapping)
        or tuple(comparisons) != FAMILY_ORDER[1:]
    ):
        raise RuntimeError("R0 sealed bootstrap family order mismatch")
    arrays: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        values = np.asarray(sampled[family], dtype=np.float64)
        if (
            values.shape != (samples,)
            or not np.isfinite(values).all()
            or bool((values < 0.0).any())
            or bool((values > 1.0).any())
        ):
            raise RuntimeError(f"R0 sealed bootstrap samples are invalid for {family}")
        arrays[family] = values
    alpha = (1.0 - float(bootstrap["confidence"])) / 2.0
    baseline = arrays[FAMILY_ORDER[0]]
    for family in FAMILY_ORDER[1:]:
        delta = arrays[family] - baseline
        expected_comparison = {
            "headroom_samples": [float(value) for value in delta],
            "headroom_mean": float(delta.mean()),
            "headroom_ci_lower": float(np.quantile(delta, alpha)),
            "headroom_ci_upper": float(np.quantile(delta, 1.0 - alpha)),
        }
        if comparisons.get(family) != expected_comparison:
            raise RuntimeError(
                f"R0 sealed bootstrap arithmetic mismatch for {family}"
            )

    required_pp = float(summary["required_headroom_percentage_points"])
    required_fraction = float(summary["required_headroom_average_mAP_fraction"])
    if not math.isclose(required_fraction, required_pp / 100.0, abs_tol=1.0e-15):
        raise RuntimeError("R0 headroom unit conversion mismatch")
    if not math.isclose(
        required_pp, FROZEN_HEADROOM_PERCENTAGE_POINTS, abs_tol=0.0
    ):
        raise RuntimeError("R0 frozen headroom threshold drift")
    uniform = reopened[0]["average_mAP"]
    eligible = []
    for family in PROJECTED_FAMILIES:
        row = next(item for item in reopened if item["family"] == family)
        comparison = bootstrap["comparisons"][family]
        expected_headroom = row["average_mAP"] - uniform
        if not math.isclose(
            row["headroom_vs_uniform_average_mAP"],
            expected_headroom,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"R0 headroom point estimate mismatch for {family}")
        if not math.isclose(
            row["headroom_bootstrap_ci_lower"],
            float(comparison["headroom_ci_lower"]),
            abs_tol=1.0e-12,
        ) or not math.isclose(
            row["headroom_bootstrap_ci_upper"],
            float(comparison["headroom_ci_upper"]),
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"R0 bootstrap CI copy mismatch for {family}")
        if float(comparison["headroom_ci_lower"]) > required_fraction:
            eligible.append(family)
    selected = next((family for family in PROJECTED_FAMILIES if family in eligible), None)
    if (
        selected is None
        or summary.get("selected_weakest_projected_family") != selected
        or summary.get("eligible_projected_families") != eligible
    ):
        raise RuntimeError("R0 weakest-feasible-family decision mismatch")
    return {
        "schema": "duca_r0_headroom_gate_v2",
        "ok": True,
        "git_commit": expected_commit,
        "r0_summary_path": str(source),
        "r0_summary_sha256": summary_file_sha256,
        "selected_weakest_projected_family": selected,
        "eligible_projected_families": eligible,
        "required_headroom_percentage_points": required_pp,
        "required_headroom_average_mAP_fraction": required_fraction,
        "average_mAP": {row["family"]: row["average_mAP"] for row in reopened},
        "bootstrap_self_sha256": bootstrap["bootstrap_sha256"],
        "official_evaluator_reexecuted_per_resample": True,
        "consumer_revalidated_sealed_bootstrap_without_reexecution": True,
        "test_subset_consumed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--family-evaluation", action="append", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--allocation-artifact", required=True)
    parser.add_argument("--allocation-artifact-sha256", required=True)
    parser.add_argument("--family-summary", required=True)
    parser.add_argument("--family-summary-sha256", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--pretrain-sha256", required=True)
    parser.add_argument("--blocked-videos", required=True)
    parser.add_argument("--blocked-videos-sha256", required=True)
    parser.add_argument("--bootstrap-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    parser.add_argument("--required-headroom-percentage-points", type=float, default=0.20)
    args = parser.parse_args(argv)
    summary = finalize_r0(
        expected_commit=args.expected_commit,
        family_evaluations=args.family_evaluation,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_epoch=args.checkpoint_epoch,
        config_path=args.config,
        config_sha256=args.config_sha256,
        allocation_artifact_path=args.allocation_artifact,
        allocation_artifact_sha256=args.allocation_artifact_sha256,
        family_summary_path=args.family_summary,
        family_summary_sha256=args.family_summary_sha256,
        pretrain_path=args.pretrain,
        pretrain_sha256=args.pretrain_sha256,
        blocked_videos_path=args.blocked_videos,
        blocked_videos_sha256=args.blocked_videos_sha256,
        bootstrap_output_path=args.bootstrap_output,
        summary_output_path=args.summary_output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_workers=args.bootstrap_workers,
        required_headroom_percentage_points=args.required_headroom_percentage_points,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
