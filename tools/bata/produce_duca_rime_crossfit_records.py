from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bata.build_duca_rime_training_targets import (
    HARD_UTILITY_SCHEMA,
    OBSERVATION_SCHEMA,
)
from tools.bata.create_duca_rime_splits import TRAIN_ROLES, validate_rime_splits


MEASUREMENT_SCHEMA = "duca_rime_counterfactual_measurement_v1"
O3_SOURCE_SCHEMA = "duca_rime_o3_crossfit_prediction_v1"
O4_SOURCE_SCHEMA = "duca_rime_o4_calibrated_risk_prediction_v1"
PRICE_SOURCE_SCHEMA = "duca_rime_price_prediction_v1"
SUMMARY_SCHEMA = "duca_rime_crossfit_record_producer_v1"
DEFAULT_BUDGETS = (192, 256, 384, 512)
TARGET_FIT_ROLES = (
    "hard_label_generation",
    "utility_risk_fit",
    "dual_risk_calibration",
)


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
        raise ValueError(f"counterfactual JSONL must contain object rows: {source}")
    return source, rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in materialized
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "record_count": len(materialized),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    output["content_sha256"] = _canonical_sha256(output)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _finite_vector(value: Any, *, label: str, nonempty: bool = True) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a numeric vector")
    output = [float(item) for item in value]
    if (nonempty and not output) or not all(math.isfinite(item) for item in output):
        raise ValueError(f"{label} must be nonempty and finite")
    return output


def _finite_matrix(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{label} must be a nonempty numeric matrix")
    rows = [
        _finite_vector(row, label=f"{label}[{index}]")
        for index, row in enumerate(value)
    ]
    widths = {len(row) for row in rows}
    if len(widths) != 1 or widths == {0}:
        raise ValueError(f"{label} rows must have one nonzero width")
    return rows


def _solve(gram: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    size = len(rhs)
    matrix = [
        [float(value) for value in gram[row]] + [float(rhs[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) <= 1.0e-15:
            raise ValueError("ridge normal equation remained singular")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor == 0.0:
                continue
            matrix[row] = [
                left - factor * right
                for left, right in zip(matrix[row], matrix[column])
            ]
    return [matrix[index][-1] for index in range(size)]


def _fit_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    ridge: float,
) -> list[float]:
    if len(features) != len(targets) or not features:
        raise ValueError("ridge features/targets must be nonempty and aligned")
    width = len(features[0])
    if width < 1 or width > 64 or any(len(row) != width for row in features):
        raise ValueError("ridge feature width must be stable and in [1, 64]")
    design = [[1.0, *map(float, row)] for row in features]
    size = width + 1
    gram = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for row, target in zip(design, targets):
        target = float(target)
        if not math.isfinite(target):
            raise ValueError("ridge targets must be finite")
        for left in range(size):
            rhs[left] += row[left] * target
            for right in range(size):
                gram[left][right] += row[left] * row[right]
    regularizer = float(ridge)
    if not math.isfinite(regularizer) or regularizer <= 0.0:
        raise ValueError("ridge regularizer must be positive and finite")
    for index in range(size):
        gram[index][index] += regularizer
    return _solve(gram, rhs)


def _predict(coefficients: Sequence[float], features: Sequence[float]) -> float:
    if len(coefficients) != len(features) + 1:
        raise ValueError("ridge coefficient/feature width mismatch")
    return float(coefficients[0]) + sum(
        float(weight) * float(value)
        for weight, value in zip(coefficients[1:], features)
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _role_maps(manifest: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, set[str]]]:
    by_video = {}
    by_role = {}
    for role in TRAIN_ROLES:
        videos = {str(value) for value in manifest["train_roles"][role]["videos"]}
        by_role[role] = videos
        for video in videos:
            if video in by_video:
                raise ValueError("split manifest assigns one video to multiple train roles")
            by_video[video] = role
    return by_video, by_role


def _validate_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    budgets: tuple[int, ...],
    assignment_sha256: str,
    role_by_video: Mapping[str, str],
) -> list[dict[str, Any]]:
    normalized = []
    identities = set()
    budget_width = None
    frame_width = None
    frame_fit_width = None
    common_source_artifact_sha256 = None
    for line_number, row in enumerate(rows, start=1):
        prefix = f"counterfactual measurement line {line_number}"
        video = str(row.get("video_id", ""))
        start = int(row.get("window_start_frame", -1))
        provenance = row.get("provenance")
        row_budgets = tuple(int(value) for value in row.get("candidate_budgets", ()))
        budget_features = _finite_matrix(
            row.get("budget_features"),
            label=f"{prefix} budget_features",
        )
        actual_utility = _finite_vector(
            row.get("actual_utility"),
            label=f"{prefix} actual_utility",
        )
        failures = [int(value) for value in row.get("observed_pair_failure", ())]
        frame_features = _finite_matrix(
            row.get("frame_features"),
            label=f"{prefix} frame_features",
        )
        frame_fit_features = _finite_matrix(
            row.get("frame_fit_features"),
            label=f"{prefix} frame_fit_features",
        )
        frame_utility = _finite_vector(
            row.get("actual_frame_utility"),
            label=f"{prefix} actual_frame_utility",
        )
        frame_counterfactuals = row.get("frame_counterfactuals")
        ledgers = row.get("cost_ledger")
        expected_record_sha256 = str(row.get("record_sha256", ""))
        record_without_sha = dict(row)
        record_without_sha.pop("record_sha256", None)
        source_artifact_sha256 = str(
            (provenance or {}).get("source_artifact_sha256", "")
        )
        if (
            row.get("schema_version") != MEASUREMENT_SCHEMA
            or not video
            or video not in role_by_video
            or row.get("split_role") != role_by_video.get(video)
            or start < 0
            or (video, start) in identities
            or row_budgets != budgets
            or row.get("split_assignment_sha256") != assignment_sha256
            or not isinstance(provenance, Mapping)
            or provenance.get("measurement_kind")
            != "measured_detector_counterfactual"
            or provenance.get("fit_split") not in {"train", "training", "train_only"}
            or provenance.get("uses_official_final") is not False
            or provenance.get("uses_gt_for_supervision") is not True
            or provenance.get("uses_gt_at_deployment") is not False
            or provenance.get("uses_teacher_at_deployment") is not False
            or provenance.get("uses_prediction_cache_at_deployment") is not False
            or provenance.get("cheap_features_only_at_deployment") is not True
            or provenance.get("counterfactual_utility") is not True
            or provenance.get("proposal_score_surrogate_utility") is not False
            or provenance.get("pad_to_kmax") is not False
            or not isinstance(provenance.get("detector_checkpoint_sha256"), str)
            or len(str(provenance.get("detector_checkpoint_sha256"))) != 64
            or not isinstance(provenance.get("source_artifact_sha256"), str)
            or len(str(provenance.get("source_artifact_sha256"))) != 64
            or len(budget_features) != len(budgets)
            or len(actual_utility) != len(budgets)
            or len(failures) != len(budgets)
            or any(value not in {0, 1} for value in failures)
            or len(frame_fit_features) != len(frame_utility)
            or not isinstance(frame_counterfactuals, Sequence)
            or isinstance(frame_counterfactuals, (str, bytes))
            or len(frame_counterfactuals) != len(frame_fit_features)
            or len(expected_record_sha256) != 64
            or _canonical_sha256(record_without_sha) != expected_record_sha256
            or not isinstance(ledgers, Sequence)
            or isinstance(ledgers, (str, bytes))
            or len(ledgers) != len(budgets)
        ):
            raise ValueError(f"{prefix} violates the real train-only source contract")
        for budget, ledger in zip(budgets, ledgers):
            if (
                not isinstance(ledger, Mapping)
                or any(
                    int(ledger.get(key, -1)) != budget
                    for key in (
                        "requested_k",
                        "effective_k",
                        "unique_k",
                        "backbone_input_k",
                        "padded_k",
                    )
                )
                or ledger.get("max_gap_violation") is not False
            ):
                raise ValueError(f"{prefix} has a non-exact or padded cost ledger")
        for features, utility, counterfactual in zip(
            frame_fit_features,
            frame_utility,
            frame_counterfactuals,
        ):
            if (
                not isinstance(counterfactual, Mapping)
                or int(counterfactual.get("added_position", -1)) < 0
                or int(counterfactual.get("added_position", -1))
                >= len(frame_features)
                or int(counterfactual.get("removed_position", -1)) < 0
                or not math.isclose(
                    float(counterfactual.get("utility", math.nan)),
                    float(utility),
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                )
                or features
                != frame_features[int(counterfactual["added_position"])]
            ):
                raise ValueError(
                    f"{prefix} has a misbound frame counterfactual measurement"
                )
        current_budget_width = len(budget_features[0])
        current_frame_width = len(frame_features[0])
        current_frame_fit_width = len(frame_fit_features[0])
        if budget_width is None:
            budget_width = current_budget_width
            frame_width = current_frame_width
            frame_fit_width = current_frame_fit_width
            common_source_artifact_sha256 = source_artifact_sha256
        if (
            budget_width != current_budget_width
            or frame_width != current_frame_width
            or frame_fit_width != current_frame_fit_width
            or frame_width != frame_fit_width
            or common_source_artifact_sha256 != source_artifact_sha256
        ):
            raise ValueError("counterfactual feature dimensions drift across windows")
        identities.add((video, start))
        normalized.append(
            {
                "video_id": video,
                "window_start_frame": start,
                "role": role_by_video[video],
                "candidate_budgets": budgets,
                "budget_features": budget_features,
                "actual_utility": actual_utility,
                "observed_pair_failure": failures,
                "frame_features": frame_features,
                "frame_fit_features": frame_fit_features,
                "actual_frame_utility": frame_utility,
                "frame_counterfactuals": [
                    dict(value) for value in frame_counterfactuals
                ],
                "cost_ledger": [dict(value) for value in ledgers],
                "provenance": dict(provenance),
                "record_sha256": expected_record_sha256,
            }
        )
    return normalized


def _fit_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    fit_roles: Sequence[str],
    ridge: float,
) -> dict[str, Any]:
    allowed = set(map(str, fit_roles))
    fit_rows = [row for row in rows if row["role"] in allowed]
    fit_videos = sorted({str(row["video_id"]) for row in fit_rows})
    if len(fit_videos) < 3:
        raise ValueError("cross-fit producer requires at least three fit videos")
    budget_x, utility_y, risk_y = [], [], []
    frame_x, frame_y = [], []
    for row in fit_rows:
        budget_x.extend(row["budget_features"])
        utility_y.extend(row["actual_utility"])
        risk_y.extend(row["observed_pair_failure"])
        frame_x.extend(row["frame_fit_features"])
        frame_y.extend(row["actual_frame_utility"])
    utility = _fit_ridge(budget_x, utility_y, ridge=ridge)
    risk = _fit_ridge(budget_x, risk_y, ridge=ridge)
    frame = _fit_ridge(frame_x, frame_y, ridge=ridge)
    residuals = [
        abs(float(label) - min(1.0, max(0.0, _predict(risk, features))))
        for features, label in zip(budget_x, risk_y)
    ]
    return {
        "fit_roles": sorted(allowed),
        "fit_video_ids": fit_videos,
        "utility_coefficients": utility,
        "risk_coefficients": risk,
        "frame_coefficients": frame,
        "risk_absolute_residual_q95": _percentile(residuals, 0.95),
    }


def _provenance(
    *,
    model: Mapping[str, Any],
    eval_videos: Sequence[str],
    source_path: Path,
    source_sha256: str,
    split_manifest: Path,
    split_manifest_sha256: str,
    assignment_sha256: str,
    model_kind: str,
) -> dict[str, Any]:
    return {
        "fit_split": "train_only",
        "cross_fitted": True,
        "uses_validation_or_test": False,
        "fit_roles": list(model["fit_roles"]),
        "fit_video_ids": list(model["fit_video_ids"]),
        "eval_video_ids": sorted(map(str, eval_videos)),
        "split_manifest_path": str(split_manifest),
        "split_manifest_sha256": split_manifest_sha256,
        "split_assignment_sha256": assignment_sha256,
        "source_measurements_path": str(source_path),
        "source_measurements_sha256": source_sha256,
        "model_kind": model_kind,
        "uses_gt_for_supervision": True,
        "uses_gt_at_deployment": False,
        "uses_teacher_at_deployment": False,
        "uses_prediction_cache_at_deployment": False,
    }


def produce_crossfit_records(
    *,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    measurements_jsonl: str | Path,
    output_root: str | Path,
    candidate_budgets: Sequence[int] = DEFAULT_BUDGETS,
    ridge: float = 1.0e-3,
    risk_threshold: float = 0.25,
) -> dict[str, Any]:
    budgets = tuple(int(value) for value in candidate_budgets)
    if (
        len(budgets) < 3
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] <= 0
    ):
        raise ValueError("cross-fit candidate budgets must be increasing and contain >=3 K")
    if not 0.0 <= float(risk_threshold) <= 1.0:
        raise ValueError("risk threshold must lie in [0, 1]")
    split_path = Path(split_manifest).expanduser().resolve()
    split_sha = _sha256_file(split_path)
    if split_sha != str(split_manifest_sha256):
        raise ValueError("cross-fit split manifest SHA-256 drift")
    split_validation = validate_rime_splits(
        split_path,
        expected_sha256=split_manifest_sha256,
    )
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    role_by_video, videos_by_role = _role_maps(manifest)
    source_path, source_rows = _read_jsonl(measurements_jsonl)
    source_sha = _sha256_file(source_path)
    rows = _validate_measurements(
        source_rows,
        budgets=budgets,
        assignment_sha256=split_validation["assignment_sha256"],
        role_by_video=role_by_video,
    )
    covered = {str(row["video_id"]) for row in rows}
    required_roles = set(TARGET_FIT_ROLES) | {
        "detector_selector_train",
        "certification_development",
        "utility_risk_fit",
        "dual_risk_calibration",
    }
    for role in required_roles:
        if not videos_by_role[role] <= covered:
            missing = sorted(videos_by_role[role] - covered)
            raise ValueError(
                f"counterfactual measurements do not cover role {role}: {missing[:3]}"
            )

    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh cross-fit output root is required: {output}")
    output.mkdir(parents=True)

    def model_for(eval_role: str, fit_roles: Sequence[str]) -> dict[str, Any]:
        if eval_role in set(fit_roles):
            raise ValueError("cross-fit evaluation role must be disjoint from fit roles")
        return _fit_models(rows, fit_roles=fit_roles, ridge=ridge)

    o3_role = "utility_risk_fit"
    o3_model = model_for(
        o3_role,
        ("hard_label_generation", "dual_risk_calibration"),
    )
    o4_role = "dual_risk_calibration"
    o4_model = model_for(
        o4_role,
        ("hard_label_generation", "utility_risk_fit"),
    )
    target_role = "detector_selector_train"
    target_model = model_for(target_role, TARGET_FIT_ROLES)
    o2_role = "certification_development"
    o2_model = model_for(o2_role, TARGET_FIT_ROLES)

    def eval_rows(role: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["role"] == role]

    o3_eval = eval_rows(o3_role)
    o3_videos = sorted({row["video_id"] for row in o3_eval})
    o3_provenance = _provenance(
        model=o3_model,
        eval_videos=o3_videos,
        source_path=source_path,
        source_sha256=source_sha,
        split_manifest=split_path,
        split_manifest_sha256=split_sha,
        assignment_sha256=split_validation["assignment_sha256"],
        model_kind="ridge_counterfactual_utility",
    )
    o3_rows = []
    for row in o3_eval:
        for budget_index, budget in enumerate(budgets):
            common = {
                "schema_version": O3_SOURCE_SCHEMA,
                "video_id": row["video_id"],
                "window_start_frame": row["window_start_frame"],
                "budget": budget,
                "actual_gain": float(row["actual_utility"][budget_index]),
                "provenance": o3_provenance,
            }
            o3_rows.append(
                {
                    **common,
                    "score_family": "learned",
                    "predicted_gain": _predict(
                        o3_model["utility_coefficients"],
                        row["budget_features"][budget_index],
                    ),
                }
            )
            o3_rows.append(
                {
                    **common,
                    "score_family": "constant_null",
                    "predicted_gain": 0.0,
                }
            )
            o3_rows.append(
                {
                    **common,
                    "score_family": "budget_only_null",
                    "predicted_gain": budget / float(budgets[-1]),
                }
            )

    o4_eval = eval_rows(o4_role)
    o4_videos = sorted({row["video_id"] for row in o4_eval})
    o4_provenance = _provenance(
        model=o4_model,
        eval_videos=o4_videos,
        source_path=source_path,
        source_sha256=source_sha,
        split_manifest=split_path,
        split_manifest_sha256=split_sha,
        assignment_sha256=split_validation["assignment_sha256"],
        model_kind="ridge_counterfactual_pair_risk",
    )
    o4_rows = []
    price_rows = []
    for row in o4_eval:
        predicted_utility = [
            _predict(o4_model["utility_coefficients"], features)
            for features in row["budget_features"]
        ]
        predicted_risk = [
            min(
                1.0,
                max(
                    0.0,
                    _predict(o4_model["risk_coefficients"], features),
                ),
            )
            for features in row["budget_features"]
        ]
        risk_upper = [
            min(1.0, value + o4_model["risk_absolute_residual_q95"])
            for value in predicted_risk
        ]
        fallback_to_kmax = all(
            value > float(risk_threshold) for value in predicted_risk[:-1]
        )
        for index, budget in enumerate(budgets):
            ledger = row["cost_ledger"][index]
            o4_rows.append(
                {
                    "schema_version": O4_SOURCE_SCHEMA,
                    "video_id": row["video_id"],
                    "window_start_frame": row["window_start_frame"],
                    "budget": budget,
                    "predicted_risk": predicted_risk[index],
                    "observed_pair_failure": int(
                        row["observed_pair_failure"][index]
                    ),
                    "requested_k": int(ledger["requested_k"]),
                    "effective_k": int(ledger["effective_k"]),
                    "unique_k": int(ledger["unique_k"]),
                    "backbone_input_k": int(ledger["backbone_input_k"]),
                    "padded_k": int(ledger["padded_k"]),
                    "risk_fallback": bool(
                        fallback_to_kmax and budget == budgets[-1]
                    ),
                    "provenance": o4_provenance,
                }
            )
        price_rows.append(
            {
                "schema_version": PRICE_SOURCE_SCHEMA,
                "video_id": row["video_id"],
                "window_start_frame": row["window_start_frame"],
                "candidate_budgets": list(budgets),
                "predicted_utility": predicted_utility,
                "predicted_risk": predicted_risk,
                "risk_upper": risk_upper,
                "provenance": o4_provenance,
            }
        )

    target_eval = eval_rows(target_role)
    target_videos = sorted({row["video_id"] for row in target_eval})
    target_provenance = _provenance(
        model=target_model,
        eval_videos=target_videos,
        source_path=source_path,
        source_sha256=source_sha,
        split_manifest=split_path,
        split_manifest_sha256=split_sha,
        assignment_sha256=split_validation["assignment_sha256"],
        model_kind="ridge_crossfit_training_target",
    )
    observation_rows = []
    hard_rows = []
    for row in target_eval:
        for index, budget in enumerate(budgets):
            observation_rows.append(
                {
                    "schema_version": OBSERVATION_SCHEMA,
                    "video_id": row["video_id"],
                    "window_start_frame": row["window_start_frame"],
                    "budget": budget,
                    "utility": _predict(
                        target_model["utility_coefficients"],
                        row["budget_features"][index],
                    ),
                    "observed_pair_failure": int(
                        row["observed_pair_failure"][index]
                    ),
                    "provenance": target_provenance,
                }
            )
        hard_rows.append(
            {
                "schema_version": HARD_UTILITY_SCHEMA,
                "video_id": row["video_id"],
                "window_start_frame": row["window_start_frame"],
                "hard_frame_utility": [
                    _predict(target_model["frame_coefficients"], features)
                    for features in row["frame_features"]
                ],
                "provenance": target_provenance,
            }
        )

    artifacts = {
        "o3": _write_jsonl(output / "o3_source.jsonl", o3_rows),
        "o4": _write_jsonl(output / "o4_source.jsonl", o4_rows),
        "price": _write_jsonl(output / "price_source.jsonl", price_rows),
        "observations": _write_jsonl(
            output / "budget_observations.jsonl",
            observation_rows,
        ),
        "hard_utility": _write_jsonl(
            output / "hard_frame_utility.jsonl",
            hard_rows,
        ),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "produced",
        "claim_scope": "train_only_crossfit_records_not_gate_result",
        "source_measurements": {
            "path": str(source_path),
            "sha256": source_sha,
            "schema_version": MEASUREMENT_SCHEMA,
            "measurement_kind": "measured_detector_counterfactual",
            "proposal_score_surrogate_utility": False,
        },
        "split_manifest": {
            "path": str(split_path),
            "sha256": split_sha,
            "assignment_sha256": split_validation["assignment_sha256"],
        },
        "candidate_budgets": list(budgets),
        "ridge": float(ridge),
        "risk_threshold": float(risk_threshold),
        "models": {
            "o3": o3_model,
            "o4_price": o4_model,
            "training_targets": target_model,
            "o2_decoder": {
                **o2_model,
                "eval_role": o2_role,
                "runtime_decoder_api": "decode_rime_panel",
                "claim_scope": (
                    "counterfactual_detector_objective_decoder_family_regret_"
                    "not_tad_map"
                ),
            },
        },
        "artifacts": artifacts,
        "official_final_subset_consumed": False,
        "uses_gt_at_deployment": False,
        "uses_teacher_at_deployment": False,
        "uses_prediction_cache_at_deployment": False,
    }
    _write_json(output / "producer_summary.json", summary)
    summary["output_path"] = str(output / "producer_summary.json")
    summary["output_sha256"] = _sha256_file(output / "producer_summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit leakage-free train-only utility/risk models from real detector "
            "counterfactuals and emit DUCA-RIME O3/O4/price/target records."
        )
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--measurements-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--candidate-budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--risk-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)
    result = produce_crossfit_records(
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        measurements_jsonl=args.measurements_jsonl,
        output_root=args.output_root,
        candidate_budgets=args.candidate_budgets,
        ridge=args.ridge,
        risk_threshold=args.risk_threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
