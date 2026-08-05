"""Summarize result-blind dynamic SCNR role-calibration telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROLE_ORDER = ("context", "roi", "residual")
FIELD_ORDER = (
    "q_base",
    "delta_roi",
    "delta_residual",
    "residual_minus_roi",
    "roi_minus_context",
    "residual_minus_context",
    "winning_modifier",
    "winner_top1_minus_top2_margin",
)
SCOPE_ORDER = ("valid", "selected", "unselected")
STATISTIC_ORDER = ("min", "p05", "p25", "p50", "p75", "p95", "max", "mean")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("role calibration summary requires a non-empty population")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    checked = [_finite(value, "role calibration statistic") for value in values]
    return {
        "min": min(checked),
        "p05": _quantile(checked, 0.05),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "max": max(checked),
        "mean": sum(checked) / len(checked),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_counts(value: Any, *, expected_total: int, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(ROLE_ORDER):
        raise ValueError(f"{label} role set changed")
    counts: dict[str, int] = {}
    for role in ROLE_ORDER:
        count = value[role]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label}.{role} must be one non-negative integer")
        counts[role] = int(count)
    if sum(counts.values()) != int(expected_total):
        raise ValueError(f"{label} role counts do not partition their population")
    return counts


def _validated_fractions(
    value: Any,
    *,
    counts: Mapping[str, int],
    expected_total: int,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(ROLE_ORDER):
        raise ValueError(f"{label} role set changed")
    fractions: dict[str, float] = {}
    for role in ROLE_ORDER:
        fraction = _finite(value[role], f"{label}.{role}")
        expected = counts[role] / float(expected_total)
        if not 0.0 <= fraction <= 1.0 or not math.isclose(
            fraction,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}.{role} disagrees with role counts")
        fractions[role] = fraction
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} fractions do not partition their population")
    return fractions


def _validate_selected_over_valid_ratios(
    value: Any,
    *,
    valid_fractions: Mapping[str, float],
    selected_fractions: Mapping[str, float],
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(ROLE_ORDER):
        raise ValueError("selected/valid role-fraction ratio role set changed")
    for role in ROLE_ORDER:
        expected = (
            selected_fractions[role] / valid_fractions[role]
            if valid_fractions[role] > 0.0
            else None
        )
        observed = value[role]
        if expected is None:
            if observed is not None:
                raise ValueError(
                    f"selected/valid role-fraction ratio.{role} must be null"
                )
            continue
        ratio = _finite(observed, f"selected/valid role-fraction ratio.{role}")
        if ratio < 0.0 or not math.isclose(
            ratio,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"selected/valid role-fraction ratio.{role} is inconsistent"
            )


def _validate_distribution(
    value: Any,
    *,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    required = {
        "count",
        *STATISTIC_ORDER,
        "std_population",
        "negative_count",
        "zero_count",
        "positive_count",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} distribution schema changed")
    count = value["count"]
    sign_counts = tuple(
        value[key] for key in ("negative_count", "zero_count", "positive_count")
    )
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or int(count) != int(expected_count)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in sign_counts
        )
        or sum(map(int, sign_counts)) != int(count)
    ):
        raise ValueError(f"{label} distribution counts are invalid")
    statistics = {
        key: _finite(value[key], f"{label}.{key}")
        for key in (*STATISTIC_ORDER, "std_population")
    }
    ordered = [
        statistics[key]
        for key in ("min", "p05", "p25", "p50", "p75", "p95", "max")
    ]
    if any(right < left for left, right in zip(ordered[:-1], ordered[1:])):
        raise ValueError(f"{label} distribution quantiles are not monotone")
    if statistics["std_population"] < 0.0:
        raise ValueError(f"{label} distribution standard deviation is negative")
    return {
        "count": int(count),
        **statistics,
        "negative_count": int(sign_counts[0]),
        "zero_count": int(sign_counts[1]),
        "positive_count": int(sign_counts[2]),
    }


def summarize_dynamic_role_calibration_telemetry(
    path: str | Path,
) -> dict[str, Any]:
    """Validate and summarize one complete development-only calibration replay."""

    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    dataset_count = int(payload.get("dataset_count", -1))
    if (
        payload.get("schema_version")
        not in {
            "georoute_formal_development_telemetry_v1",
            "georoute_diagnostic_telemetry_v1",
        }
        or payload.get("development_only") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("gt_for_route_used") is not False
        or payload.get("teacher_for_route_used") is not False
        or payload.get("oracle_used") is not False
        or payload.get("raw_prediction_cache_used") is not False
        or dataset_count <= 0
        or not isinstance(records, list)
        or len(records) != dataset_count
        or int(payload.get("record_count", -1)) != dataset_count
        or int(payload.get("unique_dataset_count", -1)) != dataset_count
        or int(payload.get("sampler_padding_count", -1)) != 0
    ):
        raise ValueError("dynamic role calibration population contract failed")
    dataset_indices = [
        record.get("dataset_index") if isinstance(record, Mapping) else None
        for record in records
    ]
    if (
        any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in dataset_indices
        )
        or len(set(dataset_indices)) != dataset_count
        or set(dataset_indices) != set(range(dataset_count))
    ):
        raise ValueError("dynamic role calibration dataset indices are invalid")

    aggregate_roles = {
        scope: Counter({role: 0 for role in ROLE_ORDER})
        for scope in SCOPE_ORDER
    }
    dominant_roles: Counter[str] = Counter()
    dominant_fractions: list[float] = []
    missing_role_windows: Counter[str] = Counter()
    distributions: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: {scope: [] for scope in SCOPE_ORDER} for field in FIELD_ORDER
    }

    for record in records:
        route = record.get("route")
        calibration = (
            route.get("policy_calibration") if isinstance(route, Mapping) else None
        )
        if (
            not isinstance(route, Mapping)
            or route.get("schema_version")
            != "georoute_dynamic_diagnostic_window_telemetry_v1"
            or route.get("measurement_scope")
            != "accuracy_replay_only_excluded_from_timed_cost"
            or not isinstance(calibration, Mapping)
            or calibration.get("schema_version")
            != "scnr_dynamic_role_calibration_window_v1"
            or calibration.get("measurement_scope")
            != "accuracy_replay_only_excluded_from_timed_cost"
            or calibration.get("diagnostic_only") is not True
            or calibration.get("changes_route_or_execution") is not False
            or calibration.get("role_target_fractions_used") is not False
            or calibration.get("fixed_role_quota_used") is not False
            or calibration.get("q_base_shared_across_roles") is not True
            or calibration.get("context_modifier_definition")
            != "exact_zero_baseline_no_learned_q_ctx"
            or calibration.get("gt_for_route_used") is not False
            or calibration.get("teacher_used") is not False
            or calibration.get("oracle_used") is not False
            or calibration.get("official_test_opened") is not False
            or calibration.get("paper_claim_allowed") is not False
            or tuple(calibration.get("role_order", ())) != ROLE_ORDER
        ):
            raise ValueError("dynamic role calibration window schema changed")

        population_counts = {
            scope: calibration.get(f"{scope}_candidate_count")
            for scope in SCOPE_ORDER
        }
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in population_counts.values()
        ) or (
            population_counts["selected"] + population_counts["unselected"]
            != population_counts["valid"]
        ):
            raise ValueError("dynamic role calibration candidate counts are invalid")

        current_roles = {
            scope: _validated_counts(
                calibration.get(f"{scope}_role_counts"),
                expected_total=population_counts[scope],
                label=f"{scope} role",
            )
            for scope in SCOPE_ORDER
        }
        current_fractions = {
            scope: _validated_fractions(
                calibration.get(f"{scope}_role_fractions"),
                counts=current_roles[scope],
                expected_total=population_counts[scope],
                label=f"{scope} role fraction",
            )
            for scope in SCOPE_ORDER
        }
        _validate_selected_over_valid_ratios(
            calibration.get("selected_over_valid_role_fraction_ratio"),
            valid_fractions=current_fractions["valid"],
            selected_fractions=current_fractions["selected"],
        )
        outer_selected = route.get("roles", {}).get("aggregate_counts")
        if dict(outer_selected or {}) != current_roles["selected"]:
            raise ValueError("dynamic role calibration differs from outer route roles")
        for scope in current_roles:
            aggregate_roles[scope].update(current_roles[scope])

        missing = [
            role for role in ROLE_ORDER if current_roles["selected"][role] == 0
        ]
        if calibration.get("selected_missing_roles") != missing:
            raise ValueError("dynamic role calibration missing-role list changed")
        missing_role_windows.update(missing)
        dominant = str(calibration.get("selected_dominant_role", ""))
        expected_dominant = max(
            ROLE_ORDER,
            key=lambda role: (
                current_roles["selected"][role],
                -ROLE_ORDER.index(role),
            ),
        )
        dominant_fraction = _finite(
            calibration.get("selected_dominant_role_fraction"),
            "selected dominant role fraction",
        )
        expected_fraction = current_roles["selected"][expected_dominant] / float(
            population_counts["selected"]
        )
        if dominant != expected_dominant or not math.isclose(
            dominant_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("dynamic role calibration dominant role is invalid")
        dominant_roles[dominant] += 1
        dominant_fractions.append(dominant_fraction)

        fields = calibration.get("fields")
        if not isinstance(fields, Mapping) or set(fields) != set(FIELD_ORDER):
            raise ValueError("dynamic role calibration field set changed")
        for field in FIELD_ORDER:
            scoped = fields[field]
            if not isinstance(scoped, Mapping) or set(scoped) != set(SCOPE_ORDER):
                raise ValueError(f"dynamic role calibration {field} scope set changed")
            for scope in SCOPE_ORDER:
                distributions[field][scope].append(
                    _validate_distribution(
                        scoped[scope],
                        expected_count=population_counts[scope],
                        label=f"{field}.{scope}",
                    )
                )

    role_summary: dict[str, Any] = {}
    for scope, counts in aggregate_roles.items():
        total = sum(counts.values())
        role_summary[scope] = {
            "counts": dict(counts),
            "fractions": {
                role: counts[role] / float(total) for role in ROLE_ORDER
            },
        }
    role_summary.update(
        {
            "dominant_window_counts": dict(dominant_roles),
            "dominant_fraction_across_windows": _summary(dominant_fractions),
            "windows_missing_role": {
                role: int(missing_role_windows[role]) for role in ROLE_ORDER
            },
            "role_balance_enforced": False,
            "context_modifier_definition": (
                "exact_zero_baseline_no_learned_q_ctx"
            ),
        }
    )

    field_summary: dict[str, Any] = {}
    for field in FIELD_ORDER:
        field_summary[field] = {}
        for scope in SCOPE_ORDER:
            rows = distributions[field][scope]
            total_count = sum(int(row["count"]) for row in rows)
            field_summary[field][scope] = {
                "count": total_count,
                "global_min": min(float(row["min"]) for row in rows),
                "global_max": max(float(row["max"]) for row in rows),
                "weighted_mean": sum(
                    float(row["mean"]) * int(row["count"]) for row in rows
                )
                / float(total_count),
                "negative_count": sum(int(row["negative_count"]) for row in rows),
                "zero_count": sum(int(row["zero_count"]) for row in rows),
                "positive_count": sum(int(row["positive_count"]) for row in rows),
                "window_statistics": {
                    statistic: _summary(
                        [float(row[statistic]) for row in rows]
                    )
                    for statistic in ("p05", "p50", "p95", "mean")
                },
                "quantile_scope": (
                    "distribution_across_per_window_quantiles_not_global_candidate_quantile"
                ),
            }

    population_sha256 = str(payload.get("population_sha256", ""))
    if len(population_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in population_sha256.lower()
    ):
        raise ValueError("dynamic role calibration population hash is invalid")
    return {
        "schema_version": "scnr_dynamic_role_calibration_summary_v1",
        "dataset_count": dataset_count,
        "record_count": len(records),
        "population_sha256": population_sha256,
        "telemetry_file_sha256": _sha256_file(path),
        "roles": role_summary,
        "fields": field_summary,
        "interpretation_boundary": {
            "diagnostic_only": True,
            "role_target_fractions_used": False,
            "changes_route_or_execution": False,
            "floor_selection_allowed": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = summarize_dynamic_role_calibration_telemetry(args.telemetry)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
