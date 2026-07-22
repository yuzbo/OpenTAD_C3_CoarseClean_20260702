#!/usr/bin/env python3
"""Result schema and result-blind validation for GeoRoute-AdaTAD evidence.

This module contains no model, launcher, or evaluation code.  It validates
machine-readable run records before analysis and keeps development selection
separate from a later sealed official-test package.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


SCHEMA_VERSION = "georoute-paper-result-v1"
VALID_STAGES = {"P0", "P1", "P2", "P3"}
VALID_SPLIT_ROLES = {"development", "official_test"}
VALID_VARIANTS = {
    "dense_native",
    "fixed_lattice",
    "fixed_lattice_geometry",
    "random",
    "free_token_select",
    "roi_only",
    "roi_residual",
    "roi_residual_no_context",
    "roi_residual_stride4",
    "roi_residual_stride8",
    "roi_residual_no_absolute_coordinates",
    "tome",
    "amod",
    "roi_residual_amod",
}
REQUIRED_TIOU = ("0.3", "0.4", "0.5", "0.6", "0.7")
REQUIRED_COST_SCOPE = (
    "decode",
    "preprocess",
    "host_to_device",
    "scout",
    "route",
    "patch_embed",
    "backbone",
    "adapter",
    "detector",
    "nms",
)


class GeoRouteResultSchemaError(ValueError):
    """Raised when a record cannot be used as auditable paper evidence."""


def canonical_json_sha256(value: Any) -> str:
    """Return a stable digest for a JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load either a list JSON file or JSONL records without silently skipping rows."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise GeoRouteResultSchemaError(
                        f"{path}:{line_number}: invalid JSONL: {exc.msg}"
                    ) from exc
        return records
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise GeoRouteResultSchemaError("JSON result input must be a list of records")
    return payload


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeoRouteResultSchemaError(f"{path} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise GeoRouteResultSchemaError(f"{path} must be finite")
    return value


def _positive_number(value: Any, path: str) -> float:
    value = _finite_number(value, path)
    if value <= 0:
        raise GeoRouteResultSchemaError(f"{path} must be positive")
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeoRouteResultSchemaError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise GeoRouteResultSchemaError(f"{path} must be a non-empty string")
    return value


def record_identity(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Identity that must be unique in a result package."""

    return (
        record["study_id"],
        record["stage"],
        record["split_role"],
        record["dataset"],
        record["detector"],
        record["variant"],
        record["seed"],
        record["budget"]["tokens_per_tubelet"],
    )


def validate_record(record: Mapping[str, Any], *, development_only: bool = False) -> Dict[str, Any]:
    """Validate one evidence record and return a normalized shallow copy."""

    if not isinstance(record, Mapping):
        raise GeoRouteResultSchemaError("each result record must be an object")
    normalized = dict(record)
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise GeoRouteResultSchemaError(
            f"schema_version must equal {SCHEMA_VERSION!r}"
        )
    _require_string(normalized.get("study_id"), "study_id")
    stage = normalized.get("stage")
    if stage not in VALID_STAGES:
        raise GeoRouteResultSchemaError(f"stage must be one of {sorted(VALID_STAGES)}")
    split_role = normalized.get("split_role")
    if split_role not in VALID_SPLIT_ROLES:
        raise GeoRouteResultSchemaError(
            f"split_role must be one of {sorted(VALID_SPLIT_ROLES)}"
        )
    if development_only and split_role != "development":
        raise GeoRouteResultSchemaError(
            "development-only analysis refuses official_test evidence"
        )
    _require_string(normalized.get("dataset"), "dataset")
    _require_string(normalized.get("detector"), "detector")
    if normalized.get("variant") not in VALID_VARIANTS:
        raise GeoRouteResultSchemaError(
            f"variant must be one of {sorted(VALID_VARIANTS)}"
        )
    seed = normalized.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise GeoRouteResultSchemaError("seed must be an integer")

    budget = _require_mapping(normalized.get("budget"), "budget")
    tokens = budget.get("tokens_per_tubelet")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise GeoRouteResultSchemaError("budget.tokens_per_tubelet must be a positive integer")
    source_tokens = budget.get("source_tokens_per_tubelet")
    if not isinstance(source_tokens, int) or isinstance(source_tokens, bool) or source_tokens < tokens:
        raise GeoRouteResultSchemaError(
            "budget.source_tokens_per_tubelet must be an integer >= selected tokens"
        )
    selected = budget.get("selected_tokens_per_tubelet")
    if not isinstance(selected, int) or isinstance(selected, bool) or selected <= 0:
        raise GeoRouteResultSchemaError(
            "budget.selected_tokens_per_tubelet must be a positive integer"
        )
    if normalized["variant"] != "dense_native" and selected != tokens:
        raise GeoRouteResultSchemaError(
            "matched sparse variants must report selected_tokens_per_tubelet equal to budget"
        )
    if normalized["variant"] == "dense_native" and selected != source_tokens:
        raise GeoRouteResultSchemaError(
            "dense_native must report all source tokens as selected"
        )
    if normalized["variant"] == "dense_native" and tokens != source_tokens:
        raise GeoRouteResultSchemaError(
            "dense_native must set tokens_per_tubelet equal to source token count"
        )

    metrics = _require_mapping(normalized.get("metrics"), "metrics")
    _finite_number(metrics.get("average_map"), "metrics.average_map")
    map_by_tiou = _require_mapping(metrics.get("map_by_tiou"), "metrics.map_by_tiou")
    for threshold in REQUIRED_TIOU:
        _finite_number(map_by_tiou.get(threshold), f"metrics.map_by_tiou[{threshold}]")

    cost = _require_mapping(normalized.get("cost"), "cost")
    _positive_number(cost.get("end_to_end_p50_ms"), "cost.end_to_end_p50_ms")
    _positive_number(cost.get("end_to_end_p95_ms"), "cost.end_to_end_p95_ms")
    _positive_number(cost.get("peak_memory_mb"), "cost.peak_memory_mb")
    _positive_number(cost.get("gross_gpu_energy_j"), "cost.gross_gpu_energy_j")
    scope = _require_mapping(cost.get("scope"), "cost.scope")
    missing_scope = [name for name in REQUIRED_COST_SCOPE if scope.get(name) is not True]
    if missing_scope:
        raise GeoRouteResultSchemaError(
            "cost.scope must explicitly charge all components; missing true flags: "
            + ", ".join(missing_scope)
        )

    evidence = _require_mapping(normalized.get("evidence"), "evidence")
    for field in (
        "runtime_commit",
        "config_sha256",
        "checkpoint_sha256",
        "prediction_sha256",
        "run_receipt_sha256",
    ):
        _require_string(evidence.get(field), f"evidence.{field}")

    diagnostics = _require_mapping(normalized.get("diagnostics"), "diagnostics")
    if diagnostics.get("one_heavy_backbone_forward") is not True:
        raise GeoRouteResultSchemaError(
            "diagnostics.one_heavy_backbone_forward must be true"
        )
    if diagnostics.get("uses_grid_sample") is not False:
        raise GeoRouteResultSchemaError("diagnostics.uses_grid_sample must be false")
    if diagnostics.get("uses_resized_local_crop") is not False:
        raise GeoRouteResultSchemaError(
            "diagnostics.uses_resized_local_crop must be false"
        )
    _finite_number(diagnostics.get("packed_attention_tokens"), "diagnostics.packed_attention_tokens")
    _finite_number(diagnostics.get("packed_mlp_tokens"), "diagnostics.packed_mlp_tokens")

    estimator = normalized.get("policy_estimator")
    if estimator not in {"none", "score_function", "straight_through"}:
        raise GeoRouteResultSchemaError(
            "policy_estimator must be none, score_function, or straight_through"
        )
    if estimator == "straight_through" and normalized.get("estimator_bias_label") != "biased_surrogate":
        raise GeoRouteResultSchemaError(
            "straight_through records must carry estimator_bias_label='biased_surrogate'"
        )
    if estimator == "score_function" and normalized.get("score_function_kat_passed") is not True:
        raise GeoRouteResultSchemaError(
            "score_function records require score_function_kat_passed=true"
        )
    return normalized


def validate_records(
    records: Iterable[Mapping[str, Any]], *, development_only: bool = False
) -> List[Dict[str, Any]]:
    normalized = [validate_record(record, development_only=development_only) for record in records]
    if not normalized:
        raise GeoRouteResultSchemaError("result package must contain at least one record")
    identities: Dict[Tuple[Any, ...], int] = {}
    for index, record in enumerate(normalized):
        identity = record_identity(record)
        if identity in identities:
            raise GeoRouteResultSchemaError(
                f"duplicate run identity at records {identities[identity]} and {index}: {identity}"
            )
        identities[identity] = index
    study_ids = {record["study_id"] for record in normalized}
    if len(study_ids) != 1:
        raise GeoRouteResultSchemaError("one result package must contain exactly one study_id")
    return normalized


def grouped_records(records: Iterable[Mapping[str, Any]]) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["stage"],
            record["split_role"],
            record["dataset"],
            record["detector"],
            record["variant"],
            record["budget"]["tokens_per_tubelet"],
        )
        groups[key].append(record)
    return dict(groups)
