#!/usr/bin/env python3
"""Authorize only categorical SCNR role analysis across legacy/strict triplets."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from tools.bata.analyze_georoute_dynamic_role_calibration import (
    ROLE_ORDER,
    SCOPE_ORDER,
    _finite,
    _validated_counts,
    _validated_fractions,
    _validate_selected_over_valid_ratios,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.run_georoute_phase_m_replay import _atomic_write_json, _inside
from tools.bata.run_georoute_role_instrumentation_pair import (
    _validate_formal_telemetry,
)


ANALYSIS_SCHEMA = "scnr_categorical_role_invariance_analysis_v1"
LEGACY_TRIPLET_SCHEMA = "georoute_role_instrumentation_causal_triplet_v1"
STRICT_TRIPLET_SCHEMA = "georoute_role_instrumentation_strict_causal_triplet_v2"
LEGACY_STATUS = "FAIL_BASELINE_REPLAY_NONDETERMINISM"
STRICT_STATUS = "PASS_STRICT_TRIPLET_NEUTRALITY_SOURCE_REPLAY_DRIFT_DIAGNOSTIC_ONLY"
MODE_ORDER = ("role_off_a", "role_off_b", "role_on")
PAYLOAD_ORDER = (
    "legacy_role_off_a",
    "legacy_role_off_b",
    "legacy_role_on",
    "strict_role_off_a",
    "strict_role_off_b",
    "strict_role_on",
)
ROUTE_EXCLUDED_KEYS = frozenset({"geometry", "policy_calibration"})
POLICY_EXCLUDED_KEYS = frozenset({"fields"})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    value = dict(payload)
    claimed = value.pop(field, None)
    if not isinstance(claimed, str) or canonical_sha256(value) != claimed:
        raise ValueError(f"{label} self-hash is invalid")


def _indexed_records(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = payload.get("records")
    dataset_count = payload.get("dataset_count")
    if (
        isinstance(dataset_count, bool)
        or not isinstance(dataset_count, int)
        or dataset_count <= 0
        or not isinstance(records, list)
        or len(records) != dataset_count
    ):
        raise ValueError("categorical invariance population is incomplete")
    indexed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("categorical invariance contains a malformed record")
        index = record.get("dataset_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in indexed:
            raise ValueError("categorical invariance dataset indices are invalid")
        indexed[int(index)] = record
    if set(indexed) != set(range(dataset_count)):
        raise ValueError("categorical invariance dataset population changed")
    return indexed


def _route_without_exclusions(route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in route.items()
        if key not in ROUTE_EXCLUDED_KEYS
    }


def _policy_without_continuous_fields(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in policy.items()
        if key not in POLICY_EXCLUDED_KEYS
    }


def summarize_categorical_role_invariance_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the bridge and summarize only backend-invariant role categories."""

    if set(payloads) != set(PAYLOAD_ORDER):
        raise ValueError("categorical invariance requires the complete six payloads")
    populations = {
        str(payload.get("population_sha256", "")) for payload in payloads.values()
    }
    dataset_counts = {
        int(payload.get("dataset_count", -1)) for payload in payloads.values()
    }
    if len(populations) != 1 or len(dataset_counts) != 1:
        raise ValueError("categorical invariance population binding changed")
    population_sha256 = next(iter(populations))
    dataset_count = next(iter(dataset_counts))
    if (
        len(population_sha256) != 64
        or any(character not in "0123456789abcdef" for character in population_sha256)
        or dataset_count <= 0
    ):
        raise ValueError("categorical invariance population receipt is invalid")

    indexed = {name: _indexed_records(payload) for name, payload in payloads.items()}
    categorical_routes: dict[str, list[dict[str, Any]]] = {}
    geometry_payloads: dict[str, list[Any]] = {}
    for name in PAYLOAD_ORDER:
        routes: list[dict[str, Any]] = []
        geometries: list[Any] = []
        expects_policy = name.endswith("role_on")
        for index in range(dataset_count):
            route = indexed[name][index].get("route")
            if not isinstance(route, Mapping):
                raise ValueError("categorical invariance route record is missing")
            has_policy = "policy_calibration" in route
            if has_policy is not expects_policy:
                raise ValueError("categorical invariance treatment presence changed")
            routes.append(_route_without_exclusions(route))
            geometries.append(copy.deepcopy(route.get("geometry")))
        categorical_routes[name] = routes
        geometry_payloads[name] = geometries
    route_reference = categorical_routes[PAYLOAD_ORDER[0]]
    if any(categorical_routes[name] != route_reference for name in PAYLOAD_ORDER[1:]):
        raise ValueError(
            "categorical route payload changed across replay/backend modes"
        )

    policies: dict[str, list[Mapping[str, Any]]] = {}
    categorical_policies: dict[str, list[dict[str, Any]]] = {}
    continuous_fields: dict[str, list[Any]] = {}
    for name in ("legacy_role_on", "strict_role_on"):
        current: list[Mapping[str, Any]] = []
        categorical: list[dict[str, Any]] = []
        fields: list[Any] = []
        for index in range(dataset_count):
            route = indexed[name][index]["route"]
            policy = route.get("policy_calibration")
            if not isinstance(policy, Mapping) or "fields" not in policy:
                raise ValueError("categorical role policy schema changed")
            current.append(policy)
            categorical.append(_policy_without_continuous_fields(policy))
            fields.append(copy.deepcopy(policy["fields"]))
        policies[name] = current
        categorical_policies[name] = categorical
        continuous_fields[name] = fields
    if categorical_policies["legacy_role_on"] != categorical_policies["strict_role_on"]:
        raise ValueError("categorical role policy changed across SDPA backends")

    aggregate_roles = {
        scope: Counter({role: 0 for role in ROLE_ORDER}) for scope in SCOPE_ORDER
    }
    aggregate_candidates = Counter({scope: 0 for scope in SCOPE_ORDER})
    dominant_roles: Counter[str] = Counter()
    missing_role_windows: Counter[str] = Counter()
    source_records = indexed["legacy_role_on"]
    for index in range(dataset_count):
        route = source_records[index]["route"]
        calibration = route["policy_calibration"]
        population_counts = {
            scope: calibration.get(f"{scope}_candidate_count") for scope in SCOPE_ORDER
        }
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in population_counts.values()
        ) or (
            population_counts["selected"] + population_counts["unselected"]
            != population_counts["valid"]
        ):
            raise ValueError("categorical role candidate counts are invalid")
        current_roles = {
            scope: _validated_counts(
                calibration.get(f"{scope}_role_counts"),
                expected_total=population_counts[scope],
                label=f"{scope} categorical role",
            )
            for scope in SCOPE_ORDER
        }
        current_fractions = {
            scope: _validated_fractions(
                calibration.get(f"{scope}_role_fractions"),
                counts=current_roles[scope],
                expected_total=population_counts[scope],
                label=f"{scope} categorical role fraction",
            )
            for scope in SCOPE_ORDER
        }
        _validate_selected_over_valid_ratios(
            calibration.get("selected_over_valid_role_fraction_ratio"),
            valid_fractions=current_fractions["valid"],
            selected_fractions=current_fractions["selected"],
        )
        if (
            dict(route.get("roles", {}).get("aggregate_counts", {}))
            != current_roles["selected"]
        ):
            raise ValueError("categorical policy differs from outer selected roles")
        missing = [role for role in ROLE_ORDER if current_roles["selected"][role] == 0]
        dominant = max(
            ROLE_ORDER,
            key=lambda role: (
                current_roles["selected"][role],
                -ROLE_ORDER.index(role),
            ),
        )
        dominant_fraction = _finite(
            calibration.get("selected_dominant_role_fraction"),
            "selected dominant categorical role fraction",
        )
        expected_fraction = current_roles["selected"][dominant] / float(
            population_counts["selected"]
        )
        if (
            calibration.get("selected_missing_roles") != missing
            or calibration.get("selected_dominant_role") != dominant
            or abs(dominant_fraction - expected_fraction) > 1e-12
        ):
            raise ValueError("categorical role dominance metadata is inconsistent")
        for scope in SCOPE_ORDER:
            aggregate_roles[scope].update(current_roles[scope])
            aggregate_candidates[scope] += int(population_counts[scope])
        dominant_roles[dominant] += 1
        missing_role_windows.update(missing)

    role_summary: dict[str, Any] = {}
    for scope in SCOPE_ORDER:
        counts = aggregate_roles[scope]
        total = int(aggregate_candidates[scope])
        if sum(counts.values()) != total:
            raise ValueError("aggregate categorical role partition changed")
        role_summary[scope] = {
            "candidate_count": total,
            "counts": {role: int(counts[role]) for role in ROLE_ORDER},
            "fractions": {role: counts[role] / float(total) for role in ROLE_ORDER},
        }
    role_summary.update(
        dominant_window_counts={role: int(dominant_roles[role]) for role in ROLE_ORDER},
        windows_missing_role={
            role: int(missing_role_windows[role]) for role in ROLE_ORDER
        },
        role_balance_enforced=False,
        context_modifier_definition="exact_zero_baseline_no_learned_q_ctx",
    )
    legacy_fields = continuous_fields["legacy_role_on"]
    strict_fields = continuous_fields["strict_role_on"]
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "dataset_count": dataset_count,
        "population_sha256": population_sha256,
        "categorical_invariance": {
            "all_six_route_payloads_equal_after_excluding_geometry_and_policy": True,
            "route_categorical_sha256": canonical_sha256(route_reference),
            "legacy_vs_strict_policy_categorical_records_equal": dataset_count,
            "policy_categorical_sha256": canonical_sha256(
                categorical_policies["legacy_role_on"]
            ),
            "geometry_payload_parity": all(
                geometry_payloads[name] == geometry_payloads[PAYLOAD_ORDER[0]]
                for name in PAYLOAD_ORDER[1:]
            ),
            "continuous_policy_fields_parity": legacy_fields == strict_fields,
            "legacy_continuous_fields_sha256": canonical_sha256(legacy_fields),
            "strict_continuous_fields_sha256": canonical_sha256(strict_fields),
        },
        "roles": role_summary,
        "interpretation_boundary": {
            "categorical_role_analysis_allowed": True,
            "continuous_score_calibration_analysis_allowed": False,
            "geometry_analysis_allowed": False,
            "performance_analysis_allowed": False,
            "floor_selection_allowed": False,
            "model_repair_selection_allowed": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        },
    }


def _load_triplet_result(root: Path, *, strict: bool) -> dict[str, Any]:
    path = root / "triplet_result.json"
    payload = _read_json(path)
    _verify_self_hash(payload, field="result_sha256", label="triplet result")
    expected_schema = STRICT_TRIPLET_SCHEMA if strict else LEGACY_TRIPLET_SCHEMA
    expected_status = STRICT_STATUS if strict else LEGACY_STATUS
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("status") != expected_status
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
        or payload.get("role_calibration_statistics_summarized") is not False
    ):
        raise ValueError("triplet result cannot support categorical invariance")
    comparison_path = Path(payload["prediction_integrity_comparison_path"]).resolve()
    if (
        not _inside(comparison_path, root)
        or sha256_file(comparison_path)
        != payload["prediction_integrity_comparison_sha256"]
    ):
        raise ValueError("triplet comparison receipt changed")
    comparisons = _read_json(comparison_path)
    if strict:
        parity_keys = (
            "role_off_a_vs_role_off_b",
            "role_off_a_vs_role_on",
            "role_off_b_vs_role_on",
        )
        if (
            payload.get("strict_deterministic_algorithms") is not True
            or payload.get("sdp_backend") != "math"
            or payload.get("deterministic_override_changes_heavy_execution") is not True
            or payload.get("source_execution_reproduced") is not False
            or any(
                comparisons.get(key, {}).get("raw_sha256_parity") is not True
                for key in parity_keys
            )
        ):
            raise ValueError("strict triplet neutrality did not close")
    elif (
        comparisons.get("role_off_a_vs_role_off_b", {}).get("raw_sha256_parity")
        is not False
    ):
        raise ValueError("legacy replay did not expose the registered SDPA drift")
    return payload


def build_categorical_role_invariance_analysis(
    *,
    legacy_root: str | Path,
    strict_root: str | Path,
    variant: str,
) -> dict[str, Any]:
    legacy_root = Path(legacy_root).resolve()
    strict_root = Path(strict_root).resolve()
    legacy_result = _load_triplet_result(legacy_root, strict=False)
    strict_result = _load_triplet_result(strict_root, strict=True)
    if (
        legacy_result.get("variant") != variant
        or strict_result.get("variant") != variant
        or legacy_result.get("seed") != strict_result.get("seed")
        or legacy_result.get("source_experiment_commit")
        != strict_result.get("source_experiment_commit")
        or legacy_result.get("source_population_sha256")
        != strict_result.get("source_population_sha256")
        or legacy_result.get("source_dataset_count")
        != strict_result.get("source_dataset_count")
        or legacy_result.get("source_artifacts")
        != strict_result.get("source_artifacts")
    ):
        raise ValueError("legacy/strict triplet lineage changed")
    for key in ("bound_config", "checkpoint", "prediction"):
        path_key = f"{key}_path"
        sha_key = f"{key}_sha256"
        artifact = Path(strict_result["source_artifacts"][path_key]).resolve()
        if sha256_file(artifact) != strict_result["source_artifacts"][sha_key]:
            raise ValueError(f"source {key} artifact changed")

    payloads: dict[str, Mapping[str, Any]] = {}
    artifact_receipts: dict[str, Any] = {}
    for prefix, root, result in (
        ("legacy", legacy_root, legacy_result),
        ("strict", strict_root, strict_result),
    ):
        for mode in MODE_ORDER:
            mode_result = result.get("modes", {}).get(mode)
            if not isinstance(mode_result, Mapping):
                raise ValueError("triplet mode receipt is missing")
            telemetry_path = Path(mode_result["telemetry_path"]).resolve()
            if (
                not _inside(telemetry_path, root)
                or sha256_file(telemetry_path) != mode_result["telemetry_sha256"]
            ):
                raise ValueError("triplet telemetry artifact changed")
            telemetry = _read_json(telemetry_path)
            binding = telemetry.get("phase_m_binding")
            if not isinstance(binding, Mapping):
                raise ValueError("triplet telemetry binding is missing")
            _validate_formal_telemetry(
                telemetry,
                expected_binding=binding,
                expected_population_sha256=result["source_population_sha256"],
                expected_dataset_count=int(result["source_dataset_count"]),
                role_calibration_enabled=mode == "role_on",
            )
            payloads[f"{prefix}_{mode}"] = telemetry
            artifact_receipts[f"{prefix}_{mode}"] = {
                "telemetry_path": str(telemetry_path),
                "telemetry_sha256": mode_result["telemetry_sha256"],
            }
    summary = summarize_categorical_role_invariance_payloads(payloads)
    result = {
        **summary,
        "variant": variant,
        "seed": int(strict_result["seed"]),
        "source_experiment_commit": strict_result["source_experiment_commit"],
        "legacy_runtime_commit": legacy_result["runtime_commit"],
        "strict_runtime_commit": strict_result["runtime_commit"],
        "legacy_triplet_root": str(legacy_root),
        "strict_triplet_root": str(strict_root),
        "legacy_triplet_result_sha256": sha256_file(
            legacy_root / "triplet_result.json"
        ),
        "strict_triplet_result_sha256": sha256_file(
            strict_root / "triplet_result.json"
        ),
        "artifact_receipts": artifact_receipts,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_categorical_role_invariance_analysis(
        legacy_root=args.legacy_root,
        strict_root=args.strict_root,
        variant=args.variant,
    )
    _atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
