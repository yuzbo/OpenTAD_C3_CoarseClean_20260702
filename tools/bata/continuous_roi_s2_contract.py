from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path


CONTINUOUS_ROI_S2_SCHEMA = "continuous_roi_s2_preregistration_v2_1"
CONTINUOUS_ROI_S2_AUDIT_SCHEMA = "continuous_roi_s2_static_audit_v1"
CONTINUOUS_ROI_S2_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "methods"
    / "continuous_roi_s2_v2_1_protocol.json"
)
EXPECTED_TRAINED_FAMILIES = ("D160", "G96", "U128")
EXPECTED_GEOMETRY_FAMILIES = ("anchor", "fixed_size", "variable_size")
EXPECTED_CELLS = (
    "ANCHOR",
    "FS-RAND",
    "VS-RAND",
    "FS-PREF",
    "VS-PREF",
    "D0-FIX",
    "D0-PREF",
)
EXPECTED_OUTCOMES = (
    "NO_DECISION_INVALID_EVIDENCE",
    "INCONCLUSIVE_GEOMETRY_OR_SUPPORT",
    "INCONCLUSIVE_REFERENCE_SUPPORT",
    "SUFFICIENT_AND_VARIABLE_SIZE_HEADROOM",
    "SUFFICIENT_BUT_COST_NOT_VIABLE",
    "SUFFICIENT_CONTINUOUS_NO_VARIABLE_SIZE_HEADROOM",
    "SUFFICIENT_FIXED_SIZE_ONLY",
    "REFERENCE_REPRESENTATION_INSUFFICIENT",
)


def canonical_json_bytes(payload: Mapping) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def protocol_core_sha256(payload: Mapping) -> str:
    core = copy.deepcopy(dict(payload))
    core.pop("declared_protocol_sha256", None)
    return canonical_sha256(core)


def finalize_self_hash(payload: Mapping, hash_key: str) -> dict:
    checked = copy.deepcopy(dict(payload))
    checked.pop(hash_key, None)
    checked[hash_key] = canonical_sha256(checked)
    return checked


def load_protocol(path: str | Path = CONTINUOUS_ROI_S2_PROTOCOL_PATH) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Continuous-RoI S2 protocol must be a JSON object")
    return dict(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: Mapping, expected, name: str) -> None:
    _require(
        set(value) == set(expected),
        f"{name} keys changed: expected={sorted(expected)} got={sorted(value)}",
    )


def _validate_identity(payload: Mapping, checks: list[str]) -> None:
    _require(
        payload.get("schema_version") == CONTINUOUS_ROI_S2_SCHEMA,
        "Continuous-RoI S2 schema changed",
    )
    declared = payload.get("declared_protocol_sha256")
    _require(
        isinstance(declared, str) and len(declared) == 64,
        "declared protocol SHA-256 is missing",
    )
    actual = protocol_core_sha256(payload)
    _require(actual == declared, "declared protocol SHA-256 does not match")
    parent = payload.get("parent_research_commit")
    _require(
        isinstance(parent, str) and len(parent) == 40,
        "parent research commit must be immutable",
    )
    checks.append("identity")


def _validate_scope(payload: Mapping, checks: list[str]) -> None:
    scope = payload["scope"]
    _require(scope["route"] == "offline_tad_spatial_crop", "route changed")
    _require(
        scope["s2_question"] == "pre_policy_representation_sufficiency",
        "S2 scientific question changed",
    )
    _require(scope["s3_separate"] is True, "S2/S3 must remain separate")
    _require(
        scope["learned_policy_authorized"] is False,
        "S2 must not authorize a learned policy",
    )
    _require(
        scope["official_test_open_allowed"] is False,
        "S2 must keep official test sealed",
    )
    _require(
        payload["data"]["official_test_sealed"] is True,
        "data contract must keep official test sealed",
    )
    checks.append("scope")


def _decode_box(
    sx: float,
    sy: float,
    sa: float,
    sr: float,
    geometry: Mapping,
) -> tuple[float, float, float, float]:
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-value))
    area_min = float(geometry["area_min"])
    area_max = float(geometry["area_max"])
    ratio_min = float(geometry["ratio_min"])
    ratio_max = float(geometry["ratio_max"])
    source_aspect = float(geometry["source_aspect"])
    area = area_min + (area_max - area_min) * sigmoid(sa)
    ratio = math.exp(
        math.log(ratio_min)
        + math.log(ratio_max / ratio_min) * sigmoid(sr)
    )
    width = math.sqrt(area * ratio / source_aspect)
    height = math.sqrt(area * source_aspect / ratio)
    center_x = 0.5 * width + (1.0 - width) * sigmoid(sx)
    center_y = 0.5 * height + (1.0 - height) * sigmoid(sy)
    return center_x, center_y, width, height


def _validate_geometry(payload: Mapping, checks: list[str]) -> None:
    geometry = payload["geometry"]
    _require(payload["data"]["source_hw"] == [180, 320], "source geometry changed")
    _require(geometry["chunks"] == 48, "clip count changed")
    _require(geometry["frames_per_chunk"] == 16, "clip length changed")
    _require(geometry["knots"] == 12, "knot count changed")
    _require(geometry["chunks_per_knot"] == 4, "knot cadence changed")
    _require(
        geometry["decoder"] == "bounded_area_pixel_aspect_v1",
        "geometry decoder changed",
    )
    _require(
        geometry["posthoc_repair_allowed"] is False,
        "post-hoc box repair is forbidden",
    )
    _require(
        float(geometry["area_min"]) < float(geometry["area_max"]),
        "invalid area bounds",
    )
    _require(
        float(geometry["ratio_min"]) < float(geometry["ratio_max"]),
        "invalid ratio bounds",
    )
    anchor = geometry["anchor"]
    decoded = _decode_box(
        *map(float, anchor["logits"]),
        geometry=geometry,
    )
    expected = (
        float(anchor["cx"]),
        float(anchor["cy"]),
        float(anchor["width"]),
        float(anchor["height"]),
    )
    _require(
        max(abs(left - right) for left, right in zip(decoded, expected)) <= 1e-12,
        "anchor logits no longer decode to the S1 source box",
    )
    for sx in (-20.0, 20.0):
        for sy in (-20.0, 20.0):
            for sa in (-20.0, 20.0):
                for sr in (-20.0, 20.0):
                    cx, cy, width, height = _decode_box(
                        sx, sy, sa, sr, geometry
                    )
                    box = (
                        cx - 0.5 * width,
                        cy - 0.5 * height,
                        cx + 0.5 * width,
                        cy + 0.5 * height,
                    )
                    _require(all(math.isfinite(item) for item in box), "non-finite box")
                    _require(
                        min(box) >= -1e-12 and max(box) <= 1.0 + 1e-12,
                        "decoder is not analytically in bounds",
                    )
    checks.append("geometry")


def _validate_views_and_models(payload: Mapping, checks: list[str]) -> None:
    views = payload["views"]
    _require(views["dense"] == 160, "dense view changed")
    _require(views["global"] == 96, "global view changed")
    _require(views["local"] == 128, "local view changed")
    _require(
        views["global"] ** 2 + views["local"] ** 2 == views["dense"] ** 2,
        "global/local and dense pixel budgets no longer match",
    )
    _require(
        views["pixels_per_frame_dense"] == views["pixels_per_frame_u128"] == 25600,
        "declared pixel budgets changed",
    )
    _require(
        views["temporal_points_detector"] == 768,
        "detector temporal contract changed",
    )

    models = payload["models"]
    _require(
        tuple(models["trained_families"]) == EXPECTED_TRAINED_FAMILIES,
        "trained model families changed",
    )
    _require(models["u128_contains_selector"] is False, "U128 contains a selector")
    _require(
        models["u128_shared_videomae_instances"] == 1,
        "U128 must instantiate one shared VideoMAE",
    )
    _require(
        models["u128_videomae_evaluations"] == 2,
        "U128 must charge two VideoMAE evaluations",
    )
    params = models["new_u128_parameters"]
    _require(params["policy_head"] == 0, "S2 must not add policy parameters")
    _require(
        params["fusion"] + params["auxiliary_heads"] + params["policy_head"]
        == params["total"]
        == 609449,
        "U128 parameter arithmetic changed",
    )
    checks.append("views_and_models")


def _validate_training_support(payload: Mapping, checks: list[str]) -> None:
    training = payload["training"]
    _require(training["policy_head_present"] is False, "S2 policy head reappeared")
    _require(
        training["family_source"] == "exogenous_stateless_geometry",
        "S2 geometry is no longer exogenous",
    )
    _require(
        training["geometry_receives_detector_gradient"] is False,
        "S2 geometry must not masquerade as a learned policy",
    )
    _require(
        training["scheduler_and_ema_on_success_only"] is True,
        "scheduler/EMA must follow successful updates only",
    )
    _require(
        tuple(training["family_cycle"]) == EXPECTED_GEOMETRY_FAMILIES,
        "common-support family cycle changed",
    )
    family_counts = training["family_counts"]
    _require_exact_keys(
        family_counts,
        EXPECTED_GEOMETRY_FAMILIES,
        "training.family_counts",
    )
    expected_samples = (
        training["successful_updates"]
        * training["samples_per_successful_update"]
    )
    _require(
        sum(family_counts.values()) == expected_samples,
        "geometry-family exposure count does not match successful updates",
    )
    _require(
        len(set(family_counts.values())) == 1,
        "common-support geometry families are not balanced",
    )
    _require(
        training["optimizer_updates_per_epoch"] * training["epochs"]
        == training["successful_updates"]
        == 4800,
        "successful-update schedule changed",
    )
    _require(
        training["models_per_seed"] * len(training["seeds"])
        == training["total_runs"]
        == 9,
        "training run count changed",
    )

    models = payload["models"]
    cell_contracts = models["cell_contracts"]
    _require_exact_keys(cell_contracts, EXPECTED_CELLS, "models.cell_contracts")
    _require(
        set(models["decision_cells"]) == set(EXPECTED_CELLS),
        "decision cell list changed",
    )
    for name, cell in cell_contracts.items():
        _require(
            cell["checkpoint_family"] == "U128",
            f"{name} does not use the common-support U128 checkpoint",
        )
        _require(
            cell["geometry_family"] in family_counts,
            f"{name} geometry is outside the U128 training support",
        )

    fixed = cell_contracts["FS-PREF"]
    variable = cell_contracts["VS-PREF"]
    _require(
        fixed["candidate_count"] == variable["candidate_count"] == 17,
        "fixed/variable references have unequal candidate budgets",
    )
    _require(
        fixed["gt_privilege"]
        == variable["gt_privilege"]
        == "post_seal_temporal_gate_gt",
        "fixed/variable references have unequal GT privilege",
    )
    _require(
        fixed["decision_critical"] is True
        and variable["decision_critical"] is True,
        "primary paired references must remain decision-critical",
    )
    checks.append("training_support")


def _validate_reference(payload: Mapping, checks: list[str]) -> None:
    reference = payload["candidate_reference"]
    _require(
        reference["confidence_optimization_allowed"] is False,
        "confidence optimization cannot certify S2 reference adequacy",
    )
    _require(
        reference["paired_center_trajectories"] is True,
        "fixed/variable center trajectories must be paired",
    )
    _require(
        reference["anchor_candidates"] + reference["non_anchor_candidates"]
        == reference["total_candidates"]
        == 17,
        "candidate population arithmetic changed",
    )
    prefixes = reference["nested_prefixes"]
    _require(
        prefixes == sorted(set(prefixes))
        and prefixes[0] == 1
        and prefixes[-1] == reference["total_candidates"],
        "nested candidate prefixes are invalid",
    )
    _require(
        reference["sobol_draw_shape"] == [16, 12, 4],
        "Sobol population shape changed",
    )
    join = reference["privileged_join"]
    _require(
        join["gt_visible_before_raw_seal"] is False,
        "gate GT may not be visible before raw seal",
    )
    _require(
        join["same_for_fixed_and_variable"] is True,
        "fixed/variable privileged joins must match",
    )
    adequacy = reference["search_adequacy"]
    _require(
        adequacy["confidence_convergence_is_evidence"] is False,
        "confidence convergence cannot be search adequacy evidence",
    )
    _require(
        adequacy["quartile_count_per_dimension"] == [4, 4, 4, 4],
        "result-independent Sobol support changed",
    )
    checks.append("reference")


def _validate_cost_and_resources(payload: Mapping, checks: list[str]) -> None:
    cost = payload["cost"]
    _require(
        cost["measured_roi_policy_head"] is False,
        "S2 cannot charge a measured policy head",
    )
    reserve = cost["future_selector_reserve"]
    _require(reserve["count"] == 1, "future selector reserve must be charged once")
    _require(
        cost["separate_ledgers"]
        == ["training", "finite_reference", "prospective_deployable"],
        "cost ledger families changed",
    )
    nvml = cost["nvml"]
    _require(
        float(nvml["target_interval_ms"]) == 20.0
        and float(nvml["maximum_gap_ms"]) == 100.0,
        "NVML cadence must retain the audited 20/100 ms contract",
    )
    _require(
        nvml["publication"] == "post_sampling_atomic_jsonl_v1",
        "NVML trace publication mode changed",
    )

    abba = payload["abba"]
    total = (
        abba["gate_windows"]
        * abba["seeds"]
        * abba["block_invocations"]
        * abba["blocks_per_seed_window"]
    )
    _require(total == abba["total_invocations"] == 1548, "ABBA total changed")
    _require(
        abba["invocations_per_arm"] * len(abba["arms"]) == total,
        "ABBA per-arm arithmetic is inconsistent",
    )
    _require(
        abba["block_invocations"] == 4 and len(abba["arms"]) == 2,
        "ABBA block must contain two invocations per arm",
    )

    resources = payload["resources"]
    _require(
        resources["outer_allocation"]
        == {"cpus": 8, "gpus": 2, "purpose": "site_memory_policy_only"},
        "outer Slurm allocation changed",
    )
    _require(
        resources["inner_step"]
        == {"cpus": 5, "gpus": 1, "memory_mib": 96000, "model_device": "cuda:0"},
        "inner Slurm step changed",
    )
    _require(
        resources["override_cuda_visible_devices"] is False,
        "CUDA_VISIBLE_DEVICES override is forbidden",
    )
    _require(
        resources["preserve_failed_namespace"] is True
        and resources["delete_failed_namespace"] is False,
        "failed formal namespaces must remain immutable",
    )
    _require(
        resources["max_namespace_bytes"] <= 16 * 1024**3,
        "formal namespace cap is not executable on audited storage",
    )
    _require(
        resources["minimum_free_space_bytes_above_estimate"] == 8 * 1024**3,
        "dynamic storage reserve changed",
    )
    _require(resources["python_no_user_site"] is True, "user site must be disabled")
    _require(resources["required_numpy"] == "1.23.5", "formal NumPy changed")
    checks.append("cost_and_resources")


def resolve_outcome(
    *,
    evidence_valid: bool,
    geometry_valid: bool,
    reference_adequate: bool,
    variable_sufficient: bool,
    fixed_sufficient: bool,
    variable_headroom: bool,
    cost_viable: bool,
) -> str:
    if not evidence_valid:
        return "NO_DECISION_INVALID_EVIDENCE"
    if not geometry_valid:
        return "INCONCLUSIVE_GEOMETRY_OR_SUPPORT"
    if not reference_adequate:
        return "INCONCLUSIVE_REFERENCE_SUPPORT"
    if variable_sufficient and variable_headroom and cost_viable:
        return "SUFFICIENT_AND_VARIABLE_SIZE_HEADROOM"
    if variable_sufficient and variable_headroom and not cost_viable:
        return "SUFFICIENT_BUT_COST_NOT_VIABLE"
    if variable_sufficient and not variable_headroom:
        return "SUFFICIENT_CONTINUOUS_NO_VARIABLE_SIZE_HEADROOM"
    if not variable_sufficient and fixed_sufficient:
        return "SUFFICIENT_FIXED_SIZE_ONLY"
    return "REFERENCE_REPRESENTATION_INSUFFICIENT"


def _validate_statistics_and_outcomes(payload: Mapping, checks: list[str]) -> None:
    statistics = payload["statistics"]
    _require(
        statistics["power_uses_s2_outputs"] is False,
        "result-blind power audit cannot use S2 outputs",
    )
    _require(
        statistics["bootstrap_replicates"] == 20000,
        "bootstrap replicate count changed",
    )
    _require(
        tuple(payload["outcomes"]) == EXPECTED_OUTCOMES,
        "outcome vocabulary changed",
    )
    observed = set()
    for state in range(1 << 7):
        values = [bool(state & (1 << bit)) for bit in range(7)]
        outcome = resolve_outcome(
            evidence_valid=values[0],
            geometry_valid=values[1],
            reference_adequate=values[2],
            variable_sufficient=values[3],
            fixed_sufficient=values[4],
            variable_headroom=values[5],
            cost_viable=values[6],
        )
        _require(outcome in EXPECTED_OUTCOMES, "state machine emitted unknown outcome")
        observed.add(outcome)
    _require(
        observed == set(EXPECTED_OUTCOMES),
        "state machine does not reach every registered outcome",
    )
    checks.append("statistics_and_outcomes")


def validate_protocol(payload: Mapping) -> dict:
    payload = copy.deepcopy(dict(payload))
    checks: list[str] = []
    _validate_identity(payload, checks)
    _validate_scope(payload, checks)
    _validate_geometry(payload, checks)
    _validate_views_and_models(payload, checks)
    _validate_training_support(payload, checks)
    _validate_reference(payload, checks)
    _validate_cost_and_resources(payload, checks)
    _validate_statistics_and_outcomes(payload, checks)
    audit = {
        "schema_version": CONTINUOUS_ROI_S2_AUDIT_SCHEMA,
        "protocol_schema": payload["schema_version"],
        "protocol_sha256": protocol_core_sha256(payload),
        "parent_research_commit": payload["parent_research_commit"],
        "checks": checks,
        "check_count": len(checks),
        "state_assignments_checked": 1 << 7,
        "static_protocol_valid": True,
        "implementation_authorized": True,
        "training_authorized": False,
        "official_test_open_allowed": False,
    }
    return finalize_self_hash(audit, "audit_sha256")
