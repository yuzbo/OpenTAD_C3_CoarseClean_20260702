"""Result-blind GeoRoute-AdaTAD development experiment contract.

The contract deliberately separates the native dense upper bound from the
matched-K controls.  It is not an official-test or paper-result entrypoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GEOROUTE_EXPERIMENT_SCHEMA = "georoute_adatad_experiment_contract_v2"
DEVELOPMENT_SEEDS = (3407, 3408, 3409)
P1_EPOCHS = 20
P2_EPOCHS = 60
P3_EPOCHS = 60
MATCHED_K = 64

# The implementation names stay concise, while figures and tables use names
# that make the scientific comparison explicit.  This mapping is deliberately
# one-to-one: a result writer may never silently relabel an implementation
# variant after observing metrics.
PAPER_VARIANT_NAMES = {
    "dense_native": "dense_native",
    "fixed_lattice": "fixed_lattice",
    "fixed_lattice_geometry": "fixed_lattice_geometry",
    "random": "random",
    "free": "free_token_select",
    "roi": "roi_only",
    "hybrid": "roi_residual",
    "hybrid_no_context": "roi_residual_no_context",
    "hybrid_roi_stride4": "roi_residual_stride4",
    "hybrid_roi_stride8": "roi_residual_stride8",
    "hybrid_no_absolute_coordinates": "roi_residual_no_absolute_coordinates",
}


# The dense native run is an upper-compute reference.  Fair selection claims
# are made only among fixed_lattice/random/free/roi/hybrid, all of which use K.
VARIANTS: dict[str, dict[str, Any]] = {
    "dense_native": dict(
        route_mode="dense",
        policy_estimator="none",
        tokens_per_tubelet=None,
        context_tokens=0,
        roi_fraction=0.0,
        description="native full-token upper-compute reference",
        matched_k=False,
    ),
    "fixed_lattice": dict(
        route_mode="uniform",
        policy_estimator="none",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.0,
        description="deterministic spatial lattice exact-K control",
        matched_k=True,
    ),
    "fixed_lattice_geometry": dict(
        route_mode="uniform",
        policy_estimator="none",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.0,
        geometry_side_channel=True,
        description="fixed lattice with the learned geometry adapter side-channel but no geometry-based selection",
        matched_k=True,
    ),
    "random": dict(
        route_mode="random",
        policy_estimator="none",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.0,
        description="data-independent seeded random exact-K control",
        matched_k=True,
    ),
    "free": dict(
        route_mode="free",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=0,
        roi_fraction=0.0,
        description="free residual TokenSelect-only control",
        matched_k=True,
    ),
    "roi": dict(
        route_mode="roi",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=0,
        roi_fraction=1.0,
        description="continuous geometry ROI-only route",
        matched_k=True,
    ),
    "hybrid": dict(
        route_mode="hybrid",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.50,
        description="continuous ROI plus residual free-token route",
        matched_k=True,
    ),
    "hybrid_no_context": dict(
        route_mode="hybrid",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=0,
        roi_fraction=0.50,
        description="hybrid route without deterministic context",
        matched_k=True,
    ),
    "hybrid_roi_stride4": dict(
        route_mode="hybrid",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.50,
        geometry_stride_tubelets=2,
        description="hybrid route with four-source-frame geometry knots",
        matched_k=True,
    ),
    "hybrid_roi_stride8": dict(
        route_mode="hybrid",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.50,
        geometry_stride_tubelets=4,
        description="hybrid route with eight-source-frame geometry knots",
        matched_k=True,
    ),
    "hybrid_no_absolute_coordinates": dict(
        route_mode="hybrid",
        policy_estimator="straight_through",
        tokens_per_tubelet=MATCHED_K,
        context_tokens=8,
        roi_fraction=0.50,
        absolute_position_enabled=False,
        absolute_coordinates_enabled=False,
        description="hybrid route without absolute source-coordinate encoding",
        matched_k=True,
    ),
}

P1_VARIANTS = (
    "dense_native",
    "fixed_lattice",
    "fixed_lattice_geometry",
    "random",
    "free",
    "roi",
    "hybrid",
)
P3_BUDGET_VARIANTS = (32, 48, 64, 96)
P3_ABLATION_VARIANTS = (
    "hybrid_no_context",
    "hybrid_roi_stride4",
    "hybrid_roi_stride8",
    "hybrid_no_absolute_coordinates",
)


def paper_variant_name(implementation_variant: str) -> str:
    """Return the frozen paper label for one implementation variant."""

    try:
        return PAPER_VARIANT_NAMES[implementation_variant]
    except KeyError as exc:
        raise ValueError(f"no paper label is registered for {implementation_variant!r}") from exc


def stage_epochs(stage: str) -> int:
    try:
        return {"p1": P1_EPOCHS, "p2": P2_EPOCHS, "p3": P3_EPOCHS}[stage]
    except KeyError as exc:
        raise ValueError("GeoRoute stage must be p1, p2, or p3") from exc


def stage_cell_relative_path(
    *,
    stage: str,
    variant: str,
    seed: int,
    token_budget: int | None,
) -> Path:
    """Return the immutable relative namespace for one development cell.

    The exact-K budget is a part of a P3 cell's experimental identity.  In
    particular, a single method/seed occurs at K=32/48/64/96 in the frozen
    budget curve; collapsing those runs into one directory would overwrite
    evidence rather than create a comparison.
    """

    if stage not in {"p1", "p2", "p3"}:
        raise ValueError("GeoRoute stage must be p1, p2, or p3")
    if variant not in VARIANTS:
        raise ValueError(f"unsupported GeoRoute variant {variant!r}")
    if int(seed) not in DEVELOPMENT_SEEDS:
        raise ValueError("GeoRoute seed is outside the frozen development seed set")
    if token_budget is not None and int(token_budget) <= 0:
        raise ValueError("GeoRoute token budget must be positive when provided")
    budget_label = "matched_k" if token_budget is None else f"k{int(token_budget)}"
    return Path(stage) / variant / budget_label / f"seed{int(seed)}"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean_text(value: str, *, name: str) -> str:
    value = str(value)
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be non-empty clean text")
    return value


def load_development_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        raise ValueError("GeoRoute development manifest must contain a splits mapping")
    fit = [str(item) for item in payload["splits"].get("fit", [])]
    gate = [str(item) for item in payload["splits"].get("gate", [])]
    if not fit or not gate or set(fit) & set(gate):
        raise ValueError("GeoRoute development fit/gate split is invalid")
    payload["splits"] = {"fit": sorted(set(fit)), "gate": sorted(set(gate))}
    payload["manifest_file_sha256"] = sha256_file(path)
    return payload


def assert_development_annotation(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, dict) or not database:
        raise ValueError("GeoRoute development annotation has no database")
    subsets = {str(record.get("subset", "")) for record in database.values() if isinstance(record, dict)}
    if subsets != {"training"}:
        raise ValueError("GeoRoute development annotation must contain training records only")
    return {
        "path": str(annotation_path.resolve()),
        "sha256": sha256_file(annotation_path),
        "video_ids": sorted(map(str, database)),
        "official_test_records_loaded": 0,
    }


def variant_spec(name: str, *, token_budget: int | None = None) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"unsupported GeoRoute variant {name!r}")
    spec = copy.deepcopy(VARIANTS[name])
    if token_budget is not None:
        if name == "dense_native" or not (1 <= int(token_budget) <= 220):
            raise ValueError("only matched-K variants may override a budget in [1,220]")
        spec["tokens_per_tubelet"] = int(token_budget)
    return spec


def bind_development_config(
    *,
    source_config_path: str | Path,
    variant: str,
    stage: str,
    seed: int,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    token_budget: int | None = None,
):
    """Create one immutable development-only bound config."""

    from mmengine.config import Config

    if stage not in {"p1", "p2", "p3"}:
        raise ValueError("GeoRoute stage must be p1, p2, or p3")
    if int(seed) not in DEVELOPMENT_SEEDS:
        raise ValueError("GeoRoute seed is outside the frozen seed set")
    source_config_path = Path(source_config_path).resolve()
    manifest = load_development_manifest(manifest_path)
    annotation = assert_development_annotation(development_annotation_path)
    required_ids = set(manifest["splits"]["fit"]) | set(manifest["splits"]["gate"])
    if not required_ids <= set(annotation["video_ids"]):
        raise ValueError("GeoRoute manifest names videos absent from development annotation")
    class_map_path = Path(class_map_path).resolve()
    video_root = Path(development_video_root).resolve()
    pretrained_path = Path(pretrained_checkpoint_path).resolve()
    for path in (class_map_path, pretrained_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not video_root.is_dir() or "test" in video_root.as_posix().lower():
        raise ValueError("GeoRoute development video root must exist and must not be a test root")
    spec = variant_spec(variant, token_budget=token_budget)
    cfg = Config.fromfile(str(source_config_path))
    for split_name, block_list in (
        ("train", manifest["splits"]["gate"]),
        ("val", manifest["splits"]["fit"]),
        ("test", manifest["splits"]["fit"]),
    ):
        split = cfg.dataset[split_name]
        split.ann_file = annotation["path"]
        split.class_map = str(class_map_path)
        split.data_path = str(video_root)
        split.subset_name = "training"
        split.block_list = list(block_list)
    cfg.dataset.test.test_mode = True
    cfg.evaluation.ground_truth_filename = annotation["path"]
    cfg.evaluation.subset = "training"
    cfg.model.backbone.custom.pretrain = str(pretrained_path)
    custom = cfg.model.backbone.custom
    custom.georoute_route_mode = spec["route_mode"]
    custom.georoute_policy_estimator = spec["policy_estimator"]
    custom.georoute_context_tokens = int(spec["context_tokens"])
    custom.georoute_roi_fraction = float(spec["roi_fraction"])
    custom.georoute_random_seed = int(seed)
    custom.georoute_pooling_mode = "uniform_selected"
    custom.georoute_adapter_mode = "coordinate_lineage_packed"
    if spec["tokens_per_tubelet"] is not None:
        custom.georoute_tokens_per_tubelet = int(spec["tokens_per_tubelet"])
    if "geometry_stride_tubelets" in spec:
        custom.georoute_geometry_stride_tubelets = int(spec["geometry_stride_tubelets"])
    if "absolute_position_enabled" in spec:
        custom.georoute_absolute_position_enabled = bool(spec["absolute_position_enabled"])
    if "absolute_coordinates_enabled" in spec:
        custom.georoute_absolute_coordinates_enabled = bool(spec["absolute_coordinates_enabled"])
    if "geometry_side_channel" in spec:
        custom.georoute_geometry_side_channel = bool(spec["geometry_side_channel"])
    epochs = stage_epochs(stage)
    cfg.scheduler.max_epoch = epochs
    cfg.workflow.end_epoch = epochs
    cfg.workflow.val_start_epoch = epochs
    cfg.workflow.checkpoint_policy = "final_only"
    cfg.work_dir = str(Path(work_dir).resolve())
    binding = {
        "schema_version": GEOROUTE_EXPERIMENT_SCHEMA,
        "stage": stage,
        "variant": variant,
        "variant_spec": spec,
        "seed": int(seed),
        "source_config": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "fit_video_ids": manifest["splits"]["fit"],
        "gate_video_ids": manifest["splits"]["gate"],
        "development_annotation": annotation,
        "class_map_sha256": sha256_file(class_map_path),
        "development_video_root": str(video_root),
        "pretrained_checkpoint_sha256": sha256_file(pretrained_path),
        "work_dir": cfg.work_dir,
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "valid_native_support": "floor_complete_patches_with_explicit_mask",
        "pooling_mode": "uniform_selected",
        "adapter_mode": "coordinate_lineage_packed",
        "checkpoint_policy": "final_only_atomic",
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_runtime_binding = binding
    return cfg


def high_iou_score(metrics: Mapping[str, Any]) -> float:
    try:
        return 0.5 * (float(metrics["mAP@0.6"]) + float(metrics["mAP@0.7"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("GeoRoute result lacks mAP@0.6/mAP@0.7") from exc


def select_p1_roi_candidate(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared NativeTokenSelect-first promotion rule.

    Every registered control, including random, participates in the gate.
    Geometry can advance only after it strictly beats the ROI-free selector and
    geometry-only side-channel control under the matched-K cost ledger.
    """

    required = set(P1_VARIANTS)
    if set(records) != required:
        raise ValueError(f"P1 record set must equal {sorted(required)}")
    for name, record in records.items():
        if record.get("stage") != "p1" or record.get("variant") != name:
            raise ValueError(f"invalid P1 record for {name}")
        if record.get("official_test_opened") is not False:
            raise ValueError("P1 record opened official test")
        profile = record.get("profile", {})
        if not isinstance(profile.get("development_window_wall_p50_ms"), (int, float)):
            raise ValueError(f"P1 record has no measured development window cost: {name}")
        if profile.get("paper_grade_end_to_end_claim_allowed") is not False:
            raise ValueError("P1 profile must not be relabelled as paper-grade end-to-end cost")
        high_iou_score(record["metrics"])
    scores = {
        name: high_iou_score(record["metrics"])
        for name, record in records.items()
    }
    costs = {
        name: float(
            record["profile"]["development_window_wall_p50_ms"]
        )
        for name, record in records.items()
    }
    native_accuracy_gate = scores["free"] > max(
        scores["fixed_lattice"],
        scores["random"],
        scores["fixed_lattice_geometry"],
    )
    native_cost_gate = costs["free"] < costs["dense_native"]
    hybrid_accuracy_gate = native_accuracy_gate and scores["hybrid"] > max(
        scores["free"],
        scores["random"],
        scores["fixed_lattice_geometry"],
    )
    hybrid_cost_gate = native_cost_gate and costs["hybrid"] <= min(
        costs["free"],
        costs["random"],
        costs["fixed_lattice_geometry"],
    )
    best_learned = max(
        scores["free"],
        scores["roi"],
        scores["hybrid"],
    )
    best_nonlearned = max(
        scores["fixed_lattice"],
        scores["random"],
    )
    if not (native_accuracy_gate and native_cost_gate):
        if best_learned <= best_nonlearned:
            status = "STOP_LEARNED_ROUTING"
        else:
            status = "STOP_AMBIGUOUS_NO_PREDECLARED_WINNER"
        promoted_variant = None
        promoted_route = "C"
    elif hybrid_accuracy_gate and hybrid_cost_gate:
        status = "ADVANCE_HYBRID_TO_P2"
        promoted_variant = "hybrid"
        promoted_route = "A"
    else:
        status = "ADVANCE_FREE_TO_P2"
        promoted_variant = "free"
        promoted_route = "B"
    decision = {
        "schema_version": GEOROUTE_EXPERIMENT_SCHEMA,
        "stage": "p1_selection",
        "selection_metric": "mean(mAP@0.6,mAP@0.7)",
        "scores": scores,
        "development_window_wall_p50_ms": costs,
        "native_accuracy_gate": native_accuracy_gate,
        "native_cost_gate": native_cost_gate,
        "hybrid_accuracy_gate": hybrid_accuracy_gate,
        "hybrid_cost_gate": hybrid_cost_gate,
        "selected_variant": promoted_variant,
        "selected_route": promoted_route,
        "status": status,
        "paper_claim_allowed": False,
        "official_test_opened": False,
    }
    decision["selection_sha256"] = canonical_sha256(decision)
    return decision


def select_p2_roi_candidate(
    records_by_variant: Mapping[str, list[Mapping[str, Any]]],
    *,
    candidate_variant: str,
) -> dict[str, Any]:
    """Apply the frozen three-seed promotion rule after P2.

    P2 is a development-only decision.  It cannot open the official test or
    establish a paper claim.  A structured route has to improve the paired
    high-IoU mean over the free-token control and be no slower under the same
    development-window measurement before P3 is authorized.
    """

    if candidate_variant == "free":
        reference_variants = (
            "fixed_lattice",
            "random",
            "fixed_lattice_geometry",
        )
    elif candidate_variant in {"roi", "hybrid"}:
        reference_variants = (
            "free",
            "fixed_lattice",
            "random",
            "fixed_lattice_geometry",
        )
    else:
        raise ValueError("P2 candidate must be free, roi, or hybrid")
    expected = {*reference_variants, str(candidate_variant)}
    if set(records_by_variant) != expected:
        raise ValueError(f"P2 record variants must equal {sorted(expected)}")

    normalized: dict[str, dict[int, Mapping[str, Any]]] = {}
    for variant, records in records_by_variant.items():
        by_seed: dict[int, Mapping[str, Any]] = {}
        for record in records:
            if record.get("stage") != "p2" or record.get("variant") != variant:
                raise ValueError(f"invalid P2 record for {variant}")
            if record.get("official_test_opened") is not False:
                raise ValueError("P2 record opened official test")
            seed = record.get("seed")
            if seed not in DEVELOPMENT_SEEDS or seed in by_seed:
                raise ValueError(f"P2 seed set is invalid for {variant}")
            high_iou_score(record["metrics"])
            profile = record.get("profile", {})
            if not isinstance(profile.get("development_window_wall_p50_ms"), (int, float)):
                raise ValueError(f"P2 record lacks development-window cost for {variant}")
            if profile.get("paper_grade_end_to_end_claim_allowed") is not False:
                raise ValueError("P2 profile is incorrectly labelled paper-grade")
            by_seed[int(seed)] = record
        if set(by_seed) != set(DEVELOPMENT_SEEDS):
            raise ValueError(f"P2 record set for {variant} does not contain all frozen seeds")
        normalized[variant] = by_seed

    candidate_values = [high_iou_score(normalized[candidate_variant][seed]["metrics"]) for seed in DEVELOPMENT_SEEDS]
    candidate_costs = [
        float(normalized[candidate_variant][seed]["profile"]["development_window_wall_p50_ms"])
        for seed in DEVELOPMENT_SEEDS
    ]
    mean_candidate_cost = sum(candidate_costs) / len(candidate_costs)
    paired_deltas_by_reference = {}
    reference_mean_costs = {}
    for reference in reference_variants:
        values = [
            high_iou_score(normalized[reference][seed]["metrics"])
            for seed in DEVELOPMENT_SEEDS
        ]
        costs = [
            float(
                normalized[reference][seed]["profile"][
                    "development_window_wall_p50_ms"
                ]
            )
            for seed in DEVELOPMENT_SEEDS
        ]
        paired_deltas_by_reference[reference] = [
            candidate - baseline
            for candidate, baseline in zip(candidate_values, values)
        ]
        reference_mean_costs[reference] = sum(costs) / len(costs)
    paired_mean_deltas = {
        reference: sum(values) / len(values)
        for reference, values in paired_deltas_by_reference.items()
    }
    advance = all(value > 0.0 for value in paired_mean_deltas.values()) and all(
        mean_candidate_cost <= cost
        for cost in reference_mean_costs.values()
    )
    if advance and candidate_variant == "free":
        status = "ADVANCE_NATIVE_ROUTE_B_TO_P3"
    elif advance:
        status = "ADVANCE_GEOMETRY_ROUTE_A_TO_P3"
    else:
        status = "STOP_LEARNED_ROUTING"
    decision = {
        "schema_version": GEOROUTE_EXPERIMENT_SCHEMA,
        "stage": "p2_selection",
        "candidate_variant": candidate_variant,
        "selection_metric": "paired mean(mAP@0.6,mAP@0.7)",
        "reference_variants": list(reference_variants),
        "paired_high_iou_deltas_by_reference": paired_deltas_by_reference,
        "paired_mean_high_iou_deltas_by_reference": paired_mean_deltas,
        "candidate_mean_development_window_wall_p50_ms": mean_candidate_cost,
        "reference_mean_development_window_wall_p50_ms": reference_mean_costs,
        "development_window_cost_not_worse_than_all_references": all(
            mean_candidate_cost <= cost for cost in reference_mean_costs.values()
        ),
        "status": status,
        "paper_claim_allowed": False,
        "official_test_opened": False,
    }
    decision["selection_sha256"] = canonical_sha256(decision)
    return decision
