"""Frozen paired full-stack cost contract for SCNR residual centering.

The study consumes the two immutable seed-3407 matched-training checkpoints. It
never trains or resumes either arm.  One Slurm allocation profiles eight serial
passes on one GPU in the pre-registered ABBA+BAAB order, with one continuous
NVML sidecar.  Passing this development cost non-inferiority gate may authorize
additional seeds; it never opens official test or a paper claim by itself.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_SEED,
    DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
    _finite,
    _integrate_power_samples,
    _load_json_object,
    _load_jsonl_objects,
    _quantile,
    _self_hash_matches,
    _validated_file_receipt,
    validate_dynamic_floor_m2_checkpoint_sidecar,
    validate_dynamic_floor_m2_config,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_residual_centering_training_contract import (
    RESIDUAL_CENTERING_BASE_ARM,
    RESIDUAL_CENTERING_TRAINING_FINALIZATION_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_STUDY_ID,
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
    finalize_residual_centering_training,
    residual_centering_training_cell_relative_path,
    residual_centering_training_variant_spec,
    validate_residual_centering_training_config,
    validate_residual_centering_training_stage_result,
)
from tools.bata.spatial_zoom_s1_cost import make_profile_exposure_id


RESIDUAL_CENTERING_COST_STUDY_ID = "scnr_residual_centering_paired_cost_v1"
RESIDUAL_CENTERING_COST_BINDING_SCHEMA = (
    "scnr_residual_centering_paired_cost_binding_v1"
)
RESIDUAL_CENTERING_COST_SAMPLE_SCHEMA = (
    "scnr_residual_centering_paired_cost_sample_v1"
)
RESIDUAL_CENTERING_COST_PROFILE_SCHEMA = (
    "scnr_residual_centering_paired_cost_profile_v1"
)
RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA = (
    "scnr_residual_centering_paired_cost_deployment_v1"
)
RESIDUAL_CENTERING_COST_FINALIZATION_SCHEMA = (
    "scnr_residual_centering_paired_cost_finalization_v1"
)
RESIDUAL_CENTERING_COST_ORDER = (
    "none_control",
    "residual_window_center",
    "residual_window_center",
    "none_control",
    "residual_window_center",
    "none_control",
    "none_control",
    "residual_window_center",
)
# Every tuple is (none_control pass, residual_window_center pass).
RESIDUAL_CENTERING_COST_PAIRS = ((0, 1), (3, 2), (5, 4), (6, 7))
RESIDUAL_CENTERING_COST_WARMUP_SAMPLES = 50
RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS = 20
RESIDUAL_CENTERING_COST_BOOTSTRAP_ITERATIONS = 10_000
RESIDUAL_CENTERING_COST_BOOTSTRAP_SEED = 20_260_806
RESIDUAL_CENTERING_COST_NONINFERIORITY_RATIO = 1.05
RESIDUAL_CENTERING_COST_PRIMARY_METRICS = {
    "end_to_end_p50": ("end_to_end_serial_ms", "p50"),
    "energy_per_sample": ("gpu_energy_j", "mean"),
}
RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS = (
    "opentad",
    "configs",
    "tools/train.py",
    "tools/test.py",
    "tools/bata/georoute_dynamic_floor_m2_contract.py",
    "tools/bata/georoute_residual_centering_training_contract.py",
    "tools/bata/run_georoute_residual_centering_training.py",
    "tools/bata/profile_georoute_dynamic_floor_m2.py",
    "tools/bata/profile_spatial_zoom_s1.py",
    "tools/bata/spatial_zoom_s1_power.py",
)


def _is_sensitive_runtime_path(path: str) -> bool:
    normalized = str(path).replace("\\", "/").strip("/")
    return any(
        normalized == protected or normalized.startswith(f"{protected}/")
        for protected in RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS
    )


def validate_frozen_residual_centering_cost_contract() -> None:
    if RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER != (
        "none_control",
        "residual_window_center",
    ):
        raise RuntimeError("residual-centering training variant order changed")
    if RESIDUAL_CENTERING_COST_ORDER != (
        "none_control",
        "residual_window_center",
        "residual_window_center",
        "none_control",
        "residual_window_center",
        "none_control",
        "none_control",
        "residual_window_center",
    ):
        raise RuntimeError("residual-centering ABBA+BAAB order changed")
    if RESIDUAL_CENTERING_COST_PAIRS != ((0, 1), (3, 2), (5, 4), (6, 7)):
        raise RuntimeError("residual-centering paired-pass mapping changed")
    if any(
        RESIDUAL_CENTERING_COST_ORDER[none_index] != "none_control"
        or RESIDUAL_CENTERING_COST_ORDER[center_index]
        != "residual_window_center"
        for none_index, center_index in RESIDUAL_CENTERING_COST_PAIRS
    ):
        raise RuntimeError("residual-centering cost pairs lost arm identity")
    if (
        RESIDUAL_CENTERING_COST_WARMUP_SAMPLES != 50
        or RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS != 20
        or RESIDUAL_CENTERING_COST_BOOTSTRAP_ITERATIONS != 10_000
        or RESIDUAL_CENTERING_COST_NONINFERIORITY_RATIO != 1.05
    ):
        raise RuntimeError("residual-centering cost measurement policy changed")


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be one JSON object")
    return payload


def residual_centering_cost_stage_paths(training_run_root: str | Path) -> dict[str, Path]:
    root = Path(training_run_root).resolve()
    return {
        variant: (
            root
            / residual_centering_training_cell_relative_path(variant=variant)
            / "stage_result.json"
        ).resolve()
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    }


def validate_residual_centering_cost_source(
    training_run_root: str | Path,
    *,
    expected_model_runtime_commit: str,
) -> dict[str, Any]:
    """Recompute the immutable accuracy authorization consumed by cost replay."""

    root = Path(training_run_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("residual-centering training run root is invalid")
    stage_paths = residual_centering_cost_stage_paths(root)
    stages = {
        variant: validate_residual_centering_training_stage_result(
            _read_object(path, label=f"{variant} stage result"),
            expected_variant=variant,
            expected_commit=expected_model_runtime_commit,
        )
        for variant, path in stage_paths.items()
    }
    finalization_path = (root / "finalization" / "finalization.json").resolve()
    finalization = _read_object(
        finalization_path, label="residual-centering training finalization"
    )
    expected_jobs = {
        variant: str(stages[variant]["slurm_job_id"])
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    }
    recomputed = finalize_residual_centering_training(
        stages,
        expected_commit=expected_model_runtime_commit,
        expected_job_ids=expected_jobs,
    )
    critical = (
        "study_id",
        "status",
        "decision",
        "runtime_commit",
        "expected_stage_job_ids",
        "accuracy_screen_conditions",
        "center_minus_none_metrics_pp",
        "paired_cost_authorized",
        "paired_cost_protocol",
        "seeds_3408_3409_opened",
        "official_test_opened",
        "paper_claim_allowed",
    )
    stage_receipts = finalization.get("stage_result_receipts")
    if (
        finalization.get("schema_version")
        != RESIDUAL_CENTERING_TRAINING_FINALIZATION_SCHEMA
        or not _self_hash_matches(finalization, field="finalization_sha256")
        or any(finalization.get(key) != recomputed.get(key) for key in critical)
        or finalization.get("status")
        != "PASS_ACCURACY_SCREEN_PAIRED_COST_AUTHORIZED"
        or finalization.get("paired_cost_authorized") is not True
        or finalization.get("seeds_3408_3409_opened") is not False
        or finalization.get("official_test_opened") is not False
        or finalization.get("paper_claim_allowed") is not False
        or not isinstance(stage_receipts, Mapping)
        or set(stage_receipts) != set(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER)
    ):
        raise ValueError("residual-centering training did not authorize paired cost")
    for variant, path in stage_paths.items():
        receipt = stage_receipts[variant]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("present") is not True
            or receipt.get("path") != str(path)
            or receipt.get("sha256") != sha256_file(path)
        ):
            raise ValueError("residual-centering finalization stage receipt changed")
    shared_hashes = {
        stage["binding"]["shared_protocol_sha256"] for stage in stages.values()
    }
    populations = {
        stage["accuracy_replays"]["accuracy_a"]["population_sha256"]
        for stage in stages.values()
    }
    if len(shared_hashes) != 1 or len(populations) != 1:
        raise ValueError("residual-centering source arms are not matched")
    return {
        "training_run_root": str(root),
        "stages": stages,
        "stage_paths": stage_paths,
        "stage_result_receipts": {
            variant: {
                "path": str(path),
                "sha256": sha256_file(path),
                "stage_result_sha256": stages[variant]["stage_result_sha256"],
            }
            for variant, path in stage_paths.items()
        },
        "training_finalization": finalization,
        "training_finalization_receipt": {
            "path": str(finalization_path),
            "sha256": sha256_file(finalization_path),
            "finalization_sha256": finalization["finalization_sha256"],
        },
        "shared_protocol_sha256": shared_hashes.pop(),
        "accuracy_population_sha256": populations.pop(),
    }


def validate_residual_centering_cost_deployment(
    deployment: Mapping[str, Any],
    *,
    run_root: str | Path,
    training_run_root: str | Path,
    expected_model_runtime_commit: str,
    expected_execution_commit: str,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable one-job deployment receipt before profiling."""

    payload = dict(deployment)
    resolved_run_root = Path(run_root).resolve()
    resolved_training_root = Path(training_run_root).resolve()
    source = validate_residual_centering_cost_source(
        resolved_training_root,
        expected_model_runtime_commit=expected_model_runtime_commit,
    )
    jobs = payload.get("jobs")
    job_id = str(jobs.get("paired_cost", "")) if isinstance(jobs, Mapping) else ""
    execution_delta = payload.get("execution_delta")
    changed_files = (
        execution_delta.get("changed_files")
        if isinstance(execution_delta, Mapping)
        else None
    )
    if (
        payload.get("schema_version") != RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA
        or payload.get("status")
        != "DEPLOYED_RESIDUAL_CENTERING_SINGLE_JOB_PAIRED_COST"
        or payload.get("study_id") != RESIDUAL_CENTERING_COST_STUDY_ID
        or payload.get("model_runtime_commit") != expected_model_runtime_commit
        or payload.get("execution_commit") != expected_execution_commit
        or payload.get("run_root") != str(resolved_run_root)
        or payload.get("training_run_root") != str(resolved_training_root)
        or tuple(payload.get("cost_order", ())) != RESIDUAL_CENTERING_COST_ORDER
        or tuple(
            tuple(pair) for pair in payload.get("paired_pass_indices", ())
        )
        != RESIDUAL_CENTERING_COST_PAIRS
        or not isinstance(jobs, Mapping)
        or set(jobs) != {"paired_cost"}
        or not job_id.isdigit()
        or (expected_job_id is not None and job_id != str(expected_job_id))
        or payload.get("single_slurm_job") is not True
        or payload.get("single_visible_gpu") is not True
        or payload.get("continuous_power_sidecar") is not True
        or payload.get("training_or_resume_allowed") is not False
        or payload.get("source_model_or_config_changed") is not False
        or not isinstance(execution_delta, Mapping)
        or execution_delta.get("model_runtime_commit")
        != expected_model_runtime_commit
        or execution_delta.get("execution_commit") != expected_execution_commit
        or execution_delta.get("model_runtime_is_ancestor") is not True
        or execution_delta.get("source_model_or_config_changed") is not False
        or tuple(execution_delta.get("sensitive_runtime_paths", ()))
        != RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS
        or not isinstance(changed_files, list)
        or any(not isinstance(path, str) or not path for path in changed_files)
        or any(_is_sensitive_runtime_path(path) for path in changed_files)
        or execution_delta.get("changed_files_sha256")
        != canonical_sha256(changed_files)
        or payload.get("stage_result_receipts") != source["stage_result_receipts"]
        or payload.get("training_finalization_receipt")
        != source["training_finalization_receipt"]
        or payload.get("all_jobs_held_until_immutable_receipt") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
        or not _self_hash_matches(payload, field="deployment_sha256")
    ):
        raise ValueError("residual-centering cost deployment receipt is invalid")
    return payload


def build_residual_centering_cost_config(
    stage: Mapping[str, Any], *, variant: str
) -> Any:
    """Build and self-bind the exact cost configuration for one trained arm."""

    from mmengine.config import Config

    validated_stage = validate_residual_centering_training_stage_result(
        stage, expected_variant=variant
    )
    accuracy_path = Path(
        validated_stage["config_receipts"]["accuracy_a"]["path"]
    )
    cfg = Config.fromfile(str(accuracy_path))
    training_binding = validate_residual_centering_training_config(
        cfg, variant=variant, phase="accuracy"
    )
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled = False
    cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    cfg.georoute_development_profile = dict(enabled=False)
    cfg.solver.test.batch_size = 1
    cfg.solver.test.num_workers = 0
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.post_processing.save_dict = False
    cfg.post_processing.sliding_window = True
    cost_binding: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_COST_BINDING_SCHEMA,
        "study_id": RESIDUAL_CENTERING_COST_STUDY_ID,
        "source_training_study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "variant": variant,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "model_runtime_commit": validated_stage["runtime_commit"],
        "source_stage_result_sha256": validated_stage["stage_result_sha256"],
        "source_training_binding_sha256": training_binding["binding_sha256"],
        "shared_protocol_sha256": training_binding["shared_protocol_sha256"],
        "branch_calibration_mode": residual_centering_training_variant_spec(variant)[
            "branch_calibration_mode"
        ],
        "profile_order": list(RESIDUAL_CENTERING_COST_ORDER),
        "paired_pass_indices": [list(pair) for pair in RESIDUAL_CENTERING_COST_PAIRS],
        "warmup_samples_per_pass": RESIDUAL_CENTERING_COST_WARMUP_SAMPLES,
        "power_interval_ms": RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS,
        "batch_size": 1,
        "loader_workers": 0,
        "world_size": 1,
        "diagnostic_telemetry_inside_timed_forward": False,
        "training_or_resume_allowed": False,
        "checkpoint_state_key": "state_dict_ema",
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    cost_binding["binding_sha256"] = canonical_sha256(cost_binding)
    cfg.georoute_residual_centering_cost_binding = cost_binding
    cfg.georoute_phase_m_binding = copy.deepcopy(cost_binding)
    validate_residual_centering_cost_config(
        cfg, stage=validated_stage, variant=variant
    )
    return cfg


def validate_residual_centering_cost_config(
    cfg: Any, *, stage: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    base = validate_dynamic_floor_m2_config(
        cfg, arm=RESIDUAL_CENTERING_BASE_ARM, phase="cost"
    )
    spec = residual_centering_training_variant_spec(variant)
    binding = cfg.get("georoute_residual_centering_cost_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("residual-centering cost binding is missing")
    binding = dict(binding)
    if (
        not _self_hash_matches(binding, field="binding_sha256")
        or binding.get("schema_version") != RESIDUAL_CENTERING_COST_BINDING_SCHEMA
        or binding.get("study_id") != RESIDUAL_CENTERING_COST_STUDY_ID
        or binding.get("variant") != variant
        or int(binding.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or binding.get("model_runtime_commit") != stage.get("runtime_commit")
        or binding.get("source_stage_result_sha256")
        != stage.get("stage_result_sha256")
        or binding.get("source_training_binding_sha256")
        != stage.get("binding_sha256")
        or binding.get("shared_protocol_sha256")
        != stage.get("binding", {}).get("shared_protocol_sha256")
        or binding.get("branch_calibration_mode")
        != spec["branch_calibration_mode"]
        or tuple(binding.get("profile_order", ())) != RESIDUAL_CENTERING_COST_ORDER
        or tuple(
            tuple(pair) for pair in binding.get("paired_pass_indices", ())
        )
        != RESIDUAL_CENTERING_COST_PAIRS
        or binding.get("training_or_resume_allowed") is not False
        or binding.get("official_test_opened") is not False
        or binding.get("paper_claim_allowed") is not False
        or cfg.model.backbone.custom.georoute_branch_calibration_mode
        != spec["branch_calibration_mode"]
        or cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled
        is not False
        or cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled
        is not False
        or dict(cfg.georoute_phase_m_binding) != binding
        or int(base["arm_spec"]["roi_extent_floor_cells"]) != 1
    ):
        raise ValueError("residual-centering cost config binding is invalid")
    return binding


def _summary(values: Sequence[float]) -> dict[str, float]:
    checked = [float(value) for value in values]
    if not checked or any(not math.isfinite(value) or value <= 0.0 for value in checked):
        raise ValueError("paired cost values must be finite and positive")
    return {
        "mean": sum(checked) / len(checked),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _geometric_mean(values: Sequence[float]) -> float:
    checked = [float(value) for value in values]
    if not checked or any(value <= 0.0 or not math.isfinite(value) for value in checked):
        raise ValueError("paired cost ratios must be finite and positive")
    return math.exp(sum(math.log(value) for value in checked) / len(checked))


def _metric_value(rows: Sequence[Mapping[str, Any]], *, key: str, reducer: str) -> float:
    values = [float(row[key]) for row in rows]
    if reducer == "p50":
        return _quantile(values, 0.50)
    if reducer == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unsupported paired cost reducer {reducer!r}")


def analyze_residual_centering_paired_cost(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = RESIDUAL_CENTERING_COST_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Order-balanced paired estimate with a deterministic clustered bootstrap."""

    if int(bootstrap_iterations) <= 0:
        raise ValueError("paired cost bootstrap iterations must be positive")
    rows_by_pass: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(RESIDUAL_CENTERING_COST_ORDER))
    }
    for raw in raw_rows:
        row = dict(raw)
        pass_index = int(row.get("pass_index", -1))
        if pass_index not in rows_by_pass:
            raise ValueError("paired cost row has an invalid pass index")
        rows_by_pass[pass_index].append(row)
    manifests = []
    for pass_index, rows in rows_by_pass.items():
        rows.sort(key=lambda row: int(row["sample_ordinal"]))
        if not rows or any(
            row.get("arm") != RESIDUAL_CENTERING_COST_ORDER[pass_index]
            for row in rows
        ):
            raise ValueError("paired cost pass is incomplete")
        manifest = []
        for row in rows:
            ordinal = int(row["sample_ordinal"])
            physical_window_id = str(row.get("physical_window_id", "")).strip()
            window_id = str(row.get("window_id", "")).strip()
            if (
                ordinal < 0
                or int(row.get("loader_ordinal", -1)) != ordinal
                or not physical_window_id
                or window_id
                != make_profile_exposure_id(physical_window_id, ordinal)
            ):
                raise ValueError("paired cost exposure identity is invalid")
            manifest.append((window_id, physical_window_id))
        if len({window_id for window_id, _physical in manifest}) != len(manifest):
            raise ValueError("paired cost exposure identities are not unique")
        manifests.append(manifest)
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("paired cost passes changed physical sample order")
    videos = sorted({str(row["video_id"]) for row in rows_by_pass[0]})
    if not videos:
        raise ValueError("paired cost bootstrap has no video clusters")
    by_pass_video: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for pass_index, rows in rows_by_pass.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["video_id"])].append(row)
        if set(grouped) != set(videos):
            raise ValueError("paired cost passes changed video clusters")
        by_pass_video[pass_index] = grouped

    metric_results: dict[str, Any] = {}
    rng = random.Random(RESIDUAL_CENTERING_COST_BOOTSTRAP_SEED)
    sampled_videos = [
        [rng.choice(videos) for _ in videos] for _ in range(int(bootstrap_iterations))
    ]
    sampled_pairs = [
        [rng.choice(RESIDUAL_CENTERING_COST_PAIRS) for _ in RESIDUAL_CENTERING_COST_PAIRS]
        for _ in range(int(bootstrap_iterations))
    ]
    for metric_name, (key, reducer) in RESIDUAL_CENTERING_COST_PRIMARY_METRICS.items():
        pair_rows = []
        pair_ratios = []
        for none_index, center_index in RESIDUAL_CENTERING_COST_PAIRS:
            none_value = _metric_value(rows_by_pass[none_index], key=key, reducer=reducer)
            center_value = _metric_value(
                rows_by_pass[center_index], key=key, reducer=reducer
            )
            ratio = center_value / none_value
            pair_rows.append(
                {
                    "none_pass_index": none_index,
                    "center_pass_index": center_index,
                    "none_value": none_value,
                    "center_value": center_value,
                    "center_over_none_ratio": ratio,
                }
            )
            pair_ratios.append(ratio)
        estimate = _geometric_mean(pair_ratios)
        bootstrap = []
        for video_draw, pair_draw in zip(sampled_videos, sampled_pairs):
            ratios = []
            for none_index, center_index in pair_draw:
                none_rows = [
                    row
                    for video in video_draw
                    for row in by_pass_video[none_index][video]
                ]
                center_rows = [
                    row
                    for video in video_draw
                    for row in by_pass_video[center_index][video]
                ]
                ratios.append(
                    _metric_value(center_rows, key=key, reducer=reducer)
                    / _metric_value(none_rows, key=key, reducer=reducer)
                )
            bootstrap.append(_geometric_mean(ratios))
        metric_results[metric_name] = {
            "source_field": key,
            "reducer": reducer,
            "center_over_none_ratio": estimate,
            "delta_percent": (estimate - 1.0) * 100.0,
            "cluster_pair_bootstrap_95ci": [
                _quantile(bootstrap, 0.025),
                _quantile(bootstrap, 0.975),
            ],
            "paired_pass_estimates": pair_rows,
        }
    conditions = {
        f"{name}_upper_ci_le_1.05": float(result["cluster_pair_bootstrap_95ci"][1])
        <= RESIDUAL_CENTERING_COST_NONINFERIORITY_RATIO
        for name, result in metric_results.items()
    }
    strict_pareto = all(
        float(result["center_over_none_ratio"]) <= 1.0
        for result in metric_results.values()
    ) and any(
        float(result["cluster_pair_bootstrap_95ci"][1]) <= 1.0
        for result in metric_results.values()
    )
    return {
        "schema_version": "scnr_residual_centering_paired_cost_analysis_v1",
        "profile_order": list(RESIDUAL_CENTERING_COST_ORDER),
        "paired_pass_indices": [list(pair) for pair in RESIDUAL_CENTERING_COST_PAIRS],
        "video_cluster_count": len(videos),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": RESIDUAL_CENTERING_COST_BOOTSTRAP_SEED,
        "bootstrap_unit": "video_cluster_and_counterbalanced_pass_pair",
        "noninferiority_ratio": RESIDUAL_CENTERING_COST_NONINFERIORITY_RATIO,
        "primary_metrics": metric_results,
        "conditions": conditions,
        "cost_noninferior": all(conditions.values()),
        "strict_pareto_observed": strict_pareto,
        "independent_job_variance_estimated": False,
    }


def validate_residual_centering_cost_profile(
    profile: Mapping[str, Any],
    *,
    expected_model_runtime_commit: str | None = None,
    expected_execution_commit: str | None = None,
) -> dict[str, Any]:
    """Recompute stage, raw timing, energy, and summary receipts fail closed."""

    profile = dict(profile)
    if (
        expected_model_runtime_commit is not None
        and profile.get("model_runtime_commit") != expected_model_runtime_commit
    ):
        raise ValueError("residual-centering cost model runtime commit mismatch")
    if (
        expected_execution_commit is not None
        and profile.get("execution_commit") != expected_execution_commit
    ):
        raise ValueError("residual-centering cost execution commit mismatch")
    run_root = Path(str(profile.get("run_root", ""))).resolve()
    training_root = Path(str(profile.get("training_run_root", ""))).resolve()
    expected_scope = {
        "decode": True,
        "preprocess": True,
        "host_to_device": True,
        "scout": True,
        "route": True,
        "patch_embed": True,
        "backbone": True,
        "adapter": True,
        "detector": True,
        "nms": True,
        "diagnostic_telemetry_inside_timed_forward": False,
        "same_gpu_counterbalanced": True,
        "continuous_power_sidecar": True,
        "development_only": True,
    }
    slurm = profile.get("slurm")
    hardware_identity = profile.get("hardware_identity")
    software_identity = profile.get("software_identity")
    if (
        profile.get("schema_version") != RESIDUAL_CENTERING_COST_PROFILE_SCHEMA
        or profile.get("status") != "PASS_RESIDUAL_CENTERING_PAIRED_FULL_STACK_COST"
        or profile.get("study_id") != RESIDUAL_CENTERING_COST_STUDY_ID
        or int(profile.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or tuple(profile.get("profile_order", ())) != RESIDUAL_CENTERING_COST_ORDER
        or int(profile.get("warmup_samples_per_pass", -1))
        != RESIDUAL_CENTERING_COST_WARMUP_SAMPLES
        or int(profile.get("power_interval_ms", -1))
        != RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS
        or int(profile.get("batch_size", -1)) != 1
        or int(profile.get("loader_workers", -1)) != 0
        or int(profile.get("world_size", -1)) != 1
        or profile.get("scope") != expected_scope
        or not isinstance(profile.get("population_sha256"), str)
        or len(profile["population_sha256"]) != 64
        or not isinstance(profile.get("accuracy_population_sha256"), str)
        or len(profile["accuracy_population_sha256"]) != 64
        or int(profile.get("raw_sample_count", -1)) <= 0
        or not isinstance(profile.get("hardware_fingerprint"), str)
        or len(profile["hardware_fingerprint"]) != 64
        or not isinstance(profile.get("software_fingerprint"), str)
        or len(profile["software_fingerprint"]) != 64
        or not isinstance(hardware_identity, Mapping)
        or not isinstance(software_identity, Mapping)
        or profile.get("hardware_fingerprint")
        != canonical_sha256(hardware_identity)
        or profile.get("software_fingerprint")
        != canonical_sha256(software_identity)
        or not isinstance(slurm, Mapping)
        or not str(slurm.get("job_id", "")).isdigit()
        or slurm.get("logical_device") != "cuda:0"
        or profile.get("training_or_resume_executed") is not False
        or profile.get("official_test_opened") is not False
        or profile.get("paper_claim_allowed") is not False
        or str(run_root) != profile.get("run_root")
        or not run_root.is_dir()
        or str(training_root) != profile.get("training_run_root")
        or not training_root.is_dir()
        or not _self_hash_matches(profile, field="profile_sha256")
    ):
        raise ValueError("residual-centering cost profile header is invalid")
    deployment_path = _validated_file_receipt(
        profile.get("deployment_receipt"),
        label="residual-centering cost deployment",
    )
    expected_deployment_path = (run_root / "control" / "deployment.json").resolve()
    if deployment_path != expected_deployment_path:
        raise ValueError("residual-centering deployment receipt path changed")
    deployment = validate_residual_centering_cost_deployment(
        _load_json_object(
            deployment_path, label="residual-centering cost deployment"
        ),
        run_root=run_root,
        training_run_root=training_root,
        expected_model_runtime_commit=profile["model_runtime_commit"],
        expected_execution_commit=profile["execution_commit"],
        expected_job_id=str(slurm["job_id"]),
    )
    if profile["deployment_receipt"].get("deployment_sha256") != deployment.get(
        "deployment_sha256"
    ):
        raise ValueError("residual-centering deployment self-hash receipt changed")
    source = validate_residual_centering_cost_source(
        training_root,
        expected_model_runtime_commit=profile["model_runtime_commit"],
    )
    if (
        profile.get("stage_result_receipts") != source["stage_result_receipts"]
        or profile.get("training_finalization_receipt")
        != source["training_finalization_receipt"]
        or profile.get("accuracy_population_sha256")
        != source["accuracy_population_sha256"]
    ):
        raise ValueError("residual-centering cost source receipt changed")
    pass_receipts = profile.get("pass_receipts")
    arm_summaries = profile.get("arm_summaries")
    pass_summaries = profile.get("pass_summaries")
    artifacts = profile.get("artifact_receipts")
    if (
        not isinstance(pass_receipts, list)
        or len(pass_receipts) != len(RESIDUAL_CENTERING_COST_ORDER)
        or not isinstance(arm_summaries, Mapping)
        or set(arm_summaries) != set(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER)
        or not isinstance(pass_summaries, list)
        or len(pass_summaries) != len(RESIDUAL_CENTERING_COST_ORDER)
        or not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "raw_samples",
            "power_trace",
            "sidecar_attempt_report",
            "sidecar_attempt_trace",
        }
    ):
        raise ValueError("residual-centering cost profile receipts are incomplete")
    expected_cost_hashes = {
        variant: canonical_sha256(
            build_residual_centering_cost_config(
                source["stages"][variant], variant=variant
            ).to_dict()
        )
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    }
    pass_counts = []
    for pass_index, (receipt, variant) in enumerate(
        zip(pass_receipts, RESIDUAL_CENTERING_COST_ORDER)
    ):
        unsigned = dict(receipt)
        observed = unsigned.pop("pass_sha256", None)
        stage = source["stages"][variant]
        if (
            int(receipt.get("pass_index", -1)) != pass_index
            or receipt.get("variant") != variant
            or int(receipt.get("sample_count", -1)) <= 0
            or receipt.get("population_sha256") != profile.get("population_sha256")
            or receipt.get("accuracy_population_sha256")
            != profile.get("accuracy_population_sha256")
            or receipt.get("checkpoint_sha256")
            != stage["checkpoint_receipt"]["sha256"]
            or receipt.get("bound_accuracy_config_sha256")
            != stage["config_receipts"]["accuracy_a"]["sha256"]
            or receipt.get("cost_config_sha256") != expected_cost_hashes[variant]
            or receipt.get("branch_calibration_mode")
            != residual_centering_training_variant_spec(variant)[
                "branch_calibration_mode"
            ]
            or receipt.get("diagnostic_telemetry_inside_timed_forward") is not False
            or receipt.get("training_or_resume_executed") is not False
            or observed != canonical_sha256(unsigned)
        ):
            raise ValueError("residual-centering cost pass receipt is invalid")
        pass_counts.append(int(receipt["sample_count"]))
    if len(set(pass_counts)) != 1:
        raise ValueError("residual-centering cost passes changed population size")

    artifact_paths = {
        name: _validated_file_receipt(
            receipt, label=f"residual-centering paired cost {name}"
        )
        for name, receipt in artifacts.items()
    }
    for path in artifact_paths.values():
        try:
            path.relative_to(run_root / "cost")
        except ValueError as error:
            raise ValueError("residual-centering cost artifact left run root") from error
    raw_rows = _load_jsonl_objects(
        artifact_paths["raw_samples"], label="residual-centering raw cost samples"
    )
    if len(raw_rows) != int(profile.get("raw_sample_count", -1)) or len(raw_rows) != sum(
        pass_counts
    ):
        raise ValueError("residual-centering raw cost sample count changed")
    rows_by_pass: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(RESIDUAL_CENTERING_COST_ORDER))
    }
    latency_fields = (
        "input_pipeline_serial_ms",
        "h2d_ms",
        "model_forward_ms",
        "postprocess_ms",
        "decode_to_window_output_wall_ms",
        "final_video_nms_ms",
        "end_to_end_serial_ms",
    )
    component_fields = (
        "backbone_wrapper_ms",
        "scout_ms",
        "patch_embed_ms",
        "heavy_backbone_ms",
        "sparse_adapter_ms",
        "projection_ms",
        "neck_ms",
        "head_ms",
    )
    for row in raw_rows:
        pass_index = int(row.get("pass_index", -1))
        unsigned = dict(row)
        observed = unsigned.pop("sample_sha256", None)
        if (
            row.get("schema_version") != RESIDUAL_CENTERING_COST_SAMPLE_SCHEMA
            or pass_index not in rows_by_pass
            or row.get("arm") != RESIDUAL_CENTERING_COST_ORDER[pass_index]
            or row.get("population_sha256") != profile.get("population_sha256")
            or observed != canonical_sha256(unsigned)
            or _finite(row.get("gpu_energy_j"), "sample energy") <= 0.0
            or _finite(row.get("peak_gpu_allocated_mb"), "peak allocated") <= 0.0
            or _finite(row.get("peak_gpu_reserved_mb"), "peak reserved") <= 0.0
            or any(_finite(row.get(key), key) <= 0.0 for key in latency_fields)
            or any(_finite(row.get(key), key) <= 0.0 for key in component_fields)
        ):
            raise ValueError("residual-centering raw cost sample is invalid")
        ordinal = int(row.get("sample_ordinal", -1))
        physical_window_id = str(row.get("physical_window_id", "")).strip()
        window_id = str(row.get("window_id", "")).strip()
        if (
            ordinal < 0
            or int(row.get("loader_ordinal", -1)) != ordinal
            or not physical_window_id
            or window_id != make_profile_exposure_id(physical_window_id, ordinal)
        ):
            raise ValueError("residual-centering raw exposure identity is invalid")
        route = row.get("route_audit")
        expected_mode = residual_centering_training_variant_spec(row["arm"])[
            "branch_calibration_mode"
        ]
        if (
            not isinstance(route, Mapping)
            or int(route.get("exact_window_budget", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
            or int(route.get("padded_heavy_tokens", -1)) != 0
            or route.get("branch_calibration_mode") != expected_mode
        ):
            raise ValueError("residual-centering raw route audit is invalid")
        energy_window = row.get("energy_window_monotonic_s")
        nms_window = row.get("nms_energy_window_monotonic_s")
        if (
            not isinstance(energy_window, list)
            or len(energy_window) != 2
            or not isinstance(nms_window, list)
            or len(nms_window) != 2
        ):
            raise ValueError("residual-centering raw energy window is missing")
        energy_start, energy_end = map(float, energy_window)
        nms_start, nms_end = map(float, nms_window)
        expected_nms_ms = (
            (nms_end - nms_start) * 1000.0 / pass_counts[pass_index]
        )
        if (
            energy_end <= energy_start
            or nms_end <= nms_start
            or not math.isclose(
                float(row["final_video_nms_ms"]),
                expected_nms_ms,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["end_to_end_serial_ms"]),
                float(row["decode_to_window_output_wall_ms"])
                + float(row["final_video_nms_ms"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("residual-centering full-stack timing is inconsistent")
        rows_by_pass[pass_index].append(row)
    for pass_index, rows in rows_by_pass.items():
        rows.sort(key=lambda row: int(row["sample_ordinal"]))
        if (
            len(rows) != pass_counts[pass_index]
            or [int(row["sample_ordinal"]) for row in rows]
            != list(range(pass_counts[pass_index]))
            or len({str(row["window_id"]) for row in rows}) != len(rows)
            or canonical_sha256([row["window_id"] for row in rows])
            != pass_receipts[pass_index]["sample_manifest_sha256"]
        ):
            raise ValueError("residual-centering cost pass lineage is invalid")
        summary = pass_summaries[pass_index]
        if (
            int(summary.get("pass_index", -1)) != pass_index
            or summary.get("variant") != RESIDUAL_CENTERING_COST_ORDER[pass_index]
            or int(summary.get("sample_count", -1)) != len(rows)
        ):
            raise ValueError("residual-centering pass summary is invalid")
        for field in (*latency_fields, "gpu_energy_j"):
            expected = _summary([float(row[field]) for row in rows])
            if any(
                not math.isclose(
                    float(summary["metrics"][field][key]),
                    value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for key, value in expected.items()
            ):
                raise ValueError("residual-centering pass summary is not reproducible")
    for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
        rows = [row for row in raw_rows if row["arm"] == variant]
        summary = arm_summaries[variant]
        if (
            int(summary.get("pass_count", -1)) != 4
            or int(summary.get("sample_count", -1)) != len(rows)
            or summary.get("population_sha256") != profile.get("population_sha256")
        ):
            raise ValueError("residual-centering arm summary is invalid")
        for field in latency_fields:
            expected = _summary([float(row[field]) for row in rows])
            if any(
                not math.isclose(
                    float(summary["latency_ms"][field][key]),
                    value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for key, value in expected.items()
            ):
                raise ValueError("residual-centering arm latency is not reproducible")
        resources = summary.get("resources", {})
        expected_resources = {
            "peak_gpu_allocated_mb": max(float(row["peak_gpu_allocated_mb"]) for row in rows),
            "peak_gpu_reserved_mb": max(float(row["peak_gpu_reserved_mb"]) for row in rows),
            "gross_gpu_energy_j": sum(float(row["gpu_energy_j"]) for row in rows),
            "gpu_energy_j_per_sample": sum(float(row["gpu_energy_j"]) for row in rows)
            / len(rows),
        }
        if any(
            not math.isclose(
                float(resources[key]), value, rel_tol=1e-12, abs_tol=1e-12
            )
            for key, value in expected_resources.items()
        ):
            raise ValueError("residual-centering arm resources are not reproducible")

    power_rows = _load_jsonl_objects(
        artifact_paths["power_trace"], label="residual-centering power trace"
    )
    power_samples = [
        (
            _finite(row.get("monotonic_s"), "power timestamp"),
            _finite(row.get("power_w"), "power value"),
        )
        for row in power_rows
    ]
    if len(power_samples) < 2:
        raise ValueError("residual-centering power trace is incomplete")
    power_origin = power_samples[0][0]
    if any(
        int(row.get("sequence", -1)) != index
        or power <= 0.0
        or (index > 0 and timestamp <= power_samples[index - 1][0])
        or not math.isclose(
            _finite(row.get("timestamp_ms"), "normalized power timestamp"),
            (timestamp - power_origin) * 1000.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for index, (row, (timestamp, power)) in enumerate(
            zip(power_rows, power_samples)
        )
    ):
        raise ValueError("residual-centering power trace is invalid")
    for row in raw_rows:
        start, end = map(float, row["energy_window_monotonic_s"])
        nms_start, nms_end = map(float, row["nms_energy_window_monotonic_s"])
        sample_energy = _integrate_power_samples(power_samples, start=start, end=end)
        nms_energy = _integrate_power_samples(
            power_samples, start=nms_start, end=nms_end
        )
        expected_energy = (
            None
            if sample_energy is None or nms_energy is None
            else sample_energy + nms_energy / pass_counts[int(row["pass_index"])]
        )
        if expected_energy is None or not math.isclose(
            float(row["gpu_energy_j"]), expected_energy, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("residual-centering sample energy is not reproducible")
    sidecar = _load_json_object(
        artifact_paths["sidecar_attempt_report"],
        label="residual-centering sidecar attempt report",
    )
    unsigned_sidecar = dict(sidecar)
    observed_sidecar = unsigned_sidecar.pop("attempt_sha256", None)
    if (
        sidecar.get("status") != "PASS"
        or int(sidecar.get("interval_ms", -1))
        != RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS
        or sidecar.get("trace_io_inside_sampling_loop") is not False
        or sidecar.get("trace_file_sha256")
        != sha256_file(artifact_paths["sidecar_attempt_trace"])
        or observed_sidecar != canonical_sha256(unsigned_sidecar)
    ):
        raise ValueError("residual-centering power sidecar receipt is invalid")
    return profile


def finalize_residual_centering_cost(
    profile: Mapping[str, Any] | None,
    *,
    expected_model_runtime_commit: str,
    expected_execution_commit: str,
) -> dict[str, Any]:
    errors: dict[str, str] = {}
    validated = None
    analysis: dict[str, Any] = {}
    accuracy_deltas: dict[str, float] = {}
    try:
        if profile is None:
            raise ValueError("paired cost profile is missing")
        validated = validate_residual_centering_cost_profile(
            profile,
            expected_model_runtime_commit=expected_model_runtime_commit,
            expected_execution_commit=expected_execution_commit,
        )
        raw_path = Path(validated["artifact_receipts"]["raw_samples"]["path"])
        raw_rows = _load_jsonl_objects(raw_path, label="paired cost samples")
        analysis = analyze_residual_centering_paired_cost(raw_rows)
        source = validate_residual_centering_cost_source(
            validated["training_run_root"],
            expected_model_runtime_commit=expected_model_runtime_commit,
        )
        stages = source["stages"]
        none_metrics = stages["none_control"]["accuracy_replays"]["accuracy_a"][
            "metrics"
        ]
        center_metrics = stages["residual_window_center"]["accuracy_replays"][
            "accuracy_a"
        ]["metrics"]
        accuracy_deltas = {
            key: float(center_metrics[key]) - float(none_metrics[key])
            for key in none_metrics
        }
    except (
        AttributeError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        errors["paired_cost"] = f"{type(error).__name__}:{error}"
    accuracy_passed = bool(accuracy_deltas) and (
        accuracy_deltas["mAP@0.6"] > 0.0
        and accuracy_deltas["mAP@0.7"] > 0.0
        and accuracy_deltas["average_mAP"] >= 0.0
    )
    cost_noninferior = bool(analysis) and analysis.get("cost_noninferior") is True
    passed = not errors and validated is not None and accuracy_passed and cost_noninferior
    result: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_COST_FINALIZATION_SCHEMA,
        "study_id": RESIDUAL_CENTERING_COST_STUDY_ID,
        "status": (
            "PASS_ACCURACY_AND_PAIRED_COST_NONINFERIOR_SEEDS_AUTHORIZED"
            if passed
            else "HOLD_COMPLETE_PAIRED_COST_TRADEOFF_NO_SEEDS"
            if not errors and validated is not None
            else "FAIL_INCOMPLETE_PAIRED_COST_NO_INFERENCE"
        ),
        "decision": (
            "FREEZE_SEEDS_3408_3409_MATCHED_CONFIRMATION_PROTOCOL"
            if passed
            else "HOLD_RESIDUAL_CENTERING_NO_ADDITIONAL_SEEDS"
            if not errors and validated is not None
            else "INCOMPLETE_NO_INFERENCE"
        ),
        "model_runtime_commit": expected_model_runtime_commit,
        "execution_commit": expected_execution_commit,
        "errors": errors,
        "center_minus_none_metrics_pp": accuracy_deltas,
        "accuracy_screen_passed": accuracy_passed,
        "paired_cost_analysis": analysis,
        "paired_cost_noninferior": cost_noninferior,
        "strict_pareto_observed": bool(
            analysis and analysis.get("strict_pareto_observed") is True
        ),
        "seeds_3408_3409_opened": passed,
        "additional_seed_protocol_requires_separate_freeze": passed,
        "single_job_cost_is_paper_efficiency_claim": False,
        "independent_repeated_cost_jobs_required_for_paper": True,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    result["finalization_sha256"] = canonical_sha256(result)
    return result


validate_frozen_residual_centering_cost_contract()
