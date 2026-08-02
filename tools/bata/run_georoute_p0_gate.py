"""CUDA-only P0 gate for the native-token GeoRoute AdaTAD path.

This tool is deliberately a one-step synthetic-input check.  It validates one
real ``ActionFormer.forward_train`` plus backward pass on CUDA, but it neither
loads a dataset nor calls the official evaluator.  It is not a training or
accuracy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEOROUTE_P0_GATE_SCHEMA = "georoute_adatad_p0_cuda_one_step_gate_v5"
AMP_PRODUCTION_TUBELETS = 384
AMP_PRODUCTION_SOURCE_HEIGHT = 180
AMP_PRODUCTION_SOURCE_WIDTH = 320
AMP_PRODUCTION_PATCH_SIZE = 16
AMP_PRODUCTION_PATCH_CAPACITY = (AMP_PRODUCTION_SOURCE_HEIGHT // AMP_PRODUCTION_PATCH_SIZE) * (AMP_PRODUCTION_SOURCE_WIDTH // AMP_PRODUCTION_PATCH_SIZE)
AMP_PRODUCTION_TARGET_K = 64
AMP_FULL_GRAPH_FLOOR_SCALE = 256.0


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _load_rendezvous_binding(
    *,
    runtime_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    from tools.bata.georoute_rendezvous_gate import (
        validate_rendezvous_gate_receipt,
    )

    raw_path = os.environ.get("GEOROUTE_P0_RENDEZVOUS_RECEIPT")
    if not raw_path:
        raise RuntimeError("P0 lacks its same-leaf rendezvous isolation receipt")
    candidate = Path(raw_path)
    if candidate.is_symlink():
        raise ValueError(f"P0 rendezvous isolation receipt cannot be a symlink: {candidate}")
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"P0 rendezvous isolation receipt is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P0 rendezvous isolation receipt is not a JSON object")
    validate_rendezvous_gate_receipt(
        payload,
        expected_commit=runtime_commit,
        expected_node_name=socket.gethostname(),
    )
    if str(payload.get("slurm_job_id")) != slurm_job_id:
        raise ValueError("P0 model gate and rendezvous gate used different Slurm leaves")
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "gate_sha256": str(payload["gate_sha256"]),
        "slurm_job_id": slurm_job_id,
        "status": str(payload["status"]),
    }


def build_p0_gate_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add a deterministic self-hash to a completed P0 report."""

    report = dict(payload)
    report.pop("report_sha256", None)
    report["report_sha256"] = _sha256(report)
    return report


def validate_p0_gate_report(report: Mapping[str, Any]) -> None:
    """Fail closed if the JSON claims more than the P0 gate established."""

    if report.get("schema_version") != GEOROUTE_P0_GATE_SCHEMA:
        raise ValueError("unexpected GeoRoute P0 report schema")
    without_hash = dict(report)
    observed_hash = without_hash.pop("report_sha256", None)
    if observed_hash != _sha256(without_hash):
        raise ValueError("P0 report self-hash mismatch")
    if report.get("status") != "PASS":
        raise ValueError("P0 report is not PASS")
    scope = report.get("p0_scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(key) is not expected
        for key, expected in {
            "synthetic_inputs_only": True,
            "full_training": False,
            "official_evaluation": False,
        }.items()
    ):
        raise ValueError("P0 scope must remain synthetic and exclude official evaluation")
    if report.get("official_test_opened") is not False:
        raise ValueError("P0 gate must not open official test")
    if int(report.get("heavy_backbone_forward_count", -1)) != 1:
        raise ValueError("P0 requires exactly one heavy backbone forward")
    if int(report.get("shared_backbone_instances", -1)) != 1:
        raise ValueError("P0 requires exactly one shared backbone instance")
    if report.get("uses_grid_sample") is not False:
        raise ValueError("P0 forbids grid_sample local crop resampling")
    if report.get("uses_resized_local_crop") is not False:
        raise ValueError("P0 forbids resized local crop paths")
    exact_k = report.get("exact_k")
    if not isinstance(exact_k, Mapping):
        raise ValueError("P0 exact-K evidence is missing")
    target = int(exact_k.get("target_k", -1))
    if target <= 0 or int(exact_k.get("observed_min", -1)) != target or int(exact_k.get("observed_max", -1)) != target:
        raise ValueError("P0 exact-K count differs from the requested target")
    if int(exact_k.get("duplicates", -1)) != 0:
        raise ValueError("P0 route contains duplicate native tokens")
    estimator = report.get("estimator")
    valid_estimators = {
        ("none", "no_policy_gradient"),
        ("straight_through", "biased_straight_through"),
        ("score_function", "score_function_candidate"),
    }
    if not isinstance(estimator, Mapping) or (estimator.get("name"), estimator.get("claim")) not in valid_estimators:
        raise ValueError("P0 estimator label is invalid or overclaims unbiasedness")
    source_grid = report.get("source_grid")
    if not isinstance(source_grid, Mapping) or int(source_grid.get("patch_capacity", 0)) <= 0:
        raise ValueError("P0 source-grid evidence is missing")
    amp_horizon = report.get("score_function_amp_horizon")
    if not isinstance(amp_horizon, Mapping):
        raise ValueError("P0 score-function AMP horizon evidence is missing")
    full_graph_amp = report.get("score_function_full_graph_amp")
    if not isinstance(full_graph_amp, Mapping):
        raise ValueError("P0 score-function full-graph AMP evidence is missing")
    if estimator.get("name") == "score_function":
        if (
            amp_horizon.get("status") != "PASS_AMP_PRODUCTION_HORIZON"
            or amp_horizon.get("passed") is not True
            or amp_horizon.get("source_dtype") != "torch.float16"
            or amp_horizon.get("likelihood_dtype") != "torch.float32"
            or amp_horizon.get("policy_loss_dtype") != "torch.float32"
            or int(amp_horizon.get("tubelets", -1)) != AMP_PRODUCTION_TUBELETS
            or int(amp_horizon.get("patch_capacity", -1)) != AMP_PRODUCTION_PATCH_CAPACITY
            or int(amp_horizon.get("patch_capacity", -1)) != int(source_grid.get("patch_capacity", -2))
            or int(amp_horizon.get("target_k", -1)) != AMP_PRODUCTION_TARGET_K
            or target != AMP_PRODUCTION_TARGET_K
            or int(source_grid.get("height", -1)) != AMP_PRODUCTION_SOURCE_HEIGHT
            or int(source_grid.get("width", -1)) != AMP_PRODUCTION_SOURCE_WIDTH
            or int(source_grid.get("patch_size", -1)) != AMP_PRODUCTION_PATCH_SIZE
            or int(source_grid.get("grid_height", -1)) != AMP_PRODUCTION_SOURCE_HEIGHT // AMP_PRODUCTION_PATCH_SIZE
            or int(source_grid.get("grid_width", -1)) != AMP_PRODUCTION_SOURCE_WIDTH // AMP_PRODUCTION_PATCH_SIZE
            or float(amp_horizon.get("loss_scale", -1.0)) != 256.0
            or amp_horizon.get("all_likelihoods_finite") is not True
            or amp_horizon.get("policy_loss_finite") is not True
            or amp_horizon.get("all_scaled_gradients_finite") is not True
            or float(amp_horizon.get("policy_loss_abs", 0.0)) <= float(amp_horizon.get("fp16_max", 0.0))
        ):
            raise ValueError("P0 score-function AMP production-horizon check did not pass")
        if (
            full_graph_amp.get("status") != "PASS_FULL_GRAPH_AMP_OPTIMIZER_UPDATE"
            or full_graph_amp.get("executed") is not True
            or full_graph_amp.get("autocast_dtype") != "torch.float16"
            or float(full_graph_amp.get("loss_scale_before", -1.0)) != AMP_FULL_GRAPH_FLOOR_SCALE
            or float(full_graph_amp.get("loss_scale_after", -1.0)) < AMP_FULL_GRAPH_FLOOR_SCALE
            or full_graph_amp.get("optimizer_update_succeeded") is not True
            or full_graph_amp.get("all_required_gradients_finite") is not True
            or full_graph_amp.get("scout_autocast_enabled") is not False
            or full_graph_amp.get("scout_compute_dtype") != "torch.float32"
            or full_graph_amp.get("model_backward_scope") != "detector_plus_score_function"
        ):
            raise ValueError("P0 score-function full-graph AMP optimizer update did not pass")
    elif amp_horizon.get("status") != "NOT_APPLICABLE_NON_SCORE_FUNCTION" or amp_horizon.get("executed") is not False:
        raise ValueError("P0 non-score-function arm has invalid AMP horizon evidence")
    elif full_graph_amp.get("status") != "NOT_APPLICABLE_NON_SCORE_FUNCTION" or full_graph_amp.get("executed") is not False:
        raise ValueError("P0 non-score-function arm has invalid full-graph AMP evidence")
    memory = report.get("memory")
    if not isinstance(memory, Mapping) or int(memory.get("peak_allocated_bytes", 0)) <= 0:
        raise ValueError("P0 CUDA memory evidence is missing")
    detector = report.get("detector")
    if not isinstance(detector, Mapping) or detector.get("training_forward") is not True or detector.get("backward_completed") is not True:
        raise ValueError("P0 requires a completed detector training forward and backward")
    detector_loss_keys = detector.get("detector_loss_keys")
    if not isinstance(detector_loss_keys, list) or set(detector_loss_keys) != {"cls_loss", "reg_loss"}:
        raise ValueError("P0 must backpropagate the real AdaTAD classification and regression losses")
    gradient = report.get("gradient")
    if not isinstance(gradient, Mapping) or gradient.get("all_required_gradients_finite") is not True:
        raise ValueError("P0 required gradients must be finite")
    if not gradient.get("nonzero_components"):
        raise ValueError("P0 did not record nonzero gradient components")
    if gradient.get("missing_required_components"):
        raise ValueError("P0 is missing a required detector-to-router gradient path")
    native_route = report.get("native_route")
    if not isinstance(native_route, Mapping):
        raise ValueError("P0 native-route evidence is missing")
    selected_shape = native_route.get("selected_native_tubelet_shape")
    if not isinstance(selected_shape, list) or len(selected_shape) != 7:
        raise ValueError("P0 selected native-tubelet evidence is missing")
    if int(selected_shape[2]) != target:
        raise ValueError("P0 selected native-tubelet shape disagrees with exact-K")
    output_shape = native_route.get("output_shape")
    if not isinstance(output_shape, list) or len(output_shape) != 3 or int(output_shape[-1]) != 768:
        raise ValueError("P0 did not retain the required [B,C,768] detector feature contract")
    if int(native_route.get("selected_unique_count_min", -1)) != target or int(native_route.get("selected_unique_count_max", -1)) != target:
        raise ValueError("P0 native route did not independently observe exact unique-K selection")
    before = int(native_route.get("native_packed_invocation_counter_before", -1))
    after = int(native_route.get("native_packed_invocation_counter_after", -1))
    if before < 0 or after - before != 1:
        raise ValueError("P0 native packed invocation counter is inconsistent with one heavy forward")
    if report.get("route_mode") not in {
        "dense",
        "uniform",
        "random",
        "roi",
        "free",
        "hybrid",
        "structured_context_residual",
        "structured_context_roi",
        "structured_hybrid",
        "structured_hybrid_geometry_shift",
    }:
        raise ValueError("P0 route mode is missing or unsupported")
    if report.get("route_mode") == "dense":
        dense_reference = report.get("dense_native_reference")
        if (
            not isinstance(dense_reference, Mapping)
            or dense_reference.get("passed") is not True
            or int(dense_reference.get("reference_heavy_backbone_forward_count", -1)) != 1
            or int(dense_reference.get("real_route_heavy_backbone_forward_count", -1)) != 1
            or dense_reference.get("reference_autograd_mode") != "enabled_matches_real_packed_forward"
        ):
            raise ValueError("dense P0 must include a passed native dense numerical reference")
    if report.get("estimator", {}).get("name") == "score_function":
        policy_evidence = report.get("score_function_detector_binding")
        if not isinstance(policy_evidence, Mapping) or not {
            "cls_loss",
            "reg_loss",
        } <= set(policy_evidence.get("detector_loss_keys", [])):
            raise ValueError("P0 score-function route is not bound to the real detector losses")
    component_trace = report.get("component_trace")
    if (
        not isinstance(component_trace, Mapping)
        or int(component_trace.get("packed_attention_forward_count", 0)) <= 0
        or int(component_trace.get("packed_mlp_forward_count", 0)) <= 0
        or int(component_trace.get("packed_adapter_forward_count", 0)) <= 0
        or int(component_trace.get("dense_adapter_forward_count", -1)) != 0
    ):
        raise ValueError("P0 component trace does not prove fully packed execution")
    checkpoint_receipt = report.get("checkpoint_receipt")
    if not isinstance(checkpoint_receipt, Mapping) or int(checkpoint_receipt.get("checkpoint_count", -1)) != 0:
        raise ValueError("P0 must explicitly record that it creates no checkpoint")
    storage_receipt = report.get("storage_receipt")
    if not isinstance(storage_receipt, Mapping) or storage_receipt.get("status") != "PASS_STORAGE_PREFLIGHT" or storage_receipt.get("atomic_publish_peak_included") is not True:
        raise ValueError("P0 storage preflight receipt is missing")
    measurement = report.get("checkpoint_storage_measurement")
    if (
        not isinstance(measurement, Mapping)
        or int(measurement.get("checkpoint_upper_bound_bytes", 0)) <= 0
        or int(measurement.get("auxiliary_upper_bound_bytes_per_cell", 0)) <= 0
        or measurement.get("checkpoint_policy") != "final_only"
    ):
        raise ValueError("P0 checkpoint storage measurement is missing")
    runtime_commit = report.get("runtime_commit")
    if not isinstance(runtime_commit, str) or len(runtime_commit) != 40:
        raise ValueError("P0 runtime commit is missing")
    slurm_job_id = report.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id:
        raise ValueError("P0 Slurm job identity is missing")
    rendezvous = report.get("rendezvous_isolation")
    if (
        not isinstance(rendezvous, Mapping)
        or rendezvous.get("status") != "PASS_CONCURRENT_RENDEZVOUS_ISOLATION"
        or rendezvous.get("slurm_job_id") != slurm_job_id
        or not isinstance(rendezvous.get("gate_sha256"), str)
        or len(str(rendezvous.get("gate_sha256"))) != 64
        or not isinstance(rendezvous.get("file_sha256"), str)
        or len(str(rendezvous.get("file_sha256"))) != 64
    ):
        raise ValueError("P0 report is not bound to its same-leaf rendezvous gate")
    pilot_arm = report.get("pilot_arm")
    if pilot_arm is not None:
        from tools.bata.georoute_estimator_pilot_contract import (
            REPRESENTATION_KEYS,
            pilot_arm_spec,
        )

        if not isinstance(pilot_arm, str):
            raise ValueError("P0 pilot arm must be a registered string")
        spec = pilot_arm_spec(pilot_arm)
        route_parameters = report.get("route_parameters")
        representation = report.get("representation")
        if not isinstance(route_parameters, Mapping):
            raise ValueError("P0 pilot route-parameter binding is missing")
        if not isinstance(representation, Mapping):
            raise ValueError("P0 pilot representation binding is missing")
        if (
            report.get("route_mode") != spec["route_mode"]
            or estimator.get("name") != spec["policy_estimator"]
            or target != int(spec["tokens_per_tubelet"])
            or int(route_parameters.get("context_tokens", -1)) != int(spec["context_tokens"])
            or float(route_parameters.get("roi_fraction", -1.0)) != float(spec["roi_fraction"])
            or float(route_parameters.get("policy_temperature", -1.0)) != float(spec["policy_temperature"])
            or float(route_parameters.get("score_function_weight", -1.0)) != float(spec["score_function_weight"])
            or float(
                route_parameters.get(
                    "score_function_baseline_momentum",
                    -1.0,
                )
            )
            != float(spec["score_function_baseline_momentum"])
        ):
            raise ValueError("P0 pilot route binding differs from the frozen arm")
        expected_representation = {
            "absolute_position_enabled": bool(spec["absolute_position_enabled"]),
            "geometry_side_channel": bool(spec["geometry_side_channel"]),
            "learned_geometry_enabled": bool(spec["learned_geometry_enabled"]),
            "learned_residual_enabled": bool(spec["learned_residual_enabled"]),
            **{key: bool(spec[key]) for key in REPRESENTATION_KEYS},
        }
        if {key: representation.get(key) for key in expected_representation} != expected_representation:
            raise ValueError("P0 pilot representation binding differs from the frozen arm")
        required = {
            "rpn_head",
            "projection",
            "sparse_adapter",
            "videomae_adapter",
        }
        if spec["learned_geometry_enabled"]:
            required.add("scout_geometry")
        if spec["learned_residual_enabled"]:
            required.add("scout_residual")
        if set(gradient.get("required_components", [])) != required:
            raise ValueError("P0 pilot gradient contract differs from the frozen arm")
    hybrid_causal_arm = report.get("hybrid_causal_arm")
    if hybrid_causal_arm is not None:
        from tools.bata.georoute_hybrid_causal_contract import (
            HYBRID_CAUSAL_K,
            hybrid_causal_arm_spec,
        )

        if not isinstance(hybrid_causal_arm, str):
            raise ValueError("P0 Hybrid causal arm must be a registered string")
        spec = hybrid_causal_arm_spec(hybrid_causal_arm)
        route_parameters = report.get("route_parameters")
        representation = report.get("representation")
        expected_target = (
            int(report["source_grid"]["patch_capacity"])
            if spec["route_mode"] == "dense"
            else HYBRID_CAUSAL_K
        )
        if (
            report.get("route_mode") != spec["route_mode"]
            or estimator.get("name") != spec["policy_estimator"]
            or target != expected_target
            or int(route_parameters.get("context_tokens", -1))
            != int(spec["context_tokens"])
            or int(route_parameters.get("structured_roi_tokens", -1))
            != int(spec["roi_tokens"])
            or int(route_parameters.get("structured_residual_tokens", -1))
            != int(spec["residual_tokens"])
            or int(route_parameters.get("geometry_temporal_shift_tubelets", -1))
            != int(spec["geometry_temporal_shift_tubelets"])
            or float(route_parameters.get("policy_temperature", -1.0)) != 0.7
        ):
            raise ValueError("P0 Hybrid causal route differs from its frozen arm")
        expected_representation = {
            "absolute_position_enabled": True,
            "absolute_coordinates_enabled": False,
            "roi_relative_coordinates_enabled": False,
            "geometry_projection_enabled": False,
            "geometry_side_channel": False,
            "learned_geometry_enabled": spec["roi_tokens"] > 0,
            "learned_residual_enabled": spec["residual_tokens"] > 0,
        }
        if {
            key: representation.get(key) for key in expected_representation
        } != expected_representation:
            raise ValueError("P0 Hybrid causal representation isolation changed")
        structured_audit = report.get("structured_route_audit")
        if spec["route_mode"].startswith("structured_"):
            if not isinstance(structured_audit, Mapping):
                raise ValueError("P0 Hybrid causal structured-route audit is missing")
            expected_roles = {
                "context": int(spec["context_tokens"]),
                "roi": int(spec["roi_tokens"]),
                "residual": int(spec["residual_tokens"]),
                "free": 0,
                "dense": 0,
                "uniform": 0,
                "random": 0,
            }
            if (
                structured_audit.get("routing_schema")
                != "georoute_fixed_quota_structured_routing_v1"
                or structured_audit.get("role_counts") != expected_roles
            ):
                raise ValueError("P0 Hybrid causal structured role contract changed")
            route_rng = structured_audit.get("route_rng")
            if estimator.get("name") == "score_function" and (
                not isinstance(route_rng, Mapping)
                or route_rng.get("schema_version")
                != "georoute_route_private_rng_v1"
                or route_rng.get("enabled") is not True
                or route_rng.get("global_rng_consumed") is not False
                or int(route_rng.get("study_seed", -1)) != 5227
                or int(route_rng.get("successful_update_index", -1)) != 0
            ):
                raise ValueError("P0 Hybrid causal private route RNG contract failed")
            telemetry = structured_audit.get("diagnostic_telemetry")
            if (
                not isinstance(telemetry, Mapping)
                or telemetry.get("schema_version")
                != "georoute_diagnostic_window_telemetry_v2"
                or telemetry.get("role_counts") != expected_roles
            ):
                raise ValueError("P0 Hybrid causal structured telemetry is missing")
            for key in (
                "original_trajectory_sha256",
                "routing_trajectory_sha256",
            ):
                digest = telemetry.get("geometry", {}).get(key)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("P0 Hybrid causal geometry trajectory hash is invalid")
            if estimator.get("name") == "score_function":
                for role in ("roi", "residual"):
                    if int(expected_roles[role]) == 0:
                        continue
                    gradient_record = telemetry.get("branch_gradient", {}).get(role)
                    if (
                        not isinstance(gradient_record, Mapping)
                        or gradient_record.get("applicable") is not True
                        or gradient_record.get("observed") is not True
                        or gradient_record.get("finite") is not True
                        or not isinstance(gradient_record.get("l2_norm"), (int, float))
                        or float(gradient_record["l2_norm"]) <= 0.0
                    ):
                        raise ValueError(
                            f"P0 Hybrid causal {role} policy gradient telemetry failed"
                        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Existing AdaTAD VideoMAE config; only materialized in memory.",
    )
    parser.add_argument("--output", required=True, help="Atomic JSON report path.")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional VideoMAE checkpoint overriding config.custom.pretrain.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--route-mode",
        choices=(
            "dense",
            "uniform",
            "random",
            "roi",
            "free",
            "hybrid",
            "structured_context_residual",
            "structured_context_roi",
            "structured_hybrid",
            "structured_hybrid_geometry_shift",
        ),
        default="hybrid",
    )
    parser.add_argument(
        "--policy-estimator",
        choices=("none", "straight_through", "score_function"),
        default="straight_through",
    )
    parser.add_argument("--tokens-per-tubelet", type=int, default=32)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--structured-roi-tokens", type=int, default=0)
    parser.add_argument("--structured-residual-tokens", type=int, default=0)
    parser.add_argument(
        "--geometry-temporal-shift-tubelets",
        type=int,
        default=0,
    )
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    parser.add_argument("--policy-temperature", type=float, default=0.7)
    parser.add_argument("--score-function-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-function-baseline-momentum",
        type=float,
        default=0.95,
    )
    parser.add_argument("--pilot-arm", default=None)
    parser.add_argument("--hybrid-causal-arm", default=None)
    parser.add_argument("--geometry-side-channel", type=_optional_bool, default=None)
    parser.add_argument("--absolute-position-enabled", type=_optional_bool, default=None)
    parser.add_argument("--absolute-coordinates-enabled", type=_optional_bool, default=None)
    parser.add_argument("--roi-relative-coordinates-enabled", type=_optional_bool, default=None)
    parser.add_argument("--geometry-projection-enabled", type=_optional_bool, default=None)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _configure_in_memory(config_path: Path, args):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    if (
        not math.isfinite(float(args.policy_temperature))
        or float(args.policy_temperature) <= 0.0
        or not math.isfinite(float(args.score_function_weight))
        or float(args.score_function_weight) <= 0.0
        or not math.isfinite(float(args.score_function_baseline_momentum))
        or not 0.0 <= float(args.score_function_baseline_momentum) < 1.0
    ):
        raise ValueError("P0 estimator hyperparameters are outside the frozen domain")
    backbone = cfg.model.backbone
    backbone.backbone.with_cp = False
    custom = backbone.custom
    custom.wrapper_type = "georoute_native_packed_v1"
    custom.georoute_source_key = "source"
    custom.georoute_scout_key = "scout"
    custom.georoute_window_size = 768
    custom.georoute_scout_size = 96
    custom.georoute_patch_size = 16
    custom.georoute_tubelet_size = 2
    custom.georoute_tokens_per_tubelet = int(args.tokens_per_tubelet)
    custom.georoute_context_tokens = int(args.context_tokens)
    custom.georoute_structured_context_tokens = int(args.context_tokens)
    custom.georoute_structured_roi_tokens = int(args.structured_roi_tokens)
    custom.georoute_structured_residual_tokens = int(
        args.structured_residual_tokens
    )
    custom.georoute_geometry_temporal_shift_tubelets = int(
        args.geometry_temporal_shift_tubelets
    )
    custom.georoute_roi_fraction = float(args.roi_fraction)
    custom.georoute_route_mode = str(args.route_mode)
    custom.georoute_policy_estimator = str(args.policy_estimator)
    custom.georoute_policy_temperature = float(args.policy_temperature)
    custom.georoute_score_function_weight = float(args.score_function_weight)
    custom.georoute_score_function_baseline_momentum = float(args.score_function_baseline_momentum)
    custom.georoute_score_function_temporal_reduction = "mean"
    custom.georoute_route_study_seed = int(args.seed)
    custom.georoute_pooling_mode = "uniform_selected"
    custom.georoute_adapter_mode = "coordinate_lineage_packed"
    custom.georoute_roi_temperature = 0.25
    custom.georoute_min_roi_extent = 0.2
    custom.georoute_max_roi_extent = 1.0
    custom.georoute_geometry_smoothness_weight = 0.0
    custom.georoute_area_prior_weight = 0.0
    custom.georoute_diagnostic_telemetry_enabled = bool(
        args.hybrid_causal_arm is not None
    )
    for argument_name, config_name in (
        ("geometry_side_channel", "georoute_geometry_side_channel"),
        ("absolute_position_enabled", "georoute_absolute_position_enabled"),
        ("absolute_coordinates_enabled", "georoute_absolute_coordinates_enabled"),
        (
            "roi_relative_coordinates_enabled",
            "georoute_roi_relative_coordinates_enabled",
        ),
        ("geometry_projection_enabled", "georoute_geometry_projection_enabled"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            setattr(custom, config_name, bool(value))
    custom.georoute_p0_dense_reference_check = args.route_mode == "dense"
    custom.georoute_max_batch_size = 1
    custom.norm_eval = False
    if args.pretrained is not None:
        custom.pretrain = str(Path(args.pretrained).resolve())
    elif not getattr(custom, "pretrain", None):
        raise ValueError("P0 requires a real VideoMAE checkpoint via config.custom.pretrain or --pretrained")
    checkpoint = Path(str(custom.pretrain))
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
        custom.pretrain = str(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GeoRoute P0 VideoMAE checkpoint does not exist: {checkpoint}")
    if args.policy_estimator == "score_function" and args.route_mode == "hybrid":
        raise ValueError("score-function P0 requires roi/free, not staged hybrid")
    if args.route_mode == "dense" and args.policy_estimator != "none":
        raise ValueError("dense P0 parity uses estimator=none")
    if args.route_mode == "dense" and args.context_tokens != 0:
        raise ValueError("dense P0 parity requires context_tokens=0")
    if args.route_mode not in {"dense", "uniform", "random"} and args.policy_estimator == "none":
        raise ValueError("learned P0 routes require an explicit estimator")
    if args.route_mode in {"uniform", "random"} and args.policy_estimator != "none":
        raise ValueError("uniform/random P0 controls require estimator=none")
    if args.route_mode.startswith("structured_"):
        if (
            int(args.context_tokens)
            + int(args.structured_roi_tokens)
            + int(args.structured_residual_tokens)
            != int(args.tokens_per_tubelet)
        ):
            raise ValueError("structured P0 quotas must sum to exact K")
    if args.pilot_arm is not None:
        from tools.bata.georoute_estimator_pilot_contract import pilot_arm_spec

        spec = pilot_arm_spec(args.pilot_arm)
        observed = {
            "route_mode": str(args.route_mode),
            "policy_estimator": str(args.policy_estimator),
            "tokens_per_tubelet": int(args.tokens_per_tubelet),
            "context_tokens": int(args.context_tokens),
            "roi_fraction": float(args.roi_fraction),
            "policy_temperature": float(args.policy_temperature),
            "score_function_weight": float(args.score_function_weight),
            "score_function_baseline_momentum": float(args.score_function_baseline_momentum),
            "geometry_side_channel": bool(getattr(custom, "georoute_geometry_side_channel", False)),
            "absolute_position_enabled": bool(getattr(custom, "georoute_absolute_position_enabled", True)),
            "absolute_coordinates_enabled": bool(getattr(custom, "georoute_absolute_coordinates_enabled", True)),
            "roi_relative_coordinates_enabled": bool(
                getattr(
                    custom,
                    "georoute_roi_relative_coordinates_enabled",
                    getattr(
                        custom,
                        "georoute_absolute_coordinates_enabled",
                        True,
                    ),
                )
            ),
            "geometry_projection_enabled": bool(getattr(custom, "georoute_geometry_projection_enabled", True)),
        }
        for key, value in observed.items():
            if value != spec[key]:
                raise ValueError(f"P0 pilot arm {args.pilot_arm!r} has mismatched {key}: " f"{value!r} != {spec[key]!r}")
    if args.hybrid_causal_arm is not None:
        from tools.bata.georoute_hybrid_causal_contract import (
            HYBRID_CAUSAL_K,
            HYBRID_CAUSAL_SEED,
            hybrid_causal_arm_spec,
        )

        if args.pilot_arm is not None:
            raise ValueError("P0 cannot bind two pilot namespaces")
        spec = hybrid_causal_arm_spec(args.hybrid_causal_arm)
        expected_tokens = (
            (int(args.height) // 16) * (int(args.width) // 16)
            if spec["route_mode"] == "dense"
            else HYBRID_CAUSAL_K
        )
        observed = {
            "route_mode": str(args.route_mode),
            "policy_estimator": str(args.policy_estimator),
            "tokens_per_tubelet": int(args.tokens_per_tubelet),
            "context_tokens": int(args.context_tokens),
            "roi_tokens": int(args.structured_roi_tokens),
            "residual_tokens": int(args.structured_residual_tokens),
            "geometry_temporal_shift_tubelets": int(
                args.geometry_temporal_shift_tubelets
            ),
            "seed": int(args.seed),
            "absolute_position_enabled": bool(
                getattr(custom, "georoute_absolute_position_enabled", True)
            ),
            "absolute_coordinates_enabled": bool(
                getattr(custom, "georoute_absolute_coordinates_enabled", True)
            ),
            "roi_relative_coordinates_enabled": bool(
                getattr(custom, "georoute_roi_relative_coordinates_enabled", True)
            ),
            "geometry_projection_enabled": bool(
                getattr(custom, "georoute_geometry_projection_enabled", True)
            ),
        }
        expected = {
            "route_mode": spec["route_mode"],
            "policy_estimator": spec["policy_estimator"],
            "tokens_per_tubelet": expected_tokens,
            "context_tokens": spec["context_tokens"],
            "roi_tokens": spec["roi_tokens"],
            "residual_tokens": spec["residual_tokens"],
            "geometry_temporal_shift_tubelets": spec[
                "geometry_temporal_shift_tubelets"
            ],
            "seed": HYBRID_CAUSAL_SEED,
            "absolute_position_enabled": True,
            "absolute_coordinates_enabled": False,
            "roi_relative_coordinates_enabled": False,
            "geometry_projection_enabled": False,
        }
        if observed != expected:
            raise ValueError(
                f"P0 Hybrid causal arm {args.hybrid_causal_arm!r} mismatch: "
                f"{observed!r} != {expected!r}"
            )
    return cfg


def _gradient_summary(model, *, required_components: set[str]) -> dict[str, Any]:
    components: set[str] = set()
    nonfinite: list[str] = []
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(__import__("torch").isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
        elif bool(__import__("torch").count_nonzero(parameter.grad).item()):
            if name.startswith("backbone.scout.geometry_head"):
                components.add("scout_geometry")
            elif name.startswith("backbone.scout.residual_head"):
                components.add("scout_residual")
            elif name.startswith("backbone.scout"):
                components.add("scout_stem")
            elif name.startswith("backbone.sparse_adapter"):
                components.add("sparse_adapter")
            elif ".adapter." in f".{name}":
                components.add("videomae_adapter")
            elif name.startswith("projection"):
                components.add("projection")
            elif name.startswith("rpn_head"):
                components.add("rpn_head")
            elif name.startswith("neck"):
                components.add("neck")
            else:
                components.add(name.split(".", 1)[0])
    if nonfinite:
        raise FloatingPointError(f"non-finite gradients: {nonfinite}")
    if not components:
        raise RuntimeError("no nonzero gradient reached any trainable GeoRoute/AdaTAD component")
    missing_required = sorted(required_components - components)
    return {
        "all_required_gradients_finite": not nonfinite and not missing_required,
        "nonzero_components": sorted(components),
        "missing_trainable_gradient_tensors": sorted(missing),
        "required_components": sorted(required_components),
        "missing_required_components": missing_required,
    }


def _detector_only_objective(losses: Mapping[str, Any]):
    """Extract AdaTAD detector losses without GeoRoute regularizers/policy loss."""

    excluded = {
        "cost",
        "georoute_geometry_regularization_loss",
        "georoute_score_function_loss",
    }
    detector_terms = {key: value for key, value in losses.items() if key not in excluded and __import__("torch").is_tensor(value)}
    if not detector_terms:
        raise RuntimeError("P0 did not expose any detector-only loss term")
    required_detector_terms = {"cls_loss", "reg_loss"}
    if set(detector_terms) != required_detector_terms:
        raise RuntimeError("P0 requires the real AdaTAD classification and regression losses; observed " + ", ".join(sorted(detector_terms)))
    detector_cost = sum(detector_terms.values())
    if detector_cost.ndim != 0 or not bool(__import__("torch").isfinite(detector_cost).item()):
        raise FloatingPointError("P0 detector-only objective is absent or non-finite")
    return detector_cost, sorted(detector_terms)


def _run_cuda_gate(args) -> dict[str, Any]:
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GeoRoute P0 is CUDA-only and requires a Slurm-provided CUDA device")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("P0 source height and width must be positive")
    source_capacity = (args.height // 16) * (args.width // 16)
    if source_capacity <= 0:
        raise ValueError("P0 source is smaller than one complete native patch")
    if args.policy_estimator == "score_function" and (
        int(args.height) != AMP_PRODUCTION_SOURCE_HEIGHT
        or int(args.width) != AMP_PRODUCTION_SOURCE_WIDTH
        or source_capacity != AMP_PRODUCTION_PATCH_CAPACITY
        or int(args.tokens_per_tubelet) != AMP_PRODUCTION_TARGET_K
    ):
        raise ValueError("score-function P0 must reproduce the production " "180x320/T384/N220/K64 route horizon")
    if args.tokens_per_tubelet <= 0 or args.tokens_per_tubelet > source_capacity:
        raise ValueError("P0 tokens_per_tubelet must lie in the native source-grid capacity")
    if args.route_mode == "dense" and args.tokens_per_tubelet != source_capacity:
        raise ValueError("dense P0 numerical reference must select every native source token")
    if not (0 <= args.context_tokens < args.tokens_per_tubelet):
        raise ValueError("P0 context_tokens must lie in [0,K)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if str(device) != "cuda:0":
        raise RuntimeError("P0 must use logical cuda:0 assigned by Slurm")
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = _configure_in_memory(config_path, args)

    from opentad.models import build_detector

    model = build_detector(cfg.model).to(device)
    model.train()
    if args.route_mode.startswith("structured_"):
        model.backbone.set_successful_update_index(0)
    torch.cuda.reset_peak_memory_stats(device)
    source = torch.randint(
        low=0,
        high=256,
        size=(1, 1, 3, 768, args.height, args.width),
        dtype=torch.uint8,
        device=device,
    )
    scout = torch.randint(
        low=0,
        high=256,
        size=(1, 1, 3, 768, 96, 96),
        dtype=torch.uint8,
        device=device,
    )
    masks = torch.ones((1, 768), dtype=torch.bool, device=device)
    gt_segments = [torch.tensor([[96.0, 240.0]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([0], dtype=torch.long, device=device)]
    metas = [{"video_name": "georoute_p0_synthetic", "fps": 30.0, "duration": 25.6}]

    score_function_full_graph = args.policy_estimator == "score_function"
    with torch.cuda.amp.autocast(
        dtype=torch.float16,
        enabled=score_function_full_graph,
    ):
        losses = model.forward_train(
            {"source": source, "scout": scout},
            masks=masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )
    geometry_regularizer = losses.get("georoute_geometry_regularization_loss")
    if not torch.is_tensor(geometry_regularizer) or float(geometry_regularizer.detach().abs().item()) != 0.0:
        raise RuntimeError("P0 must disable geometry regularization before detector-gradient auditing")
    detector_cost, detector_loss_keys = _detector_only_objective(losses)
    required_components = {
        "rpn_head",
        "projection",
        "sparse_adapter",
        "videomae_adapter",
    }
    if args.route_mode == "hybrid":
        required_components.update(("scout_geometry", "scout_residual"))
    elif args.route_mode.startswith("structured_"):
        if int(args.structured_roi_tokens) > 0:
            required_components.add("scout_geometry")
        if int(args.structured_residual_tokens) > 0:
            required_components.add("scout_residual")
    elif args.route_mode in {"roi", "free"}:
        required_components.add("scout_geometry" if args.route_mode == "roi" else "scout_residual")
    elif args.route_mode == "uniform" and bool(
        getattr(
            cfg.model.backbone.custom,
            "georoute_geometry_side_channel",
            False,
        )
    ):
        required_components.add("scout_geometry")
    backward_objective = detector_cost
    policy_loss = None
    if args.policy_estimator == "score_function":
        policy_loss = losses.get("georoute_score_function_loss")
        if not torch.is_tensor(policy_loss) or policy_loss.ndim != 0 or not bool(torch.isfinite(policy_loss).item()):
            raise FloatingPointError("P0 score-function route lacks a finite detector-derived policy loss")
        backward_objective = detector_cost + policy_loss
    if not bool(torch.isfinite(backward_objective).item()):
        raise FloatingPointError("P0 backward objective is non-finite")
    if score_function_full_graph:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable_parameters, lr=0.0)
        scaler = torch.cuda.amp.GradScaler(
            init_scale=AMP_FULL_GRAPH_FLOOR_SCALE,
            growth_interval=2**30,
        )
        scale_before = float(scaler.get_scale())
        scaler.scale(backward_objective).backward()
        scaler.unscale_(optimizer)
        gradient = _gradient_summary(
            model,
            required_components=required_components,
        )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer_update_succeeded = scale_after >= scale_before
        if not optimizer_update_succeeded:
            raise FloatingPointError("P0 full-graph AMP optimizer update overflowed at floor scale")
    else:
        backward_objective.backward()
        gradient = _gradient_summary(
            model,
            required_components=required_components,
        )
        scale_before = None
        scale_after = None
        optimizer_update_succeeded = None
    if gradient["missing_required_components"]:
        raise RuntimeError("P0 detector-to-router gradient audit failed: " + ", ".join(gradient["missing_required_components"]))
    if args.policy_estimator == "score_function":
        from tools.bata.run_georoute_estimator_kat import _amp_horizon_kat

        score_function_amp_horizon = _amp_horizon_kat(
            device=device,
            tubelets=AMP_PRODUCTION_TUBELETS,
            patch_capacity=source_capacity,
            target_k=int(args.tokens_per_tubelet),
            loss_scale=256.0,
        )
        if score_function_amp_horizon.get("passed") is not True:
            raise FloatingPointError("P0 score-function AMP production-horizon KAT failed")
        score_function_amp_horizon = {
            "status": "PASS_AMP_PRODUCTION_HORIZON",
            **score_function_amp_horizon,
        }
    else:
        score_function_amp_horizon = {
            "status": "NOT_APPLICABLE_NON_SCORE_FUNCTION",
            "executed": False,
        }
    audit = dict(model.backbone.latest_georoute_audit or {})
    if not audit:
        raise RuntimeError("GeoRoute backbone did not emit its native packed audit")
    if score_function_full_graph:
        score_function_full_graph_amp = {
            "status": "PASS_FULL_GRAPH_AMP_OPTIMIZER_UPDATE",
            "executed": True,
            "autocast_dtype": "torch.float16",
            "loss_scale_before": float(scale_before),
            "loss_scale_after": float(scale_after),
            "optimizer": "sgd_lr_zero_overflow_probe",
            "optimizer_update_succeeded": bool(optimizer_update_succeeded),
            "all_required_gradients_finite": bool(gradient["all_required_gradients_finite"]),
            "scout_autocast_enabled": bool(audit.get("scout_autocast_enabled", True)),
            "scout_compute_dtype": str(audit.get("scout_compute_dtype", "")),
            "model_backward_scope": "detector_plus_score_function",
        }
    else:
        score_function_full_graph_amp = {
            "status": "NOT_APPLICABLE_NON_SCORE_FUNCTION",
            "executed": False,
        }
    selected_native_shape = audit.get("selected_native_tubelet_shape")
    if not isinstance(selected_native_shape, list) or len(selected_native_shape) != 7:
        raise RuntimeError("GeoRoute audit did not retain the selected native tubelet shape")
    observed_k = int(selected_native_shape[2])
    if observed_k != int(audit["target_k"]):
        raise RuntimeError("GeoRoute selected native tubelets disagree with exact-K audit")
    if int(audit.get("selected_unique_count_min", -1)) != observed_k or int(audit.get("selected_unique_count_max", -1)) != observed_k:
        raise RuntimeError("GeoRoute P0 independently observed non-unique native selection")
    exact_k = {
        "target_k": int(audit["target_k"]),
        "observed_min": observed_k,
        "observed_max": observed_k,
        "duplicates": int(audit.get("selected_duplicate_count", -1)),
    }
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    packed = audit.get("packed")
    if not isinstance(packed, Mapping):
        raise RuntimeError("GeoRoute P0 audit lacks packed component counters")
    from tools.bata.georoute_storage import storage_capacity_receipt

    storage_receipt = storage_capacity_receipt(
        Path(args.output).resolve().parent,
        cell_count=1,
    )
    runtime_commit = os.environ.get("GEOROUTE_EXPECTED_COMMIT", "").lower()
    if len(runtime_commit) != 40:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        runtime_commit = completed.stdout.strip().lower()
    if len(runtime_commit) != 40:
        raise RuntimeError("P0 could not bind its runtime commit")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id:
        raise RuntimeError("P0 CUDA gate must run inside a Slurm leaf")
    rendezvous_binding = _load_rendezvous_binding(
        runtime_commit=runtime_commit,
        slurm_job_id=slurm_job_id,
    )
    model_state_bytes = sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in model.state_dict().values())
    trainable_parameter_bytes = sum(int(parameter.numel()) * max(4, int(parameter.element_size())) for parameter in model.parameters() if parameter.requires_grad)
    checkpoint_upper_bound = int((2 * model_state_bytes + 2 * trainable_parameter_bytes + 64 * 1024**2) * 1.25)
    torch.cuda.synchronize(device)
    report = {
        "schema_version": GEOROUTE_P0_GATE_SCHEMA,
        "status": "PASS",
        "official_test_opened": False,
        "heavy_backbone_forward_count": int(audit["heavy_backbone_forward_count"]),
        "shared_backbone_instances": int(audit["shared_backbone_instances"]),
        "uses_grid_sample": bool(audit["uses_grid_sample"]),
        "uses_resized_local_crop": bool(audit["uses_resized_local_crop"]),
        "exact_k": exact_k,
        "estimator": {
            "name": str(audit["policy_estimator"]),
            "claim": str(audit["estimator_claim"]),
        },
        "score_function_amp_horizon": score_function_amp_horizon,
        "score_function_full_graph_amp": score_function_full_graph_amp,
        "route_mode": str(audit["route_mode"]),
        "pilot_arm": args.pilot_arm,
        "hybrid_causal_arm": args.hybrid_causal_arm,
        "route_parameters": {
            "context_tokens": int(args.context_tokens),
            "structured_roi_tokens": int(args.structured_roi_tokens),
            "structured_residual_tokens": int(args.structured_residual_tokens),
            "geometry_temporal_shift_tubelets": int(
                args.geometry_temporal_shift_tubelets
            ),
            "roi_fraction": float(args.roi_fraction),
            "policy_temperature": float(cfg.model.backbone.custom.georoute_policy_temperature),
            "score_function_weight": float(cfg.model.backbone.custom.georoute_score_function_weight),
            "score_function_baseline_momentum": float(cfg.model.backbone.custom.georoute_score_function_baseline_momentum),
        },
        "representation": {
            "absolute_position_enabled": bool(audit["absolute_position_enabled"]),
            "absolute_coordinates_enabled": bool(audit["absolute_coordinates_enabled"]),
            "roi_relative_coordinates_enabled": bool(audit["roi_relative_coordinates_enabled"]),
            "geometry_projection_enabled": bool(audit["geometry_projection_enabled"]),
            "geometry_side_channel": bool(audit["geometry_side_channel"]),
            "learned_geometry_enabled": bool(audit["learned_geometry_enabled"]),
            "learned_residual_enabled": bool(audit["learned_residual_enabled"]),
        },
        "memory": memory,
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "gradient": gradient,
        "detector": {
            "training_forward": True,
            "backward_completed": True,
            "output_length": int(audit["output_shape"][-1]),
            "detector_only_loss": float(detector_cost.detach().item()),
            "detector_loss_keys": detector_loss_keys,
            "backward_objective": "detector_only" if policy_loss is None else "detector_only_plus_score_function",
            "score_function_policy_loss": None if policy_loss is None else float(policy_loss.detach().item()),
        },
        "input": {
            "source_shape": list(source.shape),
            "scout_shape": list(scout.shape),
            "source_dtype": str(source.dtype),
            "synthetic": True,
        },
        "source_grid": {
            "height": int(args.height),
            "width": int(args.width),
            "patch_size": 16,
            "grid_height": int(args.height) // 16,
            "grid_width": int(args.width) // 16,
            "patch_capacity": int(source_capacity),
            "boundary_padding": "none",
            "native_support": "floor_complete_patches",
            "ignored_bottom": int(args.height) % 16,
            "ignored_right": int(args.width) % 16,
        },
        "native_route": {
            "selected_native_tubelet_shape": selected_native_shape,
            "output_shape": list(audit["output_shape"]),
            "selected_unique_count_min": int(audit.get("selected_unique_count_min", -1)),
            "selected_unique_count_max": int(audit.get("selected_unique_count_max", -1)),
            "native_packed_invocation_counter_before": int(audit.get("native_packed_invocation_counter_before", -1)),
            "native_packed_invocation_counter_after": int(audit.get("native_packed_invocation_counter_after", -1)),
        },
        "dense_native_reference": audit.get("dense_native_reference"),
        "score_function_detector_binding": audit.get("score_function_detector_binding"),
        "structured_route_audit": {
            "routing_schema": audit.get("routing_schema"),
            "role_counts": audit.get("role_counts"),
            "route_rng": audit.get("route_rng"),
            "branch_log_probabilities": audit.get("branch_log_probabilities"),
            "diagnostic_telemetry": audit.get("diagnostic_telemetry"),
        }
        if args.hybrid_causal_arm is not None
        else None,
        "component_trace": {
            "packed_attention_forward_count": int(packed.get("packed_attention_forward_count", 0)),
            "packed_mlp_forward_count": int(packed.get("packed_mlp_forward_count", 0)),
            "packed_adapter_forward_count": int(packed.get("packed_adapter_forward_count", 0)),
            "dense_adapter_forward_count": int(packed.get("dense_adapter_forward_count", -1)),
            "adapter_execution": packed.get("adapter_execution"),
            "pooling_mode": audit.get("pooling_mode"),
            "measured_latency": False,
            "scope": "mechanical_component_execution_trace",
        },
        "checkpoint_receipt": {
            "checkpoint_count": 0,
            "policy": "p0_no_checkpoint",
        },
        "storage_receipt": storage_receipt,
        "runtime_commit": runtime_commit,
        "slurm_job_id": slurm_job_id,
        "rendezvous_isolation": rendezvous_binding,
        "checkpoint_storage_measurement": {
            "schema_version": "georoute_checkpoint_storage_measurement_v1",
            "runtime_commit": runtime_commit,
            "checkpoint_policy": "final_only",
            "model_state_tensor_bytes": model_state_bytes,
            "trainable_parameter_bytes": trainable_parameter_bytes,
            "checkpoint_upper_bound_bytes": checkpoint_upper_bound,
            "peak_checkpoint_copies_per_cell": 1,
            "auxiliary_upper_bound_bytes_per_cell": 2 * 1024**3,
            "stage_fixed_overhead_bytes": 1024**3,
            "safety_fraction": 0.25,
            "safety_bytes": 16 * 1024**3,
            "measurement_method": ("same_commit_tensor_bytes_plus_ema_plus_adamw_moments_" "plus_serialization_margin"),
        },
        "cuda": {
            "logical_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device),
        },
        "p0_scope": {
            "synthetic_inputs_only": True,
            "full_training": False,
            "official_evaluation": False,
        },
    }
    validate_p0_gate_report(build_p0_gate_report(report))
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    report = build_p0_gate_report(_run_cuda_gate(args))
    validate_p0_gate_report(report)
    _atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
