#!/usr/bin/env python3
"""Validate four PhysTime decode-cross runs from explicit evidence artifacts."""

import argparse
import functools
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from tools.bata.validate_phystime_p0_fullprecision_suite import (
    compare_prediction_decisions,
    load_ground_truth,
    proposal_recall_diagnostics,
)


EXPECTED_RUNS = {
    "selected_online": ("selected_axis", "online"),
    "selected_ema": ("selected_axis", "ema"),
    "physical_online": ("physical_metric", "online"),
    "physical_ema": ("physical_metric", "ema"),
}
AXES = ("uniform_rank_seconds", "physical_time_seconds")
METRIC_EPSILON = 1.0e-12
P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_SUITE_SHA256 = (
    "afb3e300424a57eb590a21129217e040677dc875fdede3be344352dc2bd268e7"
)
P0_GATE_SHA256 = (
    "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
)
P0_DATASET_MANIFEST_SHA256 = (
    "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
)
P0_VIDEOMAE_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)
FATAL_LOG_PATTERNS = (
    r"\btraceback\b",
    r"\bcuda out of memory\b",
    r"\boutofmemoryerror\b",
    r"\boom(?:[- ]|_)?kill(?:ed)?\b",
    r"\bkilled\b",
    r"\bsegmentation fault\b|\bsegfault\b",
    r"\bbus error\b",
    r"\bnccl\b[^\n]{0,120}\berror\b",
    r"\bloss\s*=\s*(?:nan|[-+]?inf)\b",
    r"\bnan gradient\b",
    r"\bnon[- ]finite\b",
    r"\bamp skipped optimizer step\b",
    r"\bfilenotfounderror\b",
    r"\bpytorchstreamwriter\b",
    r"\bdependencyneversatisfied\b",
)
CAPTURE_SCHEMA = "phystime_decode_replay_inputs_v2"
NUMERIC_SEMANTICS_VERSION = "source_score_dtype_legacy_order_v1"
TORCH_TO_NUMPY_DTYPE = {
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate four decode-cross conditions from explicit evidence "
            "paths. This command does not inspect or control a scheduler."
        )
    )
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--p0-suite", required=True)
    parser.add_argument("--p0-gate", required=True)
    parser.add_argument(
        "--completion",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
        help=(
            "Explicit completion artifact. Supply exactly one for each of "
            + ", ".join(EXPECTED_RUNS)
        ),
    )
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Explicit log artifact to scan for fatal markers; repeat as needed.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_completion_specs(specs):
    paths = {}
    for spec in specs:
        require(
            isinstance(spec, str) and "=" in spec,
            f"invalid completion specification: {spec!r}",
        )
        variant, raw_path = spec.split("=", 1)
        variant = variant.strip()
        raw_path = raw_path.strip()
        require(variant in EXPECTED_RUNS, f"unknown completion variant: {variant}")
        require(raw_path, f"empty completion path for {variant}")
        require(variant not in paths, f"duplicate completion variant: {variant}")
        paths[variant] = Path(raw_path).resolve()
    require(
        set(paths) == set(EXPECTED_RUNS),
        "completion variants must be exactly "
        f"{sorted(EXPECTED_RUNS)}; received {sorted(paths)}",
    )
    resolved = [str(path) for path in paths.values()]
    require(
        len(set(resolved)) == len(resolved),
        "the same completion artifact was supplied for multiple variants",
    )
    return paths


def read_json(path, description):
    path = Path(path).resolve()
    require(path.is_file(), f"missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}: {exc}") from exc


@functools.lru_cache(maxsize=None)
def _sha256_resolved(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    return _sha256_resolved(str(Path(path).resolve()))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_artifact_record(record, description, expected_path=None):
    require(isinstance(record, dict), f"{description} record is not a dictionary")
    require(record.get("path"), f"{description} record has no path")
    require(record.get("sha256"), f"{description} record has no SHA-256")
    path = Path(record["path"]).resolve()
    require(path.is_file(), f"missing {description}: {path}")
    if expected_path is not None:
        require(
            path == Path(expected_path).resolve(),
            f"{description} path mismatch",
        )
    digest = sha256_file(path)
    require(digest == record["sha256"], f"{description} SHA-256 mismatch")
    if "size_bytes" in record:
        require(
            int(record["size_bytes"]) == path.stat().st_size,
            f"{description} size mismatch",
        )
    return path


def validate_artifact_map(records, description):
    require(isinstance(records, dict) and records, f"{description} is empty")
    return {
        name: validate_artifact_record(record, f"{description}/{name}")
        for name, record in records.items()
    }


def finite_metrics(payload):
    require(isinstance(payload, dict), "metric payload is not a dictionary")
    metrics = {key: float(value) for key, value in payload.items()}
    require(metrics, "metric dictionary is empty")
    require(
        all(math.isfinite(value) for value in metrics.values()),
        "metric dictionary contains a non-finite value",
    )
    return metrics


def metric_delta(lhs, rhs):
    lhs = finite_metrics(lhs)
    rhs = finite_metrics(rhs)
    require(lhs.keys() == rhs.keys(), "metric keys differ")
    return {
        "fraction": {key: lhs[key] - rhs[key] for key in sorted(lhs)},
        "percentage_points": {
            key: 100.0 * (lhs[key] - rhs[key]) for key in sorted(lhs)
        },
    }


def require_metrics_close(actual, expected, description):
    actual = finite_metrics(actual)
    expected = finite_metrics(expected)
    require(actual.keys() == expected.keys(), f"{description} metric keys differ")
    require(
        all(
            abs(actual[key] - expected[key]) <= METRIC_EPSILON
            for key in actual
        ),
        f"{description} metric values differ",
    )


def metric_linear_combination(*terms):
    keys = None
    normalized = []
    for coefficient, payload in terms:
        values = finite_metrics(payload)
        if keys is None:
            keys = values.keys()
        require(values.keys() == keys, "interaction metric keys differ")
        normalized.append((float(coefficient), values))
    values = {
        key: sum(coefficient * payload[key] for coefficient, payload in normalized)
        for key in sorted(keys)
    }
    return {
        "fraction": values,
        "percentage_points": {
            key: 100.0 * value for key, value in values.items()
        },
    }


def scan_logs(paths):
    paths = [Path(path).resolve() for path in paths]
    require(paths, "at least one explicit log artifact is required")
    require(
        len({str(path) for path in paths}) == len(paths),
        "duplicate explicit log artifact",
    )
    findings = []
    for path in paths:
        require(path.is_file(), f"missing log: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FATAL_LOG_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append({"path": str(path), "pattern": pattern})
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[PhysTime") and " ERROR:" in stripped:
                findings.append({"path": str(path), "marker": stripped[:200]})
    require(not findings, f"fatal log markers found: {findings[:5]}")
    return {
        "files_scanned": len(paths),
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "fatal_findings": findings,
    }


def _identity(runtime_commit, runtime_tree, source_commit, source_tree):
    values = (
        str(runtime_commit),
        str(runtime_tree),
        str(source_commit),
        str(source_tree),
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{40}", value) for value in values),
        "runtime/source identity is not a complete Git commit/tree tuple",
    )
    return values


def _condition_prefix(arm):
    return "selected" if arm == "selected_axis" else "physical"


def _capture_observation_contract(capture):
    array_contract = capture["array_contract"]
    return {
        "observation_sequence_sha256": capture["observation_sequence_sha256"],
        "uniform_axis_sha256": array_contract["uniform_axis_sec"][
            "canonical_sha256"
        ],
        "physical_axis_sha256": array_contract["physical_axis_sec"][
            "canonical_sha256"
        ],
        "base_mask_sha256": array_contract["base_mask"]["canonical_sha256"],
        "native_mask_sha256": array_contract["native_mask"]["canonical_sha256"],
        "base_points_sha256": array_contract["base_points"]["canonical_sha256"],
        "class_map": capture["class_map"],
        "window_count": capture["window_count"],
        "candidate_count": capture["candidate_count"],
        "native_token_count": capture["native_token_count"],
    }


def validate_numeric_precision(numeric_precision, capture, variant):
    require(
        capture.get("schema_version") == CAPTURE_SCHEMA,
        f"{variant} capture schema mismatch",
    )
    require(
        capture.get("numeric_semantics_version") == NUMERIC_SEMANTICS_VERSION,
        f"{variant} capture numeric semantics mismatch",
    )
    source_dtype = capture.get("source_tensor_dtypes", {}).get("cls_scores")
    require(
        source_dtype in TORCH_TO_NUMPY_DTYPE,
        f"{variant} capture score source dtype is unsupported",
    )
    score_contract = capture.get("array_contract", {}).get("cls_scores", {})
    stored_dtype = score_contract.get("stored_numpy_dtype")
    require(
        stored_dtype == TORCH_TO_NUMPY_DTYPE[source_dtype]
        and score_contract.get("dtype") == stored_dtype
        and score_contract.get("semantic_role") == "ranking_scores"
        and score_contract.get("ordering_sensitive") is True
        and score_contract.get("source_torch_dtype") == source_dtype
        and score_contract.get("replay_torch_dtype") == source_dtype
        and score_contract.get("allowed_casts_before_topk") == [],
        f"{variant} source score dtype was widened or its ordering contract changed",
    )
    require(
        numeric_precision.get("source_amp_enabled")
        == capture.get("source_amp_enabled")
        and numeric_precision.get("source_tensor_dtypes")
        == capture.get("source_tensor_dtypes")
        and numeric_precision.get("numeric_semantics_version")
        == capture.get("numeric_semantics_version")
        and numeric_precision.get("score_sort_dtype") == stored_dtype
        and numeric_precision.get("score_sort_device") == "cpu"
        and numeric_precision.get("geometry_compute_dtype") == "float32"
        and numeric_precision.get("geometry_compute_device") == "cpu",
        f"{variant} numeric precision provenance mismatch",
    )


def _validate_preflight_and_gate(preflight_path, gate_path):
    preflight_path = Path(preflight_path).resolve()
    gate_path = Path(gate_path).resolve()
    preflight = read_json(preflight_path, "decode-cross preflight")
    gate = read_json(gate_path, "decode-cross gate")
    require(
        preflight.get("schema_version") == "phystime_decode_cross_preflight_v1"
        and preflight.get("validation_pass") is True
        and preflight.get("runtime", {}).get("clean") is True,
        "decode-cross preflight did not pass",
    )
    require(
        gate.get("schema_version") == "phystime_decode_cross_real_gate_v1"
        and gate.get("gate_pass") is True
        and gate.get("new_training") is False
        and int(gate.get("frozen_epoch", -1)) == 59
        and gate.get("runtime", {}).get("tree_clean") is True,
        "decode-cross real gate did not pass",
    )
    validate_artifact_record(
        gate.get("preflight", {}),
        "gate preflight",
        expected_path=preflight_path,
    )
    runtime = gate["runtime"]
    source = gate["source"]
    identity = _identity(
        runtime["commit"],
        runtime["git_tree"],
        source["commit"],
        source["git_tree"],
    )
    require(
        preflight["runtime"]["commit"] == identity[0]
        and preflight["runtime"]["git_tree"] == identity[1]
        and preflight["source"]["commit"] == identity[2]
        and preflight["source"]["git_tree"] == identity[3],
        "preflight and gate snapshot identities differ",
    )
    real_windows = gate.get("real_windows", {})
    require(
        set(real_windows) == set(EXPECTED_RUNS),
        "decode-cross gate does not contain exactly four conditions",
    )
    require(
        gate.get("all_native_direct_exact_equivalence") is True,
        "decode-cross gate native/direct parity failed",
    )
    for variant, (arm, weights) in EXPECTED_RUNS.items():
        record = real_windows[variant]
        require(
            record.get("arm") == arm
            and record.get("weights_source") == weights
            and record.get("native_direct_exact_equivalence") is True
            and record.get("raw_tensors_immutable") is True,
            f"{variant} gate condition failed",
        )
    return preflight, gate, identity


def _validate_p0(
    p0_suite_path,
    p0_gate_path,
    gate,
    identity,
    *,
    expected_suite_sha256,
    expected_gate_sha256,
):
    p0_suite_path = Path(p0_suite_path).resolve()
    p0_gate_path = Path(p0_gate_path).resolve()
    p0_suite = read_json(p0_suite_path, "P0 suite")
    p0_gate = read_json(p0_gate_path, "P0 gate")
    require(
        sha256_file(p0_suite_path) == expected_suite_sha256,
        "P0 suite differs from the reviewed artifact",
    )
    require(
        sha256_file(p0_gate_path) == expected_gate_sha256,
        "P0 gate differs from the reviewed artifact",
    )
    require(
        p0_suite.get("schema_version")
        == "phystime_p0_fullprecision_suite_completion_v1"
        and p0_suite.get("validation_pass") is True
        and p0_suite.get("runtime_commit") == P0_RUNTIME_COMMIT
        and p0_suite.get("runtime_tree") == P0_RUNTIME_TREE
        and p0_suite.get("source_commit") == identity[2]
        and p0_suite.get("source_tree") == identity[3],
        "P0 suite provenance/schema did not pass",
    )
    validate_artifact_record(
        p0_suite.get("gate", {}),
        "P0 suite gate",
        expected_path=p0_gate_path,
    )
    require(
        p0_gate.get("schema_version") == "phystime_p0_fullprecision_gate_v1"
        and p0_gate.get("gate_pass") is True
        and p0_gate.get("runtime", {}).get("commit") == P0_RUNTIME_COMMIT
        and p0_gate.get("runtime", {}).get("git_tree") == P0_RUNTIME_TREE
        and p0_gate["runtime"]["dataset_manifest_sha256"]
        == P0_DATASET_MANIFEST_SHA256
        and p0_gate["runtime"]["videomae_checkpoint_sha256"]
        == P0_VIDEOMAE_SHA256,
        "P0 gate provenance/schema did not pass",
    )
    validate_artifact_record(
        gate.get("p0", {}).get("suite", {}),
        "decode gate P0 suite",
        expected_path=p0_suite_path,
    )
    validate_artifact_record(
        gate.get("p0", {}).get("gate", {}),
        "decode gate P0 gate",
        expected_path=p0_gate_path,
    )
    return p0_suite, p0_gate


def _validate_config_and_checkpoint_binding(
    *,
    variant,
    arm,
    weights,
    completion_artifacts,
    manifest,
    preflight,
    gate,
):
    preflight_config = preflight["configs"][arm]
    gate_config = gate["configs"][arm]
    validate_artifact_record(preflight_config, f"{arm} preflight config")
    for key in (
        "canonical_config_sha256",
        "coordinate_modes",
        "inference_semantic_sha256",
        "p0_base_inference_semantic_sha256",
        "dataset_bindings",
    ):
        require(
            gate_config.get(key) == preflight_config.get(key),
            f"{variant} config binding differs between preflight and gate",
        )
    require(
        Path(manifest["config"]).resolve()
        == Path(preflight_config["path"]).resolve(),
        f"{variant} run manifest config path mismatch",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", manifest["effective_config_sha256"]),
        f"{variant} effective config SHA-256 is invalid",
    )

    gate_checkpoint = gate["checkpoints"][arm]
    preflight_checkpoint = preflight["checkpoints"][arm]
    gate_checkpoint_path = validate_artifact_record(
        gate_checkpoint,
        f"{arm} frozen checkpoint",
    )
    preflight_checkpoint_path = validate_artifact_record(
        preflight_checkpoint,
        f"{arm} preflight frozen checkpoint",
    )
    require(
        gate_checkpoint_path == preflight_checkpoint_path
        and gate_checkpoint["sha256"] == preflight_checkpoint["sha256"],
        f"{variant} checkpoint binding differs between preflight and gate",
    )
    checkpoint_path = gate_checkpoint_path
    validate_artifact_record(
        completion_artifacts["checkpoint"],
        f"{variant} completion checkpoint",
        expected_path=checkpoint_path,
    )
    require(
        Path(manifest["checkpoint"]).resolve() == checkpoint_path
        and manifest["checkpoint_sha256"] == gate_checkpoint["sha256"],
        f"{variant} manifest checkpoint binding mismatch",
    )
    state_key = "ema_state_dict_sha256" if weights == "ema" else "online_state_dict_sha256"
    require(
        manifest["checkpoint_state_dict_sha256"] == gate_checkpoint[state_key]
        and gate["real_windows"][variant]["checkpoint_state_dict_sha256"]
        == gate_checkpoint[state_key],
        f"{variant} checkpoint state-dict identity mismatch",
    )


def _validate_condition(
    *,
    variant,
    completion_path,
    preflight_path,
    gate_path,
    p0_gate_path,
    preflight,
    gate,
    p0_suite,
    identity,
):
    arm, weights = EXPECTED_RUNS[variant]
    completion_path = Path(completion_path).resolve()
    completion = read_json(completion_path, f"{variant} completion")
    require(
        completion.get("schema_version")
        == "phystime_decode_cross_completion_v1"
        and completion.get("validation_pass") is True
        and completion.get("status") == "tested",
        f"{variant} completion did not pass",
    )
    require(
        completion.get("arm") == arm
        and completion.get("weights_source") == weights
        and completion.get("new_training") is False
        and int(completion.get("evaluation_epoch", -1)) == 59
        and completion.get("same_frozen_raw_tensors_for_both_axes") is True
        and completion.get("native_direct_exact_equivalence") is True
        and completion.get("reviewed_p0_direct_exact_equivalence") is True
        and completion.get("fatal_log_findings") == [],
        f"{variant} frozen/parity/fatal-log contract mismatch",
    )
    require(
        Path(completion["run_dir"]).resolve() == completion_path.parent,
        f"{variant} completion path is not bound to its run directory",
    )
    require(
        _identity(
            completion["runtime_commit"],
            completion["runtime_tree"],
            completion["source_commit"],
            completion["source_tree"],
        )
        == identity,
        f"{variant} snapshot identity mismatch",
    )

    artifact_paths = validate_artifact_map(
        completion.get("artifacts", {}),
        f"{variant} artifacts",
    )
    required_artifacts = {
        "run_manifest",
        "decode_cross_gate",
        "preflight_manifest",
        "runtime_preflight_manifest",
        "capture_manifest",
        "capture_npz",
        "producer_completion",
        "direct_pre_cross",
        "direct_result",
        "direct_metrics",
        "checkpoint",
        "source_completion",
        "source_manifest",
        "p0_completion",
    }
    require(
        set(artifact_paths) == required_artifacts,
        f"{variant} completion artifact set mismatch",
    )
    require(
        artifact_paths["decode_cross_gate"] == Path(gate_path).resolve()
        and artifact_paths["preflight_manifest"] == Path(preflight_path).resolve(),
        f"{variant} gate/preflight artifact binding mismatch",
    )

    manifest = read_json(artifact_paths["run_manifest"], f"{variant} run manifest")
    require(
        manifest.get("schema_version") == "phystime_decode_cross_run_manifest_v1"
        and manifest.get("arm") == arm
        and manifest.get("weights_source") == weights
        and manifest.get("new_training") is False
        and int(manifest.get("evaluation_epoch", -1)) == 59
        and (
            manifest["runtime_commit"],
            manifest["runtime_tree"],
            manifest["source_commit"],
            manifest["source_tree"],
        )
        == identity,
        f"{variant} run manifest mismatch",
    )
    require(
        Path(manifest["gate"]).resolve() == Path(gate_path).resolve()
        and manifest["gate_sha256"] == sha256_file(gate_path)
        and Path(manifest["preflight_manifest"]).resolve()
        == Path(preflight_path).resolve()
        and manifest["preflight_manifest_sha256"] == sha256_file(preflight_path)
        and Path(manifest["runtime_preflight_manifest"]).resolve()
        == artifact_paths["runtime_preflight_manifest"]
        and manifest["runtime_preflight_manifest_sha256"]
        == sha256_file(artifact_paths["runtime_preflight_manifest"])
        == sha256_file(preflight_path),
        f"{variant} run manifest evidence binding mismatch",
    )
    _validate_config_and_checkpoint_binding(
        variant=variant,
        arm=arm,
        weights=weights,
        completion_artifacts=completion["artifacts"],
        manifest=manifest,
        preflight=preflight,
        gate=gate,
    )

    capture = read_json(
        artifact_paths["capture_manifest"],
        f"{variant} capture manifest",
    )
    require(
        capture.get("evaluation_epoch") == 59
        and capture.get("new_training") is False
        and capture.get("weights_source") == weights
        and capture.get("runtime", {}).get("commit") == identity[0]
        and capture.get("runtime", {}).get("git_tree") == identity[1]
        and capture.get("runtime", {}).get("effective_config_sha256")
        == manifest["effective_config_sha256"]
        and capture.get("source", {}).get("commit") == identity[2]
        and capture.get("source", {}).get("git_tree") == identity[3]
        and Path(capture.get("checkpoint", {}).get("path", "")).resolve()
        == artifact_paths["checkpoint"]
        and capture.get("checkpoint", {}).get("sha256")
        == completion["artifacts"]["checkpoint"]["sha256"],
        f"{variant} capture identity/config/checkpoint mismatch",
    )
    validate_numeric_precision(completion.get("numeric_precision", {}), capture, variant)
    observation_contract = _capture_observation_contract(capture)
    native_axis = (
        "uniform_rank_seconds"
        if arm == "selected_axis"
        else "physical_time_seconds"
    )
    require(
        completion.get("native_axis") == native_axis
        and capture.get("expected_native_coordinate_mode") == native_axis,
        f"{variant} native axis mismatch",
    )

    p0_completion = read_json(
        artifact_paths["p0_completion"],
        f"{variant} P0 completion",
    )
    p0_suite_record = p0_suite["completion_artifacts"][variant]
    validate_artifact_record(
        p0_suite_record,
        f"{variant} reviewed P0 completion",
        expected_path=artifact_paths["p0_completion"],
    )
    require(
        p0_suite_record["sha256"]
        == completion["artifacts"]["p0_completion"]["sha256"]
        and p0_completion.get("schema_version")
        == "phystime_p0_fullprecision_completion_v2"
        and p0_completion.get("validation_pass") is True
        and p0_completion.get("runtime_commit") == P0_RUNTIME_COMMIT
        and p0_completion.get("runtime_tree") == P0_RUNTIME_TREE
        and p0_completion.get("source_commit") == identity[2]
        and p0_completion.get("source_tree") == identity[3]
        and p0_completion.get("arm") == arm
        and p0_completion.get("weights_source") == weights
        and p0_completion["artifacts"]["checkpoint"]["sha256"]
        == completion["artifacts"]["checkpoint"]["sha256"],
        f"{variant} reviewed P0 provenance mismatch",
    )
    validate_artifact_record(
        p0_completion["artifacts"]["gate"],
        f"{variant} P0 gate",
        expected_path=p0_gate_path,
    )

    require(
        set(completion.get("mode_artifacts", {})) == set(AXES)
        and set(completion.get("mode_metrics", {})) == set(AXES),
        f"{variant} dual-axis result set is incomplete",
    )
    metrics = {}
    predictions = {}
    for axis_name in AXES:
        mode_paths = validate_artifact_map(
            completion["mode_artifacts"][axis_name],
            f"{variant}/{axis_name} artifacts",
        )
        require(
            set(mode_paths)
            == {
                "mode_report",
                "decoded_candidates",
                "pre_cross",
                "result",
                "metrics",
                "post_processing_audit",
            },
            f"{variant}/{axis_name} mode artifact set mismatch",
        )
        metrics[axis_name] = finite_metrics(
            completion["mode_metrics"][axis_name]
        )
        result = read_json(mode_paths["result"], f"{variant}/{axis_name} result")
        require(
            int(result.get("evaluation_epoch", -1)) == 59,
            f"{variant}/{axis_name} result epoch mismatch",
        )
        predictions[axis_name] = result
    require_metrics_close(
        completion["physical_minus_uniform_fraction"],
        metric_delta(
            metrics["physical_time_seconds"],
            metrics["uniform_rank_seconds"],
        )["fraction"],
        f"{variant} physical-minus-uniform delta",
    )
    return {
        "completion": completion,
        "metrics": metrics,
        "predictions": predictions,
        "observation_contract": observation_contract,
        "capture_contract": {
            **observation_contract,
            "native_coordinate_mode": capture[
                "expected_native_coordinate_mode"
            ],
            "window_sequence_sha256": capture["window_sequence_sha256"],
        },
        "numeric_precision": completion["numeric_precision"],
        "artifact": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
            "size_bytes": completion_path.stat().st_size,
        },
    }


def validate_suite(
    *,
    preflight_path,
    gate_path,
    p0_suite_path,
    p0_gate_path,
    completion_paths,
    log_paths,
    expected_p0_suite_sha256=P0_SUITE_SHA256,
    expected_p0_gate_sha256=P0_GATE_SHA256,
):
    require(
        set(completion_paths) == set(EXPECTED_RUNS),
        "completion path mapping does not contain exactly four conditions",
    )
    preflight, gate, identity = _validate_preflight_and_gate(
        preflight_path,
        gate_path,
    )
    p0_suite, p0_gate = _validate_p0(
        p0_suite_path,
        p0_gate_path,
        gate,
        identity,
        expected_suite_sha256=expected_p0_suite_sha256,
        expected_gate_sha256=expected_p0_gate_sha256,
    )

    conditions = {}
    shared_observation_contract = None
    for variant in EXPECTED_RUNS:
        condition = _validate_condition(
            variant=variant,
            completion_path=completion_paths[variant],
            preflight_path=preflight_path,
            gate_path=gate_path,
            p0_gate_path=p0_gate_path,
            preflight=preflight,
            gate=gate,
            p0_suite=p0_suite,
            identity=identity,
        )
        if shared_observation_contract is None:
            shared_observation_contract = condition["observation_contract"]
        require(
            condition["observation_contract"] == shared_observation_contract,
            f"{variant} sparse observation/time-axis contract differs",
        )
        conditions[variant] = condition

    metrics = {
        variant: condition["metrics"]
        for variant, condition in conditions.items()
    }
    predictions = {
        variant: condition["predictions"]
        for variant, condition in conditions.items()
    }
    ground_truth, evaluation_contract = load_ground_truth(p0_gate)

    within_checkpoint_decode = {
        variant: metric_delta(
            metrics[variant]["physical_time_seconds"],
            metrics[variant]["uniform_rank_seconds"],
        )
        for variant in EXPECTED_RUNS
    }
    cross_checkpoint_descriptive_difference = {
        weights: {
            axis_name: metric_delta(
                metrics[f"physical_{weights}"][axis_name],
                metrics[f"selected_{weights}"][axis_name],
            )
            for axis_name in AXES
        }
        for weights in ("online", "ema")
    }
    descriptive_difference_in_differences = {
        weights: metric_linear_combination(
            (1.0, metrics[f"physical_{weights}"]["physical_time_seconds"]),
            (-1.0, metrics[f"physical_{weights}"]["uniform_rank_seconds"]),
            (-1.0, metrics[f"selected_{weights}"]["physical_time_seconds"]),
            (1.0, metrics[f"selected_{weights}"]["uniform_rank_seconds"]),
        )
        for weights in ("online", "ema")
    }
    weight_source_effect = {
        arm: {
            axis_name: metric_delta(
                metrics[f"{arm}_ema"][axis_name],
                metrics[f"{arm}_online"][axis_name],
            )
            for axis_name in AXES
        }
        for arm in ("selected", "physical")
    }
    within_checkpoint_decisions = {
        variant: compare_prediction_decisions(
            predictions[variant]["physical_time_seconds"],
            predictions[variant]["uniform_rank_seconds"],
        )
        for variant in EXPECTED_RUNS
    }
    cross_checkpoint_decisions = {
        weights: {
            axis_name: compare_prediction_decisions(
                predictions[f"physical_{weights}"][axis_name],
                predictions[f"selected_{weights}"][axis_name],
            )
            for axis_name in AXES
        }
        for weights in ("online", "ema")
    }
    weight_source_decisions = {
        arm: {
            axis_name: compare_prediction_decisions(
                predictions[f"{arm}_ema"][axis_name],
                predictions[f"{arm}_online"][axis_name],
            )
            for axis_name in AXES
        }
        for arm in ("selected", "physical")
    }
    final_detection_oracle_recall = {
        variant: {
            axis_name: proposal_recall_diagnostics(
                predictions[variant][axis_name],
                ground_truth,
            )
            for axis_name in AXES
        }
        for variant in EXPECTED_RUNS
    }
    raw_rows = [
        {
            "variant": variant,
            "train_axis": (
                "uniform_rank_seconds"
                if arm == "selected_axis"
                else "physical_time_seconds"
            ),
            "weights_source": weights,
            "decode_axis": axis_name,
            "metrics": metrics[variant][axis_name],
        }
        for variant, (arm, weights) in EXPECTED_RUNS.items()
        for axis_name in AXES
    ]
    log_scan = scan_logs(log_paths)

    return {
        "schema_version": "phystime_decode_cross_evidence_suite_completion_v1",
        "validation_pass": True,
        "status": "tested",
        "completed_at_unix": time.time(),
        "evidence_mode": "explicit_artifact_paths_v1",
        "new_training": False,
        "frozen_epoch": 59,
        "runtime_commit": identity[0],
        "runtime_tree": identity[1],
        "source_commit": identity[2],
        "source_tree": identity[3],
        "evidence_inputs": {
            "preflight": {
                "path": str(Path(preflight_path).resolve()),
                "sha256": sha256_file(preflight_path),
            },
            "decode_cross_gate": {
                "path": str(Path(gate_path).resolve()),
                "sha256": sha256_file(gate_path),
            },
            "p0_suite": {
                "path": str(Path(p0_suite_path).resolve()),
                "sha256": sha256_file(p0_suite_path),
            },
            "p0_gate": {
                "path": str(Path(p0_gate_path).resolve()),
                "sha256": sha256_file(p0_gate_path),
            },
            "completions": {
                variant: conditions[variant]["artifact"]
                for variant in EXPECTED_RUNS
            },
        },
        "evaluation_contract": evaluation_contract,
        "ground_truth_count": len(ground_truth),
        "log_scan": log_scan,
        "shared_observation_contract": shared_observation_contract,
        "capture_contracts": {
            variant: conditions[variant]["capture_contract"]
            for variant in EXPECTED_RUNS
        },
        "numeric_precision_contracts": {
            variant: conditions[variant]["numeric_precision"]
            for variant in EXPECTED_RUNS
        },
        "raw_metric_rows": raw_rows,
        "within_checkpoint_physical_decode_minus_uniform_decode": (
            within_checkpoint_decode
        ),
        "fixed_decode_cross_checkpoint_descriptive_difference": (
            cross_checkpoint_descriptive_difference
        ),
        "descriptive_difference_in_differences": (
            descriptive_difference_in_differences
        ),
        "weight_source_ema_minus_online": weight_source_effect,
        "within_checkpoint_decision_diagnostics": within_checkpoint_decisions,
        "cross_checkpoint_decision_diagnostics": cross_checkpoint_decisions,
        "weight_source_decision_diagnostics": weight_source_decisions,
        "final_detection_oracle_recall_by_duration_and_iou": (
            final_detection_oracle_recall
        ),
        "claim_boundary": (
            "This frozen single-seed THUMOS replay isolates inference decode "
            "axis effects from one raw tensor artifact per checkpoint. It uses "
            "the production post-processing/evaluator semantics and explicit "
            "evidence artifacts, not scheduler or submission-state provenance. "
            "Cross-checkpoint differences are descriptive, not causal training "
            "effects. The result does not establish assignment causality, "
            "multi-seed robustness, compute savings, or a paper-ready claim."
        ),
    }


def main():
    args = parse_args()
    completion_paths = parse_completion_specs(args.completion)
    completion = validate_suite(
        preflight_path=args.preflight,
        gate_path=args.gate,
        p0_suite_path=args.p0_suite,
        p0_gate_path=args.p0_gate,
        completion_paths=completion_paths,
        log_paths=args.log,
    )
    atomic_write_json(args.output, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
