from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.duca_evidence_io import canonical_sha256, verify_content_sha256
from tools.bata.duca_gate_diagnostics import (
    implemented_uniform_axis_geometry_report,
)


ADMISSION_SCHEMA = "duca_acquisition_admission_v2"
SELECTED_AXIS_CONTRACT = "duca_rime_selected_axis_plugin_v2"
RUNTIME_PRODUCER_SCHEMA = "duca_acquisition_runtime_producer_v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a sequence",
    )
    return value


def _sha(value: Any, label: str) -> str:
    text = str(value)
    _require(_SHA256.fullmatch(text) is not None, f"{label} must be SHA-256")
    return text


def _finite_nonnegative(value: Any, label: str) -> float:
    numeric = float(value)
    _require(
        math.isfinite(numeric) and numeric >= 0.0,
        f"{label} must be finite and non-negative",
    )
    return numeric


def _artifact_binding(value: Any, label: str) -> Mapping[str, Any]:
    binding = _mapping(value, label)
    _require(bool(str(binding.get("path", "")).strip()), f"{label} path is missing")
    _sha(binding.get("sha256"), f"{label} sha256")
    return binding


def validate_duca_acquisition_admission_v2(
    payload: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    require_passed: bool = True,
) -> dict[str, Any]:
    evidence = dict(payload)
    if require_passed:
        _require("content_sha256" in evidence, "passed admission must be content-bound")
    if "content_sha256" in evidence:
        verify_content_sha256(evidence)
    _require(evidence.get("schema") == ADMISSION_SCHEMA, "invalid admission schema")
    status = str(evidence.get("status", ""))
    _require(status in {"passed", "failed"}, "invalid admission status")
    if require_passed:
        _require(status == "passed", "acquisition admission did not pass")
        _require(
            evidence.get("admission_effect") is True,
            "passed admission must have admission effect",
        )

    identity = _mapping(evidence.get("identity"), "identity")
    commit = str(identity.get("git_commit", ""))
    _require(_COMMIT.fullmatch(commit) is not None, "identity commit is not exact")
    if expected_commit is not None:
        _require(commit == str(expected_commit), "admission commit drift")
    _require(identity.get("tracked_tree_clean") is True, "tracked tree is not clean")
    _require(bool(str(identity.get("remote", "")).strip()), "remote is missing")
    _require(bool(str(identity.get("branch", "")).strip()), "branch is missing")
    _require(bool(str(identity.get("repo_root", "")).strip()), "repo_root is missing")
    _require(
        _GIT_OBJECT.fullmatch(str(identity.get("git_tree", ""))) is not None,
        "git_tree must be an exact Git object ID",
    )
    for key in (
        "config_sha256",
        "checkpoint_sha256",
        "data_manifest_sha256",
        "split_assignment_sha256",
    ):
        _sha(identity.get(key), key)

    runtime = _mapping(evidence.get("runtime"), "runtime")
    for key in ("python", "torch", "cuda_runtime", "gpu_name", "driver"):
        _require(bool(str(runtime.get(key, "")).strip()), f"runtime {key} is missing")
    _require(runtime.get("amp_enabled") is True, "AMP runtime is required")
    _require(
        str(runtime.get("amp_dtype")) in {"float16", "bfloat16"},
        "AMP dtype is unsupported",
    )
    _mapping(runtime.get("deterministic_flags"), "deterministic_flags")
    _require(bool(str(runtime.get("slurm_job_id", "")).strip()), "Slurm job ID is missing")

    producer = _mapping(evidence.get("producer"), "producer")
    _require(
        producer.get("schema") == RUNTIME_PRODUCER_SCHEMA
        and producer.get("module")
        == "tools.bata.run_duca_acquisition_runtime_gate_v2"
        and producer.get("finalized_in_runtime_producer") is True
        and producer.get("git_commit") == commit
        and producer.get("git_tree") == identity.get("git_tree")
        and str(producer.get("slurm_job_id")) == str(runtime.get("slurm_job_id")),
        "runtime producer identity is incomplete",
    )
    _require(
        bool(str(producer.get("created_at_utc", "")).strip()),
        "runtime producer creation time is missing",
    )
    _artifact_binding(producer.get("script"), "runtime producer script")
    _artifact_binding(producer.get("launcher"), "runtime producer launcher")

    artifacts = _mapping(evidence.get("artifact_bindings"), "artifact_bindings")
    for key in (
        "code_gate_receipt",
        "selected_actionformer_config",
        "standard_actionformer_config",
        "actionformer_checkpoint",
        "selected_tridet_config",
        "standard_tridet_config",
        "tridet_checkpoint",
        "train_block_list",
        "development_block_list",
        "targets_jsonl",
        "budget_protocol",
        "data_manifest",
        "split_assignment",
        "numeric_calibration",
        "scientific_protocol",
    ):
        _artifact_binding(artifacts.get(key), f"artifact {key}")

    coordinate = _mapping(
        evidence.get("coordinate_contract"),
        "coordinate_contract",
    )
    expected_coordinate = {
        "mode": "selected_axis_plugin",
        "selector_contract": SELECTED_AXIS_CONTRACT,
        "detector_output_coordinate_space": "selected_axis_index",
        "inverse_map_before_official_nms": True,
        "mapping_applied_exactly_once": True,
        "physical_head_enabled": False,
        "gt_remapped_to_selected_axis": True,
        "standard_detector_head_unchanged": True,
    }
    _require(
        all(coordinate.get(key) == value for key, value in expected_coordinate.items()),
        "selected-axis coordinate contract is incomplete",
    )

    execution = _mapping(evidence.get("execution"), "execution")
    roles = [str(value) for value in _sequence(execution.get("window_roles"), "window_roles")]
    vectors = {}
    for key in (
        "requested_k",
        "effective_k",
        "backbone_input_k",
        "active_mask_count",
        "padded_k",
    ):
        vectors[key] = [
            int(value) for value in _sequence(execution.get(key), f"execution {key}")
        ]
    count = len(roles)
    _require(count > 0, "execution evidence is empty")
    _require(
        all(len(values) == count for values in vectors.values()),
        "execution vectors have inconsistent lengths",
    )
    _require(
        {"full_window", "short_window"}.issubset(set(roles)),
        "full and short windows are both required",
    )
    for index in range(count):
        requested = vectors["requested_k"][index]
        effective = vectors["effective_k"][index]
        _require(
            requested >= effective > 0
            and effective
            == vectors["backbone_input_k"][index]
            == vectors["active_mask_count"][index]
            == vectors["padded_k"][index],
            f"execution row {index} violates exact-K/no-padding",
        )
    _sha(execution.get("positions_sha256"), "positions_sha256")
    _sha(execution.get("bucket_order_sha256"), "bucket_order_sha256")

    geometry = _mapping(evidence.get("geometry"), "geometry")
    windows = _sequence(geometry.get("windows"), "geometry windows")
    _require(len(windows) == count, "geometry/execution row count mismatch")
    for index, row in enumerate(windows):
        row = _mapping(row, f"geometry window {index}")
        positions = [
            int(value)
            for value in _sequence(row.get("selected_positions"), "selected_positions")
        ]
        report = implemented_uniform_axis_geometry_report(
            valid_len=int(row.get("valid_len")),
            positions=positions,
        )
        _require(
            row.get("implemented_map") == report,
            f"geometry window {index} does not match the implemented map",
        )
    _finite_nonnegative(
        geometry.get("roundtrip_max_abs_error"),
        "roundtrip_max_abs_error",
    )
    _require(
        geometry.get("mapping_applied_exactly_once") is True,
        "prediction mapping was not applied exactly once",
    )

    restoration = _mapping(
        evidence.get("standard_detector_restoration"),
        "standard_detector_restoration",
    )
    for backend in ("actionformer", "tridet"):
        row = _mapping(restoration.get(backend), f"{backend} restoration")
        _require(row.get("status") == "passed", f"{backend} restoration failed")
        _require(
            row.get("physical_head_enabled") is False
            and row.get("selector_disabled_null_passed") is True
            and row.get("standard_head_state_dict_compatible") is True,
            f"{backend} standard detector was not restored",
        )
        _sha(row.get("standard_config_sha256"), f"{backend} standard config")

    numeric = _mapping(evidence.get("numeric"), "numeric")
    _sha(
        numeric.get("calibration_manifest_sha256"),
        "calibration_manifest_sha256",
    )
    _sha(
        numeric.get("calibration_content_sha256"),
        "calibration_content_sha256",
    )
    _sha(
        numeric.get("runtime_fingerprint_sha256"),
        "runtime_fingerprint_sha256",
    )
    runs = _sequence(numeric.get("amp_null_runs"), "amp_null_runs")
    _require(bool(runs), "AMP null runs are missing")
    for index, run in enumerate(runs):
        row = _mapping(run, f"AMP null run {index}")
        _require(
            row.get("within_frozen_thresholds") is True,
            f"AMP null run {index} exceeded its frozen threshold",
        )
    _sha(numeric.get("state_before_sha256"), "state_before_sha256")
    _sha(numeric.get("state_after_sha256"), "state_after_sha256")
    _require(
        numeric.get("state_before_sha256") == numeric.get("state_after_sha256"),
        "runtime gate mutated model/RNG/debug state",
    )
    replay = _mapping(
        numeric.get("autocast_disabled_non_admission_replay"),
        "autocast-disabled replay",
    )
    _require(
        replay.get("admission_effect") is False,
        "autocast-disabled replay must remain diagnostic-only",
    )

    gates = _mapping(evidence.get("gates"), "gates")
    for key in (
        "structural_gate_passed",
        "numeric_gate_passed",
        "scientific_protocol_preregistered",
    ):
        _require(gates.get(key) is True, f"{key} did not pass")
    _require(
        gates.get("legacy_scalar_loss_equivalence_required") is False,
        "legacy scalar-loss equivalence cannot authorize v2",
    )
    _sha(
        gates.get("scientific_protocol_sha256"),
        "scientific_protocol_sha256",
    )
    _sha(
        gates.get("scientific_protocol_content_sha256"),
        "scientific_protocol_content_sha256",
    )

    scientific = _mapping(evidence.get("scientific_scope"), "scientific_scope")
    _require(
        scientific.get("uses_official_final") is False
        and scientific.get("paper_claim_allowed") is False
        and scientific.get("phase4_submission_enabled") is False
        and scientific.get("official_final_sealed") is True,
        "scientific evidence boundary is not sealed",
    )
    _require(
        bool(str(scientific.get("primary_endpoint", "")).strip()),
        "scientific primary endpoint is missing",
    )
    _finite_nonnegative(
        scientific.get("noninferiority_margin"),
        "noninferiority_margin",
    )
    _require(
        scientific.get("multiplicity_procedure") in {"holm", "closed_testing"},
        "scientific multiplicity procedure is unsupported",
    )

    predecessor = _mapping(
        evidence.get("predecessor_evidence"),
        "predecessor_evidence",
    )
    _require(
        str(predecessor.get("recovery_v6_job")) == "1201417"
        and predecessor.get("historical_status") == "failed_under_v1"
        and predecessor.get("historical_outcome_reclassified") is False,
        "Recovery-v6 immutable history was altered",
    )
    return evidence


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_duca_acquisition_admission_artifacts(
    payload: Mapping[str, Any],
    *,
    expected_commit: str,
    repo_root: str | Path,
    expected_branch: str | None = None,
    expected_remote: str | None = None,
    expected_artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Re-hash every admission dependency in the exact clean checkout."""

    evidence = validate_duca_acquisition_admission_v2(
        payload,
        expected_commit=expected_commit,
        require_passed=True,
    )
    root = Path(repo_root).expanduser().resolve()
    _require(root.is_dir(), f"admission repository is missing: {root}")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            encoding="utf-8",
        ).strip()

    identity = _mapping(evidence["identity"], "identity")
    _require(Path(identity["repo_root"]).resolve() == root, "admission repo_root drift")
    _require(git("rev-parse", "HEAD") == expected_commit, "current Git commit drift")
    _require(
        not git("status", "--porcelain", "--untracked-files=normal"),
        "current Git tree is dirty",
    )
    _require(
        git("rev-parse", "HEAD^{tree}") == identity["git_tree"],
        "current Git tree object drift",
    )
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    remote = git("remote", "get-url", "origin")
    _require(branch == identity["branch"], "current Git branch drift")
    _require(remote == identity["remote"], "current Git remote drift")
    if expected_branch is not None:
        _require(branch == expected_branch, "expected Git branch drift")
    if expected_remote is not None:
        _require(remote == expected_remote, "expected Git remote drift")

    artifacts = _mapping(evidence["artifact_bindings"], "artifact_bindings")
    resolved_bindings: dict[str, tuple[Path, str]] = {}
    for label, raw_binding in artifacts.items():
        binding = _artifact_binding(raw_binding, f"artifact {label}")
        path = Path(str(binding["path"])).expanduser().resolve()
        _require(path.is_file(), f"admission artifact is missing: {label}: {path}")
        actual_sha = _sha256_path(path)
        _require(
            actual_sha == binding["sha256"],
            f"admission artifact SHA-256 drift: {label}",
        )
        resolved_bindings[str(label)] = (path, actual_sha)

    if expected_artifact_paths is not None:
        for label, expected_path in expected_artifact_paths.items():
            _require(label in resolved_bindings, f"expected admission artifact absent: {label}")
            actual_path, _ = resolved_bindings[label]
            _require(
                actual_path == Path(expected_path).expanduser().resolve(),
                f"admission artifact path mismatch: {label}",
            )

    producer = _mapping(evidence["producer"], "producer")
    for label, relative in (
        ("script", Path("tools/bata/run_duca_acquisition_runtime_gate_v2.py")),
        ("launcher", Path("scripts/run_duca_acquisition_admission_v2.sh")),
    ):
        binding = _artifact_binding(producer[label], f"producer {label}")
        expected_path = (root / relative).resolve()
        _require(
            Path(str(binding["path"])).resolve() == expected_path,
            f"runtime producer {label} path drift",
        )
        _require(
            _sha256_path(expected_path) == binding["sha256"],
            f"runtime producer {label} SHA-256 drift",
        )

    config_bundle = _mapping(identity.get("config_bundle"), "config_bundle")
    checkpoint_bundle = _mapping(identity.get("checkpoint_bundle"), "checkpoint_bundle")
    _require(
        canonical_sha256(config_bundle) == identity["config_sha256"],
        "config bundle identity drift",
    )
    _require(
        canonical_sha256(checkpoint_bundle) == identity["checkpoint_sha256"],
        "checkpoint bundle identity drift",
    )
    for backend, prefix in (
        ("actionformer", "actionformer"),
        ("tridet", "tridet"),
    ):
        config_row = _mapping(config_bundle.get(backend), f"{backend} config bundle")
        for role in ("selected", "standard"):
            label = f"{role}_{prefix}_config"
            _require(
                Path(str(config_row[f"{role}_path"])).resolve()
                == resolved_bindings[label][0]
                and config_row[f"{role}_sha256"] == resolved_bindings[label][1],
                f"{backend} {role} config bundle drift",
            )
        checkpoint_row = _mapping(
            checkpoint_bundle.get(backend),
            f"{backend} checkpoint bundle",
        )
        checkpoint_label = f"{prefix}_checkpoint"
        _require(
            Path(str(checkpoint_row["checkpoint_path"])).resolve()
            == resolved_bindings[checkpoint_label][0]
            and checkpoint_row["checkpoint_sha256"]
            == resolved_bindings[checkpoint_label][1],
            f"{backend} checkpoint bundle drift",
        )
    _require(
        resolved_bindings["data_manifest"][1] == identity["data_manifest_sha256"],
        "data manifest identity drift",
    )
    _require(
        resolved_bindings["split_assignment"][1]
        == identity["split_assignment_sha256"],
        "split assignment identity drift",
    )
    _require(
        resolved_bindings["numeric_calibration"][1]
        == evidence["numeric"]["calibration_manifest_sha256"],
        "numeric calibration raw SHA-256 drift",
    )
    _require(
        resolved_bindings["scientific_protocol"][1]
        == evidence["gates"]["scientific_protocol_sha256"],
        "scientific protocol raw SHA-256 drift",
    )
    calibration = json.loads(
        resolved_bindings["numeric_calibration"][0].read_text(encoding="utf-8")
    )
    protocol = json.loads(
        resolved_bindings["scientific_protocol"][0].read_text(encoding="utf-8")
    )
    verify_content_sha256(calibration)
    verify_content_sha256(protocol)
    _require(
        calibration.get("schema") == "duca_numeric_null_calibration_v1"
        and calibration.get("status") == "frozen"
        and calibration.get("git_commit") == expected_commit
        and calibration.get("uses_official_final") is False
        and calibration.get("candidate_performance_observed") is False,
        "numeric calibration identity/scope drift",
    )
    _require(
        protocol.get("schema") == "duca_acquisition_scientific_protocol_v1"
        and protocol.get("status") == "frozen"
        and protocol.get("git_commit") == expected_commit
        and protocol.get("uses_official_final") is False
        and protocol.get("paper_claim_allowed") is False
        and protocol.get("phase4_submission_enabled") is False
        and protocol.get("official_final_sealed") is True,
        "scientific protocol identity/scope drift",
    )
    _require(
        calibration["content_sha256"]
        == evidence["numeric"]["calibration_content_sha256"],
        "numeric calibration content identity drift",
    )
    _require(
        protocol["content_sha256"]
        == evidence["gates"]["scientific_protocol_content_sha256"],
        "scientific protocol content identity drift",
    )
    anchor = _mapping(
        protocol.get("preregistration_anchor"),
        "scientific preregistration anchor",
    )
    for key in ("remote", "branch", "git_commit", "git_tree", "repo_root"):
        _require(
            str(anchor.get(key)) == str(identity.get(key)),
            f"scientific preregistration identity drift: {key}",
        )
    _require(
        anchor.get("candidate_output_root_absent") is True
        and anchor.get("candidate_results_observed") is False,
        "scientific protocol is not pre-candidate",
    )
    margin_binding = _artifact_binding(
        protocol.get("margin_source"),
        "scientific margin source",
    )
    margin_path = Path(str(margin_binding["path"])).expanduser().resolve()
    _require(margin_path.is_file(), f"scientific margin source missing: {margin_path}")
    _require(
        _sha256_path(margin_path) == margin_binding["sha256"],
        "scientific margin source raw SHA-256 drift",
    )
    margin_payload = json.loads(margin_path.read_text(encoding="utf-8"))
    verify_content_sha256(margin_payload)
    _require(
        margin_payload.get("content_sha256") == margin_binding.get("content_sha256"),
        "scientific margin source content SHA-256 drift",
    )
    scope = _mapping(evidence["scientific_scope"], "scientific_scope")
    for key in (
        "primary_endpoint",
        "noninferiority_margin",
        "multiplicity_procedure",
    ):
        _require(
            scope.get(key) == protocol.get(key),
            f"scientific receipt/protocol drift: {key}",
        )
    runtime = _mapping(evidence["runtime"], "runtime")
    runtime_fingerprint = canonical_sha256(
        {
            key: runtime[key]
            for key in (
                "python",
                "torch",
                "cuda_runtime",
                "cudnn",
                "gpu_name",
                "driver",
                "amp_enabled",
                "amp_dtype",
                "deterministic_flags",
            )
        }
    )
    _require(
        runtime_fingerprint == evidence["numeric"]["runtime_fingerprint_sha256"],
        "runtime fingerprint drift",
    )
    _require(
        runtime_fingerprint == calibration.get("runtime_fingerprint_sha256"),
        "numeric calibration/runtime identity drift",
    )
    thresholds = _mapping(calibration.get("thresholds"), "numeric thresholds")
    for run_index, run in enumerate(evidence["numeric"]["amp_null_runs"]):
        errors = _mapping(run.get("metric_errors"), f"AMP null run {run_index} errors")
        derived_pass = bool(errors) and all(
            key in thresholds
            and math.isfinite(float(value))
            and float(value) >= 0.0
            and float(value) <= float(thresholds[key])
            for key, value in errors.items()
        )
        _require(
            derived_pass and run.get("within_frozen_thresholds") is True,
            f"AMP null run {run_index} is not derivably within calibration",
        )
    code_gate_text = resolved_bindings["code_gate_receipt"][0].read_text(
        encoding="utf-8"
    )
    code_gate_fields: dict[str, str] = {}
    for raw_line in code_gate_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        _require(bool(separator) and bool(key), "code-gate receipt is malformed")
        _require(key not in code_gate_fields, f"duplicate code-gate field: {key}")
        code_gate_fields[key] = value
    _require(
        code_gate_fields.get("schema") == "duca_rime_code_gate_v1"
        and code_gate_fields.get("status") == "passed"
        and code_gate_fields.get("commit") == expected_commit
        and bool(code_gate_fields.get("slurm_job_id")),
        "code-gate receipt identity/status drift",
    )
    return evidence
