from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from tools.bata.analyze_duca_selection_quality import (
    SUMMARY_SCHEMA_VERSION,
    analyze_jsonl,
)
from tools.bata.create_duca_frontend_split import validate_split_manifest
from tools.bata.duca_p0_evaluation import canonical_sha256
from tools.bata.finalize_duca_r0_boundary_burst import revalidate_r0_summary
from tools.bata.select_duca_frontend_checkpoint import sha256_file


SCHEMA = "duca_boundary_burst_frontend_decision_v1"
FAMILY_MANIFEST_SCHEMA = "duca_boundary_burst_family_routing_manifest_v1"
FULL_MODEL_GATE_SCHEMA = "duca_boundary_burst_full_model_gate_v1"
P0_REAL_GATE_SCHEMA = "duca_frontend_p0_real_cuda_gate_v1"
P0_ASFORMER_CONSUMER_SCHEMA = "duca_p0_training_asformer_consumer_v1"
FULL_MODEL_ARTIFACT_SCHEMA = (
    "duca_protected_e2e_exact_full_model_gradient_gate_v1"
)
P0_ANALYZER_BOOTSTRAP_SAMPLES = 2000
P0_ANALYZER_RANDOM_SEED = 3407
P0_ANALYZER_REPRESENTATIVE_PER_STRATUM = 2
UNIFORM_OFFICIAL_VARIANT = "two_stage_exact_uniform"
UNIFORM_OFFICIAL_CONFIG = (
    "configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
)
GAUSSIAN_P0_VARIANT = "gaussian_matched"
GAUSSIAN_OFFICIAL_VARIANT = "gaussian_matched_g0"
GAUSSIAN_OFFICIAL_CONFIG = (
    "configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
)
R0_PROJECTED_FAMILY_ROUTES = {
    "R2Q3_privileged_boundary_burst": {
        "p0_variant": "burst_r2q3",
        "p0_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_frontend_pretrain_fixed384.py"
        ),
        "official60_variant": "boundary_burst_r2q3_g0",
        "official60_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
        ),
    },
    "R4Q5_privileged_boundary_burst": {
        "p0_variant": "burst_r4q5",
        "p0_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py"
        ),
        "official60_variant": "boundary_burst_r4q5_g0",
        "official60_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
        ),
    },
}
VARIANT_SPECS = {
    GAUSSIAN_P0_VARIANT: None,
    "burst_r2q3": "r2q3",
    "burst_r4q5": "r4q5",
}


def _atomic_write_json(
    output: Path,
    payload: Mapping[str, Any],
    *,
    require_absent: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if require_absent and output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if require_absent and output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        if os.name != "nt":
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _family_routing_contract(selected_family: Any) -> dict[str, Any]:
    selected = str(selected_family)
    if selected not in R0_PROJECTED_FAMILY_ROUTES:
        raise RuntimeError(f"unsupported R0 projected family: {selected}")
    selected_route = R0_PROJECTED_FAMILY_ROUTES[selected]
    alternate_routes = [
        route
        for family, route in R0_PROJECTED_FAMILY_ROUTES.items()
        if family != selected
    ]
    return {
        "schema": "duca_boundary_burst_r0_family_routing_v1",
        "selected_weakest_projected_family": selected,
        "selected_p0_variant": selected_route["p0_variant"],
        "selected_p0_config": selected_route["p0_config"],
        "selected_official60_variant": selected_route["official60_variant"],
        "selected_official60_config": selected_route["official60_config"],
        "required_p0_variants": [selected_route["p0_variant"]],
        "diagnostic_p0_variants": [
            GAUSSIAN_P0_VARIANT,
            *[route["p0_variant"] for route in alternate_routes],
        ],
        "required_official60_variants": [
            UNIFORM_OFFICIAL_VARIANT,
            selected_route["official60_variant"],
        ],
        "diagnostic_official60_variants": [
            GAUSSIAN_OFFICIAL_VARIANT,
            *[route["official60_variant"] for route in alternate_routes],
        ],
        "uniform_official60_config": UNIFORM_OFFICIAL_CONFIG,
        "gaussian_official60_config": GAUSSIAN_OFFICIAL_CONFIG,
        "projected_family_mappings": {
            family: dict(route)
            for family, route in R0_PROJECTED_FAMILY_ROUTES.items()
        },
        "simple_delta_role": "no_training_same_feasible_control",
    }


def _first_mismatch(
    expected: Any,
    observed: Any,
    *,
    path: str = "summary",
) -> str | None:
    """Return the first recursive difference so evidence drift is actionable."""

    if type(expected) is not type(observed):
        return f"{path} (type {type(expected).__name__} != {type(observed).__name__})"
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return f"{path} (keys differ)"
        for key in sorted(expected, key=str):
            mismatch = _first_mismatch(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path} (length {len(expected)} != {len(observed)})"
        for index, (left, right) in enumerate(zip(expected, observed)):
            mismatch = _first_mismatch(left, right, path=f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == observed else path


def _recompute_selection_summary(
    *,
    records_path: Path,
    records_sha256: str,
    summary_path: Path,
    summary_sha256: str,
) -> dict[str, Any]:
    """Accept only a summary regenerated from the sealed production records."""

    if not records_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("candidate records JSONL is empty")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("selection-quality summary is not a mapping")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("unexpected selection-quality summary schema")
    try:
        summary_records_path = Path(str(summary["records_jsonl"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("selection-quality summary lacks records identity") from exc
    if summary_records_path != records_path:
        raise RuntimeError("selection-quality summary records identity drift")

    # This is deliberately the same production analyzer and fixed P0 invocation.
    with tempfile.TemporaryDirectory(prefix="duca-p0-summary-reanalysis-") as output_dir:
        recomputed = analyze_jsonl(
            records_jsonl=records_path,
            output_dir=output_dir,
            bootstrap_samples=P0_ANALYZER_BOOTSTRAP_SAMPLES,
            random_seed=P0_ANALYZER_RANDOM_SEED,
            representative_per_stratum=P0_ANALYZER_REPRESENTATIVE_PER_STRATUM,
        )
    if sha256_file(records_path) != str(records_sha256):
        raise RuntimeError("candidate records drifted during production reanalysis")
    if sha256_file(summary_path) != str(summary_sha256):
        raise RuntimeError("candidate summary drifted during production reanalysis")
    mismatch = _first_mismatch(summary, recomputed)
    if mismatch is not None:
        raise RuntimeError(
            f"selection-quality summary disagrees with production reanalysis at {mismatch}"
        )
    return dict(summary)


def validate_p0_real_gate(
    *,
    gate_path: str | Path,
    gate_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Reopen the real P0 CUDA gate before it can authorize any consumer."""

    path = _verified_file(gate_path, gate_sha256, label="P0 real gate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    git_binding = payload.get("git_binding") if isinstance(payload, Mapping) else None
    final_git_binding = (
        payload.get("final_git_binding") if isinstance(payload, Mapping) else None
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != P0_REAL_GATE_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or not isinstance(git_binding, Mapping)
        or not isinstance(final_git_binding, Mapping)
        or git_binding.get("git_commit") != expected_commit
        or final_git_binding.get("git_commit") != expected_commit
    ):
        raise RuntimeError("P0 real gate contract drift")
    assets = payload.get("assets")
    if not isinstance(assets, Mapping):
        raise RuntimeError("P0 real gate asset binding is missing")
    official_asformer = _validate_official_asformer_source(
        assets.get("official_asformer_source"),
        label="P0 real gate official ASFormer source",
    )
    adatad_pretrain_binding = assets.get("videomae_checkpoint")
    if not isinstance(adatad_pretrain_binding, Mapping):
        raise RuntimeError("P0 real gate AdaTAD pretrain binding is missing")
    adatad_pretrain = _verified_file(
        adatad_pretrain_binding.get("path", ""),
        str(adatad_pretrain_binding.get("sha256", "")),
        label="P0 real gate AdaTAD pretrain",
    )
    config = _verified_file(
        payload.get("config_path", ""),
        str(payload.get("config_sha256", "")),
        label="P0 real gate config",
    )
    if sha256_file(path) != str(gate_sha256):
        raise RuntimeError("P0 real gate drifted during replay")
    return {
        "path": str(path),
        "sha256": str(gate_sha256),
        "schema": P0_REAL_GATE_SCHEMA,
        "git_commit": expected_commit,
        "ok": True,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "adatad_pretrain": {
            "path": str(adatad_pretrain),
            "sha256": sha256_file(adatad_pretrain),
        },
        "official_asformer_source": official_asformer,
    }


def _verified_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if (
        len(str(expected_sha256)) != 64
        or not resolved.is_file()
        or sha256_file(resolved) != str(expected_sha256)
    ):
        raise RuntimeError(f"{label} path/hash drift: {resolved}")
    return resolved


def _normalized_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _validate_official_asformer_source(
    binding: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"{label} binding is missing")
    source = _verified_file(
        binding.get("path", ""),
        str(binding.get("sha256", "")),
        label=label,
    )
    observed_normalized = _normalized_lf_sha256(source)
    declared = str(
        binding.get("config_declared_normalized_lf_sha256", "")
    ).lower()
    if (
        len(observed_normalized) != 64
        or binding.get("normalized_lf_sha256") != observed_normalized
        or declared != observed_normalized
    ):
        raise RuntimeError(f"{label} normalized-LF hash drift")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "normalized_lf_sha256": observed_normalized,
        "config_declared_normalized_lf_sha256": declared,
    }


def create_p0_training_asformer_consumer_receipt(
    *,
    gate_path: str | Path,
    gate_sha256: str,
    expected_commit: str,
    selected_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute official-ASFormer identity at the P0 training handoff."""

    gate = validate_p0_real_gate(
        gate_path=gate_path,
        gate_sha256=gate_sha256,
        expected_commit=expected_commit,
    )
    selected_config = Path(selected_config_path).expanduser().resolve()
    if (
        selected_config != Path(gate["config_path"])
        or not selected_config.is_file()
        or sha256_file(selected_config) != gate["config_sha256"]
    ):
        raise RuntimeError("P0 training selected config differs from the real gate")
    official_asformer = _validate_official_asformer_source(
        gate["official_asformer_source"],
        label="P0 training consumer official ASFormer source",
    )
    payload: dict[str, Any] = {
        "schema": P0_ASFORMER_CONSUMER_SCHEMA,
        "ok": True,
        "fail_closed": True,
        "git_commit": expected_commit,
        "p0_real_gate": gate,
        "selected_config_path": str(selected_config),
        "selected_config_sha256": sha256_file(selected_config),
        "official_asformer_source": official_asformer,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_write_json(
        Path(output_path).expanduser().resolve(), payload, require_absent=True
    )
    return payload


def validate_p0_training_asformer_consumer_receipt(
    *,
    receipt_path: str | Path,
    receipt_sha256: str,
    expected_commit: str,
    expected_p0_gate: Mapping[str, Any] | None = None,
    expected_config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = _verified_file(
        receipt_path,
        receipt_sha256,
        label="P0 training ASFormer consumer receipt",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("P0 training ASFormer consumer receipt is not a mapping")
    unsigned = dict(payload)
    self_hash = unsigned.pop("receipt_sha256", None)
    if (
        payload.get("schema") != P0_ASFORMER_CONSUMER_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or self_hash != canonical_sha256(unsigned)
    ):
        raise RuntimeError("P0 training ASFormer consumer contract drift")
    gate = payload.get("p0_real_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("P0 training ASFormer receipt lacks the real gate")
    reopened_gate = validate_p0_real_gate(
        gate_path=gate.get("path", ""),
        gate_sha256=str(gate.get("sha256", "")),
        expected_commit=expected_commit,
    )
    if expected_p0_gate is not None and dict(expected_p0_gate) != reopened_gate:
        raise RuntimeError("P0 training ASFormer receipt real-gate mismatch")
    if dict(gate) != reopened_gate:
        raise RuntimeError("P0 training ASFormer receipt real-gate drift")
    config = _verified_file(
        payload.get("selected_config_path", ""),
        str(payload.get("selected_config_sha256", "")),
        label="P0 training selected config",
    )
    if expected_config_path is not None and config != Path(
        expected_config_path
    ).expanduser().resolve():
        raise RuntimeError("P0 training selected config path drift")
    official_asformer = _validate_official_asformer_source(
        payload.get("official_asformer_source"),
        label="P0 training consumer official ASFormer source",
    )
    if official_asformer != reopened_gate["official_asformer_source"]:
        raise RuntimeError("P0 training ASFormer receipt differs from the real gate")
    if sha256_file(path) != str(receipt_sha256):
        raise RuntimeError("P0 training ASFormer receipt drifted during replay")
    return {
        "path": str(path),
        "sha256": str(receipt_sha256),
        "schema": P0_ASFORMER_CONSUMER_SCHEMA,
        "git_commit": expected_commit,
        "ok": True,
        "p0_real_gate": reopened_gate,
        "selected_config_path": str(config),
        "selected_config_sha256": sha256_file(config),
        "official_asformer_source": official_asformer,
    }


def validate_r0_runtime_bindings(
    *,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    annotation_path: str | Path,
    annotation_sha256: str,
    train_block_list: str | Path,
    train_block_list_sha256: str,
    holdout_block_list: str | Path,
    holdout_block_list_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    pretrain_path: str | Path,
    pretrain_sha256: str,
) -> dict[str, Any]:
    """Reopen every submit-time R0 binding before any model work."""

    binding = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
        annotation_path=annotation_path,
        train_block_list=train_block_list,
        holdout_block_list=holdout_block_list,
    )
    expected_reference_hashes = {
        "annotation_sha256": annotation_sha256,
        "train_block_list_sha256": train_block_list_sha256,
        "holdout_block_list_sha256": holdout_block_list_sha256,
    }
    for field, expected in expected_reference_hashes.items():
        if len(str(expected)) != 64 or binding.get(field) != str(expected):
            raise RuntimeError(f"submit-time split binding drift: {field}")

    checkpoint = _verified_file(
        checkpoint_path, checkpoint_sha256, label="R0 checkpoint"
    )
    pretrain = _verified_file(
        pretrain_path, pretrain_sha256, label="AdaTAD pretrain"
    )
    return {
        "ok": True,
        "schema": "duca_r0_runtime_bindings_v1",
        "split": binding,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha256,
    }


def _average_map_from_payload(payload: Mapping[str, Any], *, label: str) -> float:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"{label} metrics payload is not a mapping")
    value = metrics.get("average_mAP")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} average_mAP is missing or non-numeric")
    return _finite(value, f"{label}.average_mAP")


def validate_r0_headroom_summary(
    *,
    summary_path: str | Path,
    summary_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Reopen and recompute the complete R0 evidence chain before P0."""

    return revalidate_r0_summary(
        summary_path=summary_path,
        summary_file_sha256=summary_sha256,
        expected_commit=expected_commit,
    )


def create_family_routing_manifest(
    *,
    summary_path: str | Path,
    summary_sha256: str,
    expected_commit: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the only learned family allowed to gate P0 and matched R3."""

    if len(expected_commit) != 40:
        raise ValueError("expected commit must be exact")
    r0_gate = validate_r0_headroom_summary(
        summary_path=summary_path,
        summary_sha256=summary_sha256,
        expected_commit=expected_commit,
    )
    routing = _family_routing_contract(
        r0_gate.get("selected_weakest_projected_family")
    )
    payload: dict[str, Any] = {
        "schema": FAMILY_MANIFEST_SCHEMA,
        "ok": True,
        "git_commit": expected_commit,
        "r0_headroom_gate": r0_gate,
        "family_routing": routing,
        "test_subset_consumed": False,
        "fail_closed": True,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    output = Path(output_path).expanduser().resolve()
    _atomic_write_json(output, payload, require_absent=True)
    return payload


def validate_family_routing_manifest(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Reopen the manifest and independently replay its sealed R0 decision."""

    path = _verified_file(
        manifest_path, manifest_sha256, label="boundary-burst family manifest"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("boundary-burst family manifest is not a mapping")
    unsigned = dict(payload)
    recorded_self_hash = unsigned.pop("manifest_sha256", None)
    if (
        payload.get("schema") != FAMILY_MANIFEST_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or payload.get("test_subset_consumed") is not False
        or recorded_self_hash != canonical_sha256(unsigned)
    ):
        raise RuntimeError("boundary-burst family manifest contract drift")
    recorded_r0 = payload.get("r0_headroom_gate")
    if not isinstance(recorded_r0, Mapping):
        raise RuntimeError("boundary-burst family manifest lacks its R0 binding")
    reopened_r0 = validate_r0_headroom_summary(
        summary_path=recorded_r0.get("r0_summary_path", ""),
        summary_sha256=str(recorded_r0.get("r0_summary_sha256", "")),
        expected_commit=expected_commit,
    )
    mismatch = _first_mismatch(recorded_r0, reopened_r0, path="r0_headroom_gate")
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst family manifest R0 drift at {mismatch}")
    routing = _family_routing_contract(
        reopened_r0.get("selected_weakest_projected_family")
    )
    mismatch = _first_mismatch(
        payload.get("family_routing"), routing, path="family_routing"
    )
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst family routing drift at {mismatch}")
    if sha256_file(path) != str(manifest_sha256):
        raise RuntimeError("boundary-burst family manifest drifted during replay")
    return {
        "path": str(path),
        "sha256": str(manifest_sha256),
        "schema": FAMILY_MANIFEST_SCHEMA,
        "git_commit": expected_commit,
        "ok": True,
        "r0_headroom_gate": reopened_r0,
        "family_routing": routing,
    }


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mean(summary: Mapping[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    if isinstance(value, Mapping) and "mean" in value:
        value = value["mean"]
    return _finite(value, ".".join(keys))


def _effective_budget_contract_verified(
    summary: Mapping[str, Any],
    *,
    requested_budget: int = 384,
    requested_max_unselected_hole: int = 2,
) -> bool:
    """Require analyzer-derived per-sample K/G evidence, not aggregate means."""

    protocol = summary.get("protocol", {})
    evidence = protocol.get("sampling_contract_evidence", {}) if isinstance(protocol, Mapping) else {}
    return bool(
        isinstance(protocol, Mapping)
        and isinstance(evidence, Mapping)
        and protocol.get("budget_matched") is True
        and protocol.get("valid_length_matched") is True
        and protocol.get("max_hole_matched") is True
        and int(evidence.get("sample_count", 0)) == int(summary.get("sample_count", -1))
        and int(evidence.get("sample_count", 0)) > 0
        and int(evidence.get("budget_violation_count", -1)) == 0
        and int(evidence.get("max_hole_violation_count", -1)) == 0
        and int(evidence.get("requested_budget_min", -1)) == int(requested_budget)
        and int(evidence.get("requested_budget_max", -1)) == int(requested_budget)
        and int(evidence.get("requested_max_unselected_hole_min", -1))
        == int(requested_max_unselected_hole)
        and int(evidence.get("requested_max_unselected_hole_max", -1))
        == int(requested_max_unselected_hole)
        and int(evidence.get("effective_budget_min", -1)) > 0
        and int(evidence.get("effective_budget_max", -1)) <= int(requested_budget)
        and int(evidence.get("selected_count_min", -2))
        == int(evidence.get("effective_budget_min", -1))
        and int(evidence.get("selected_count_max", -2))
        == int(evidence.get("effective_budget_max", -1))
        and int(evidence.get("observed_max_unselected_hole_max", requested_max_unselected_hole + 1))
        <= int(requested_max_unselected_hole)
    )


def _read_candidate(candidate: Mapping[str, Any], variant: str) -> dict[str, Any]:
    checkpoint = Path(candidate["checkpoint_path"]).expanduser().resolve()
    summary_path = Path(candidate["summary_path"]).expanduser().resolve()
    records_path = Path(candidate["records_path"]).expanduser().resolve()
    for path, digest, label in (
        (checkpoint, candidate["checkpoint_sha256"], "checkpoint"),
        (summary_path, candidate["summary_sha256"], "summary"),
        (records_path, candidate["records_sha256"], "records"),
    ):
        if not path.is_file() or sha256_file(path) != str(digest):
            raise RuntimeError(f"{variant} candidate {label} drift: {path}")
    summary = _recompute_selection_summary(
        records_path=records_path,
        records_sha256=str(candidate["records_sha256"]),
        summary_path=summary_path,
        summary_sha256=str(candidate["summary_sha256"]),
    )

    learned = summary["selection"]["learned"]
    uniform = summary["selection"]["uniform"]
    simple_delta = summary["selection"].get("pure_delta_same_feasible_dp")
    if not isinstance(simple_delta, Mapping):
        raise RuntimeError(
            "selection summary lacks the same-exact-K/max-hole simple-delta DP"
        )
    metrics = {
        "coarse_auroc": _mean(summary, "coarse", "pooled", "auroc"),
        "coarse_auprc_lift": _mean(
            summary, "coarse", "pooled", "auprc_lift"
        ),
        "policy_transition_auroc_r0": _mean(
            summary, "transition", "r0", "policy", "auroc"
        ),
        "pure_delta_transition_auroc_r0": _mean(
            summary, "transition", "r0", "pure_abs_delta_p_action", "auroc"
        ),
        "boundary_recall_r0_gain": _mean(
            learned, "boundary_recall", "r0"
        )
        - _mean(uniform, "boundary_recall", "r0"),
        "learned_boundary_recall_r0": _mean(
            learned, "boundary_recall", "r0"
        ),
        "simple_delta_boundary_recall_r0": _mean(
            simple_delta, "boundary_recall", "r0"
        ),
        "uniform_minus_learned_endpoint_distance": _mean(
            uniform, "mean_endpoint_distance"
        )
        - _mean(learned, "mean_endpoint_distance"),
        "simple_delta_minus_learned_endpoint_distance": _mean(
            simple_delta, "mean_endpoint_distance"
        )
        - _mean(learned, "mean_endpoint_distance"),
        "learned_max_unselected_hole": _mean(
            learned, "max_unselected_hole"
        ),
        "learned_selected_count": _mean(learned, "selected_count"),
    }
    burst_key = VARIANT_SPECS[variant]
    if burst_key is not None:
        learned_burst = learned["boundary_burst"][burst_key]
        uniform_burst = uniform["boundary_burst"][burst_key]
        simple_delta_burst = simple_delta["boundary_burst"][burst_key]
        for field in (
            "endpoint_quota_recall",
            "endpoint_bilateral_recall",
            "both_endpoints_quota_recall",
        ):
            metrics[f"{field}_gain"] = _mean(learned_burst, field) - _mean(
                uniform_burst, field
            )
            metrics[f"{field}_gain_vs_simple_delta"] = _mean(
                learned_burst, field
            ) - _mean(simple_delta_burst, field)
    simple_delta_pareto_gains = [
        metrics["learned_boundary_recall_r0"]
        - metrics["simple_delta_boundary_recall_r0"],
        metrics["simple_delta_minus_learned_endpoint_distance"],
    ]
    if burst_key is not None:
        simple_delta_pareto_gains.extend(
            metrics[f"{field}_gain_vs_simple_delta"]
            for field in (
                "endpoint_quota_recall",
                "endpoint_bilateral_recall",
                "both_endpoints_quota_recall",
            )
        )
    simple_delta_stop_rule_pass = all(
        value >= 0.0 for value in simple_delta_pareto_gains
    ) and any(value > 0.0 for value in simple_delta_pareto_gains)
    gates = {
        "coarse_auroc_at_least_0_55": metrics["coarse_auroc"] >= 0.55,
        "coarse_auprc_above_prevalence": metrics["coarse_auprc_lift"] > 1.0,
        "transition_scorer_not_worse_than_pure_delta_r0": (
            metrics["policy_transition_auroc_r0"]
            >= metrics["pure_delta_transition_auroc_r0"]
        ),
        "endpoint_centering_not_worse_than_uniform": (
            metrics["uniform_minus_learned_endpoint_distance"] >= 0.0
        ),
        "learned_selector_strictly_pareto_beats_same_feasible_simple_delta": (
            simple_delta_stop_rule_pass
        ),
        "exact_effective_budget_per_sample": _effective_budget_contract_verified(
            summary,
        ),
    }
    if burst_key is not None:
        gates.update(
            {
                "burst_endpoint_quota_gain_positive": metrics[
                    "endpoint_quota_recall_gain"
                ]
                > 0.0,
                "burst_bilateral_gain_positive": metrics[
                    "endpoint_bilateral_recall_gain"
                ]
                > 0.0,
                "burst_both_endpoints_quota_gain_positive": metrics[
                    "both_endpoints_quota_recall_gain"
                ]
                > 0.0,
            }
        )
    epoch = int(candidate["epoch_one_based"])
    if epoch not in {5, 10, 15, 20}:
        raise RuntimeError("candidate epoch is outside the frozen P0 cadence")
    return {
        "variant": variant,
        "epoch_one_based": epoch,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "metrics": metrics,
        "gates": gates,
        "all_sanity_gates_pass": all(gates.values()),
    }


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(candidate["epoch_one_based"]),)


def validate_frontend_decision(
    *,
    decision_path: str | Path,
    decision_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Replay every policy source before a gate or official arm can consume it."""

    path = _verified_file(
        decision_path, decision_sha256, label="boundary-burst frontend decision"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or payload.get("status") != "GO_TO_MATCHED_U_SELECTED_G0_OFFICIAL60"
        or payload.get("git_commit") != expected_commit
        or payload.get("test_subset_consumed") is not False
    ):
        raise RuntimeError("boundary-burst frontend decision contract drift")

    manifest = payload.get("family_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("boundary-burst decision lacks the family manifest")
    reopened_manifest = validate_family_routing_manifest(
        manifest_path=manifest.get("path", ""),
        manifest_sha256=str(manifest.get("sha256", "")),
        expected_commit=expected_commit,
    )
    mismatch = _first_mismatch(
        manifest, reopened_manifest, path="family_manifest"
    )
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst decision manifest drift at {mismatch}")
    routing = reopened_manifest["family_routing"]
    mismatch = _first_mismatch(
        payload.get("family_routing"), routing, path="family_routing"
    )
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst decision routing drift at {mismatch}")
    mismatch = _first_mismatch(
        payload.get("r0_headroom_gate"),
        reopened_manifest["r0_headroom_gate"],
        path="r0_headroom_gate",
    )
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst decision R0 drift at {mismatch}")

    split_sha256 = payload.get("split_manifest_sha256")
    if not isinstance(split_sha256, str) or len(split_sha256) != 64:
        raise RuntimeError("boundary-burst decision split seal is invalid")
    split_binding = validate_split_manifest(
        payload.get("split_manifest_path"),
        expected_manifest_sha256=split_sha256,
    )
    mismatch = _first_mismatch(
        payload.get("split_binding"), split_binding, path="split_binding"
    )
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst decision split drift at {mismatch}")

    p0_gate = payload.get("p0_real_gate")
    if not isinstance(p0_gate, Mapping):
        raise RuntimeError("boundary-burst decision lacks the P0 real gate")
    reopened_p0_gate = validate_p0_real_gate(
        gate_path=p0_gate.get("path", ""),
        gate_sha256=str(p0_gate.get("sha256", "")),
        expected_commit=expected_commit,
    )
    mismatch = _first_mismatch(p0_gate, reopened_p0_gate, path="p0_real_gate")
    if mismatch is not None:
        raise RuntimeError(f"boundary-burst decision P0 gate drift at {mismatch}")
    p0_asformer = payload.get("p0_training_asformer_consumer")
    if not isinstance(p0_asformer, Mapping):
        raise RuntimeError("boundary-burst decision lacks P0 ASFormer consumption")
    reopened_p0_asformer = validate_p0_training_asformer_consumer_receipt(
        receipt_path=p0_asformer.get("path", ""),
        receipt_sha256=str(p0_asformer.get("sha256", "")),
        expected_commit=expected_commit,
        expected_p0_gate=reopened_p0_gate,
        expected_config_path=routing["selected_p0_config"],
    )
    mismatch = _first_mismatch(
        p0_asformer,
        reopened_p0_asformer,
        path="p0_training_asformer_consumer",
    )
    if mismatch is not None:
        raise RuntimeError(
            f"boundary-burst decision P0 ASFormer drift at {mismatch}"
        )

    winners = payload.get("winners")
    candidates = payload.get("candidates")
    selected_p0 = routing["selected_p0_variant"]
    selected_winner = (
        winners.get(selected_p0) if isinstance(winners, Mapping) else None
    )
    if (
        not isinstance(winners, Mapping)
        or not isinstance(candidates, Mapping)
        or not isinstance(selected_winner, Mapping)
        or selected_winner.get("variant") != selected_p0
        or selected_winner.get("all_sanity_gates_pass") is not True
        or len(candidates.get(selected_p0, [])) != 4
    ):
        raise RuntimeError("boundary-burst decision lacks the selected P0 winner")
    for variant in winners:
        if variant not in VARIANT_SPECS:
            raise RuntimeError(f"boundary-burst decision contains unknown winner: {variant}")
    winner = selected_winner
    for field, label in (
        ("checkpoint", "selected P0 checkpoint"),
        ("summary", "selected P0 quality summary"),
        ("records", "selected P0 quality records"),
    ):
        _verified_file(
            winner.get(f"{field}_path", ""),
            str(winner.get(f"{field}_sha256", "")),
            label=label,
        )
    if sha256_file(path) != str(decision_sha256):
        raise RuntimeError("boundary-burst frontend decision drifted during replay")
    return dict(payload)


def validate_full_model_gate(
    *,
    gate_path: str | Path,
    gate_sha256: str,
    decision_path: str | Path,
    decision_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Require a gate over exactly U and the R0-selected learned family."""

    decision = validate_frontend_decision(
        decision_path=decision_path,
        decision_sha256=decision_sha256,
        expected_commit=expected_commit,
    )
    decision_resolved = Path(decision_path).expanduser().resolve()
    path = _verified_file(
        gate_path, gate_sha256, label="boundary-burst full-model gate"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    routing = decision["family_routing"]
    required_variants = routing["required_official60_variants"]
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != FULL_MODEL_GATE_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or payload.get("formal_training_unlocked") is not True
        or payload.get("status") != "matched_u_selected_g0_full_model_gate_passed"
        or payload.get("git_commit") != expected_commit
        or Path(str(payload.get("frontend_decision_path", ""))).resolve()
        != decision_resolved
        or payload.get("frontend_decision_sha256") != decision_sha256
        or payload.get("gated_variants") != required_variants
        or payload.get("required_official60_variants") != required_variants
        or payload.get("family_manifest") != decision["family_manifest"]
        or payload.get("r0_headroom_gate") != decision["r0_headroom_gate"]
        or payload.get("family_routing") != routing
        or payload.get("p0_real_gate") != decision["p0_real_gate"]
        or payload.get("p0_training_asformer_consumer")
        != decision["p0_training_asformer_consumer"]
    ):
        raise RuntimeError("boundary-burst full-model gate contract drift")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("boundary-burst full-model gate artifacts are missing")
    artifact_paths: set[Path] = set()
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise RuntimeError("boundary-burst gate artifact index is malformed")
        artifact = _verified_file(
            record.get("path", ""),
            str(record.get("sha256", "")),
            label="boundary-burst gate suite artifact",
        )
        if artifact in artifact_paths or not artifact.is_relative_to(path.parent):
            raise RuntimeError("boundary-burst gate artifact path/uniqueness drift")
        artifact_paths.add(artifact)
    required_configs = {
        UNIFORM_OFFICIAL_VARIANT: routing["uniform_official60_config"],
        routing["selected_official60_variant"]: routing[
            "selected_official60_config"
        ],
    }
    for variant in required_variants:
        suffix = f"/full_model/{Path(required_configs[variant]).stem}.json"
        matches = [
            row
            for row in artifacts
            if isinstance(row, Mapping)
            and str(row.get("path", "")).replace("\\", "/").endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"boundary-burst gate lacks one required artifact for {variant}"
            )
        artifact = _verified_file(
            matches[0].get("path", ""),
            str(matches[0].get("sha256", "")),
            label=f"boundary-burst full-model artifact {variant}",
        )
        artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
        runtime = (
            artifact_payload.get("runtime", {})
            if isinstance(artifact_payload, Mapping)
            else {}
        )
        config_contract = (
            artifact_payload.get("config_contract", {})
            if isinstance(artifact_payload, Mapping)
            else {}
        )
        expected_config = Path(required_configs[variant]).expanduser().resolve()
        if not expected_config.is_file():
            raise RuntimeError(f"boundary-burst required config is missing: {variant}")
        pretrain = artifact_payload.get("adatad_pretrain", {})
        boundary_validity = artifact_payload.get("gt_boundary_validity", {})
        initialization = artifact_payload.get("selector_initialization")
        if (
            not isinstance(artifact_payload, Mapping)
            or artifact_payload.get("schema") != FULL_MODEL_ARTIFACT_SCHEMA
            or artifact_payload.get("ok") is not True
            or artifact_payload.get("status")
            != "p1_p2_exact_full_model_amp_ddp_gate_passed"
            or not isinstance(runtime, Mapping)
            or runtime.get("git_commit") != expected_commit
            or not isinstance(config_contract, Mapping)
            or config_contract.get("ok") is not True
            or config_contract.get("task") != "offline_temporal_action_detection"
            or Path(str(config_contract.get("config", ""))).expanduser().resolve()
            != expected_config
            or artifact_payload.get("config_sha256") != sha256_file(expected_config)
            or artifact_payload.get("real_thumos_loader_executed") is not True
            or artifact_payload.get("optimizer_exact_coverage") is not True
            or not isinstance(boundary_validity, Mapping)
            or int(boundary_validity.get("batch_size", 0)) <= 0
            or int(boundary_validity.get("endpoint_count", 0)) <= 0
            or int(boundary_validity.get("valid_endpoint_count", -1)) < 0
        ):
            raise RuntimeError(
                f"boundary-burst full-model artifact did not pass for {variant}"
            )
        reopened_pretrain = _verified_file(
            pretrain.get("path", "") if isinstance(pretrain, Mapping) else "",
            str(pretrain.get("sha256", ""))
            if isinstance(pretrain, Mapping)
            else "",
            label=f"boundary-burst full-model AdaTAD pretrain {variant}",
        )
        if {
            "path": str(reopened_pretrain),
            "sha256": sha256_file(reopened_pretrain),
        } != decision["p0_real_gate"]["adatad_pretrain"]:
            raise RuntimeError(
                f"boundary-burst full-model pretrain differs from P0 for {variant}"
            )
        official_asformer = _validate_official_asformer_source(
            artifact_payload.get("official_asformer_source"),
            label=f"boundary-burst full-model official ASFormer {variant}",
        )
        if (
            official_asformer
            != decision["p0_training_asformer_consumer"][
                "official_asformer_source"
            ]
        ):
            raise RuntimeError(
                f"boundary-burst full-model ASFormer differs from P0 for {variant}"
            )
        if variant == UNIFORM_OFFICIAL_VARIANT:
            if initialization is not None:
                raise RuntimeError(
                    "exact-uniform full-model gate initialized a frontend checkpoint"
                )
        else:
            winner = decision["winners"][routing["selected_p0_variant"]]
            initialization_unsigned = (
                dict(initialization) if isinstance(initialization, Mapping) else {}
            )
            initialization_self_sha256 = initialization_unsigned.pop(
                "receipt_sha256", None
            )
            if (
                not isinstance(initialization, Mapping)
                or initialization.get("schema") != "duca_frontend_initialization_v1"
                or initialization_self_sha256
                != canonical_sha256(initialization_unsigned)
                or Path(str(initialization.get("checkpoint_path", ""))).resolve()
                != Path(winner["checkpoint_path"]).resolve()
                or initialization.get("checkpoint_sha256")
                != winner["checkpoint_sha256"]
                or initialization.get("checkpoint_epoch")
                != int(winner["epoch_one_based"]) - 1
                or initialization.get("checkpoint_state_key") != "state_dict_ema"
                or initialization.get("detector_state_loaded") is not False
                or initialization.get("optimizer_state_loaded") is not False
                or initialization.get("scheduler_state_loaded") is not False
            ):
                raise RuntimeError(
                    "selected full-model gate frontend initialization drift"
                )
    if sha256_file(path) != str(gate_sha256):
        raise RuntimeError("boundary-burst full-model gate drifted during replay")
    return dict(payload)


def select_variants(
    *,
    expected_commit: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    family_manifest_path: str | Path,
    family_manifest_sha256: str,
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    p0_real_gate_path: str | Path | None = None,
    p0_real_gate_sha256: str | None = None,
    p0_asformer_consumer_path: str | Path | None = None,
    p0_asformer_consumer_sha256: str | None = None,
) -> dict[str, Any]:
    if len(expected_commit) != 40:
        raise ValueError("expected commit must be exact")
    split_binding = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
    )
    split_path = Path(split_binding["manifest_path"])
    family_manifest = validate_family_routing_manifest(
        manifest_path=family_manifest_path,
        manifest_sha256=family_manifest_sha256,
        expected_commit=expected_commit,
    )
    routing = family_manifest["family_routing"]
    selected_p0_variant = routing["selected_p0_variant"]
    if p0_real_gate_path is None or p0_real_gate_sha256 is None:
        raise RuntimeError("P0 real gate binding is required")
    p0_real_gate = validate_p0_real_gate(
        gate_path=p0_real_gate_path,
        gate_sha256=p0_real_gate_sha256,
        expected_commit=expected_commit,
    )
    if p0_asformer_consumer_path is None or p0_asformer_consumer_sha256 is None:
        raise RuntimeError("P0 training ASFormer consumer binding is required")
    p0_asformer_consumer = validate_p0_training_asformer_consumer_receipt(
        receipt_path=p0_asformer_consumer_path,
        receipt_sha256=p0_asformer_consumer_sha256,
        expected_commit=expected_commit,
        expected_p0_gate=p0_real_gate,
        expected_config_path=routing["selected_p0_config"],
    )

    candidates: dict[str, list[dict[str, Any]]] = {
        key: [] for key in VARIANT_SPECS
    }
    seen_variants: set[str] = set()
    for raw in receipt_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"mechanism completion receipt is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant"))
        if (
            variant not in VARIANT_SPECS
            or payload.get("schema") != "duca_frontend_variant_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("split_manifest_sha256") != split_manifest_sha256
            or payload.get("test_subset_consumed") is not False
        ):
            raise RuntimeError(f"invalid mechanism completion receipt: {path}")
        if variant in seen_variants:
            raise RuntimeError(f"duplicate mechanism completion receipt: {variant}")
        seen_variants.add(variant)
        rows = payload.get("candidates", [])
        if [int(row["epoch_one_based"]) for row in rows] != [5, 10, 15, 20]:
            raise RuntimeError(f"{variant} does not cover the frozen checkpoint cadence")
        candidates[variant].extend(_read_candidate(row, variant) for row in rows)

    winners: dict[str, dict[str, Any]] = {}
    for variant, rows in candidates.items():
        if variant == selected_p0_variant and len(rows) != 4:
            raise RuntimeError(
                f"R0-selected family {variant} requires four P0 checkpoints"
            )
        if variant != selected_p0_variant and len(rows) not in {0, 4}:
            raise RuntimeError(
                f"diagnostic family {variant} has a partial P0 checkpoint set"
            )
        eligible = [row for row in rows if row["all_sanity_gates_pass"]]
        if eligible:
            winners[variant] = sorted(eligible, key=_ranking_key)[0]
    ok = selected_p0_variant in winners
    payload = {
        "schema": SCHEMA,
        "ok": ok,
        "fail_closed": True,
        "status": (
            "GO_TO_MATCHED_U_SELECTED_G0_OFFICIAL60"
            if ok
            else "HOLD_SELECTED_P0_SANITY_GATE_FAILED"
        ),
        "git_commit": expected_commit,
        "decision_metric_scope": "training_subset_internal_holdout_only",
        "test_subset_consumed": False,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "split_binding": split_binding,
        "family_manifest": family_manifest,
        "r0_headroom_gate": family_manifest["r0_headroom_gate"],
        "family_routing": routing,
        "p0_real_gate": p0_real_gate,
        "p0_training_asformer_consumer": p0_asformer_consumer,
        "selection_rule": (
            "the R0-selected burst family is the only required learned P0 family; "
            "choose its earliest passing checkpoint at epoch 5/10/15/20; Gaussian "
            "and the unselected burst family are non-vetoing diagnostics; final method "
            "ranking is reserved for matched terminal-EMA U versus selected G0 TAD mAP"
        ),
        "winners": winners,
        "candidates": candidates,
        "diagnostic_failures_block_main": False,
        "paper_metric_claim_allowed": False,
    }
    output = Path(output_path).expanduser().resolve()
    _atomic_write_json(output, payload, require_absent=True)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--family-manifest", required=True)
    parser.add_argument("--family-manifest-sha256", required=True)
    parser.add_argument("--p0-real-gate", required=True)
    parser.add_argument("--p0-real-gate-sha256", required=True)
    parser.add_argument("--p0-asformer-consumer", required=True)
    parser.add_argument("--p0-asformer-consumer-sha256", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = select_variants(
        expected_commit=args.expected_commit,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        family_manifest_path=args.family_manifest,
        family_manifest_sha256=args.family_manifest_sha256,
        p0_real_gate_path=args.p0_real_gate,
        p0_real_gate_sha256=args.p0_real_gate_sha256,
        p0_asformer_consumer_path=args.p0_asformer_consumer,
        p0_asformer_consumer_sha256=args.p0_asformer_consumer_sha256,
        receipt_paths=args.receipt,
        output_path=args.output_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
