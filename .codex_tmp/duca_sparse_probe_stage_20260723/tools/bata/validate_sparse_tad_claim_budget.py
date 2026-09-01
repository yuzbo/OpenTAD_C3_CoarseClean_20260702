from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config


SCHEMA_VERSION = "c3_sparse_tad_claim_budget_gate_v1"
READY = "C3_SPARSE_TAD_CLAIM_BUDGET_GATE_PASS"
FAIL = "C3_SPARSE_TAD_CLAIM_BUDGET_GATE_FAIL"
DIAGNOSTIC_ONLY = "C3_SPARSE_TAD_CLAIM_BUDGET_GATE_DIAGNOSTIC_ONLY"
DEFAULT_MAX_BUDGET = 384
PAPER_MAIN = "paper-main"
DIAGNOSTIC = "diagnostic"

TARGET_FIELD_KEYS = {
    "target_len",
    "expected_target_len",
    "target",
    "target_budget",
    "dynamic_target_len",
    "dynamic_target",
    "dynamic_target_budget",
    "window_size",
    "selected_window_size",
}
SELECTED_COUNT_FIELD_KEYS = {
    "selected_count",
    "require_selected_count",
    "required_selected_count",
    "max_selected_count",
    "max_dynamic_budget",
}
SUPPORTING_SELECTED_COUNT_KEYS = {
    "min_selected_count",
    "mean_selected_count",
    "min_dynamic_budget",
    "mean_dynamic_budget",
}
BUCKET_FIELD_KEYS = {
    "dynamic_budget_buckets",
    "dynamic_budget_bucket_values",
    "budget_buckets",
    "dynamic_buckets",
    "selected_count_buckets",
}
CONTAINER_KEYS = {
    "metrics",
    "validation_summary",
    "ledger_summary",
    "summary",
    "claim_manifest",
    "manifest",
    "spec",
    "dynamic",
    "dynamic_budget",
    "budget",
    "diagnostics",
}
VARIANT_FIELD_KEYS = {
    "variant",
    "variant_name",
    "strategy",
    "ledger_name",
    "claim_name",
    "detector_aware_ledger_variant",
    "paction_ledger_variant",
}
DIAGNOSTIC_MARKER_KEYS = {
    "diagnostic_only",
    "diagnostic",
    "diagnostic_ceiling",
}
FORBIDDEN_CLAIM_FLAGS = {
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _finite_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number))


def _mapping_get(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    if hasattr(payload, "get"):
        try:
            return payload.get(key)
        except Exception:
            pass
    if hasattr(payload, key):
        return getattr(payload, key)
    return None


def _nested_get(payload: Any, keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        current = _mapping_get(current, key)
        if current is None:
            return None
    return current


def _as_plain_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if hasattr(payload, "to_dict"):
        maybe_dict = payload.to_dict()
        if isinstance(maybe_dict, Mapping):
            return maybe_dict
    return {}


def _append_budget_field(fields: dict[str, int], name: str, value: Any) -> None:
    parsed = _finite_int(value)
    if parsed is not None:
        fields[name] = parsed


def _candidate_budget_fields(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, int]:
    fields: dict[str, int] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key in TARGET_FIELD_KEYS or key in SELECTED_COUNT_FIELD_KEYS or key in SUPPORTING_SELECTED_COUNT_KEYS:
            _append_budget_field(fields, name, value)
        if key in BUCKET_FIELD_KEYS and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                _append_budget_field(fields, f"{name}[{index}]", item)
        if key == "selected_count_histogram" and isinstance(value, Mapping):
            for histogram_key in value:
                _append_budget_field(fields, f"{name}.{histogram_key}", histogram_key)
        if key in CONTAINER_KEYS and isinstance(value, Mapping):
            fields.update(_candidate_budget_fields(value, prefix=name))
    return fields


def _selected_count_evidence_fields(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, int]:
    fields: dict[str, int] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key in SELECTED_COUNT_FIELD_KEYS or key == "dynamic_target":
            _append_budget_field(fields, name, value)
        if key in BUCKET_FIELD_KEYS and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                _append_budget_field(fields, f"{name}[{index}]", item)
        if key == "selected_count_histogram" and isinstance(value, Mapping):
            for histogram_key in value:
                _append_budget_field(fields, f"{name}.{histogram_key}", histogram_key)
        if key in CONTAINER_KEYS and isinstance(value, Mapping):
            fields.update(_selected_count_evidence_fields(value, prefix=name))
    return fields


def _collect_variant_names(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, str]:
    variants: dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key in VARIANT_FIELD_KEYS and value is not None:
            variants[name] = str(value)
        if key in CONTAINER_KEYS and isinstance(value, Mapping):
            variants.update(_collect_variant_names(value, prefix=name))
    return variants


def _collect_true_flags(payload: Mapping[str, Any], flag_names: set[str], *, prefix: str = "") -> list[str]:
    flags: list[str] = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key in flag_names and value is True:
            flags.append(name)
        if key in CONTAINER_KEYS and isinstance(value, Mapping):
            flags.extend(_collect_true_flags(value, flag_names, prefix=name))
    return flags


def _collect_diagnostic_markers(payload: Mapping[str, Any]) -> list[str]:
    markers = _collect_true_flags(payload, DIAGNOSTIC_MARKER_KEYS)
    claim_mode = str(payload.get("claim_mode") or payload.get("mode") or "").strip().lower()
    if claim_mode in {DIAGNOSTIC, "diagnostic-only", "diagnostic_only"}:
        markers.append("claim_mode")
    return markers


def _variant_budget_violations(variant_names: Mapping[str, str], *, max_budget: int) -> dict[str, str]:
    violations: dict[str, str] = {}
    for key, value in variant_names.items():
        lowered = value.lower()
        if "fixed_768" in lowered or "fixed768" in lowered:
            violations[key] = f"{value} is a diagnostic ceiling and cannot support <= {max_budget} paper-main claim"
        if "dynamic" in lowered:
            violations[key] = f"{value} is a dynamic diagnostic and cannot support paper-main claim"
        if "paction_lattice" in lowered or "lattice_replace" in lowered:
            violations[key] = f"{value} is a p_action lattice diagnostic and cannot support paper-main claim"
    return violations


def _validate_evidence_payload(
    payload: Mapping[str, Any],
    *,
    source_type: str,
    source_path: str | Path,
    max_budget: int,
    claim_mode: str = PAPER_MAIN,
) -> dict[str, Any]:
    checked = _candidate_budget_fields(payload)
    selected_evidence = _selected_count_evidence_fields(payload)
    variants = _collect_variant_names(payload)
    diagnostic_markers = _collect_diagnostic_markers(payload)
    forbidden_flags = _collect_true_flags(payload, FORBIDDEN_CLAIM_FLAGS)
    numeric_violations = {key: value for key, value in checked.items() if int(value) > int(max_budget)}
    variant_violations = _variant_budget_violations(variants, max_budget=int(max_budget))
    diagnostic_mode = str(claim_mode).strip().lower() in {DIAGNOSTIC, "diagnostic-only", "diagnostic_only"}

    _require(checked, f"{source_type} has no budget/count fields: {source_path}")
    _require(
        selected_evidence,
        f"{source_type} missing selected-count evidence: require selected_count, require_selected_count, "
        f"max_selected_count, selected_count_histogram, or dynamic budget buckets",
    )
    _require(not forbidden_flags, f"{source_type} sets forbidden claim flags: {forbidden_flags}")
    if not diagnostic_mode:
        _require(not diagnostic_markers, f"{source_type} is marked diagnostic and cannot paper-main pass")
        _require(not numeric_violations, f"{source_type} exceeds claim budget {max_budget}: {numeric_violations}")
        _require(not variant_violations, f"{source_type} exceeds claim budget {max_budget}: {variant_violations}")

    return {
        "source_type": source_type,
        "source_path": str(source_path),
        "variant_fields": dict(variants),
        "checked_budget_fields": checked,
        "selected_count_evidence_fields": selected_evidence,
        "max_budget": int(max_budget),
        "diagnostic_markers": diagnostic_markers,
        "budget_violations_for_main_claim": numeric_violations,
        "variant_violations_for_main_claim": variant_violations,
    }


def validate_json_budget(
    path: str | Path,
    *,
    source_type: str,
    max_budget: int = DEFAULT_MAX_BUDGET,
    claim_mode: str = PAPER_MAIN,
) -> dict[str, Any]:
    return _validate_evidence_payload(
        _read_json(path),
        source_type=source_type,
        source_path=path,
        max_budget=int(max_budget),
        claim_mode=claim_mode,
    )


def validate_ledger_summary_budget(
    summary_json: str | Path,
    *,
    max_budget: int = DEFAULT_MAX_BUDGET,
    claim_mode: str = PAPER_MAIN,
) -> dict[str, Any]:
    report = validate_json_budget(
        summary_json,
        source_type="ledger_summary",
        max_budget=max_budget,
        claim_mode=claim_mode,
    )
    return {
        "summary_json": str(summary_json),
        "checked_budget_fields": report["checked_budget_fields"],
        "selected_count_evidence_fields": report["selected_count_evidence_fields"],
        "max_budget": int(max_budget),
        **report,
    }


def _maybe_cfg_int(cfg: Config, expr: Sequence[str]) -> int | None:
    value = _nested_get(cfg, expr)
    return _finite_int(value)


def validate_config_budget(
    config_path: str | Path,
    *,
    max_budget: int = DEFAULT_MAX_BUDGET,
    claim_mode: str = PAPER_MAIN,
) -> dict[str, Any]:
    cfg = Config.fromfile(str(config_path))
    checked: dict[str, int] = {}
    selected_evidence: dict[str, int] = {}
    variant_names: dict[str, str] = {}
    for name, expr in (
        ("window_size", ("window_size",)),
        ("selected_window_size", ("selected_window_size",)),
        ("detector_aware_require_selected_count", ("detector_aware_require_selected_count",)),
        ("paction_require_selected_count", ("paction_require_selected_count",)),
        ("model.backbone.backbone.total_frames", ("model", "backbone", "backbone", "total_frames")),
        ("model.projection.max_seq_len", ("model", "projection", "max_seq_len")),
        ("model.frame_selector.selected_count", ("model", "frame_selector", "selected_count")),
        ("model.frame_selector.target_len", ("model", "frame_selector", "target_len")),
    ):
        value = _maybe_cfg_int(cfg, expr)
        if value is not None:
            checked[name] = value
            if name.endswith(("selected_count", "require_selected_count")):
                selected_evidence[name] = value

    active_variant = (
        _mapping_get(cfg, "detector_aware_ledger_variant")
        or _mapping_get(cfg, "paction_ledger_variant")
        or _mapping_get(cfg, "ledger_variant")
    )
    if active_variant is not None:
        variant_names["active_variant"] = str(active_variant)
    variant_specs = _as_plain_mapping(_mapping_get(cfg, "VARIANT_SPECS"))
    if active_variant in variant_specs:
        variant_spec = _as_plain_mapping(variant_specs[active_variant])
        for key in ("target_len", "target", "require_selected_count", "required_selected_count", "max_selected_count"):
            value = _finite_int(_mapping_get(variant_spec, key))
            if value is not None:
                checked[f"VARIANT_SPECS.{active_variant}.{key}"] = value
                if key in SELECTED_COUNT_FIELD_KEYS:
                    selected_evidence[f"VARIANT_SPECS.{active_variant}.{key}"] = value
        strategy = _mapping_get(variant_spec, "strategy")
        if strategy is not None:
            variant_names[f"VARIANT_SPECS.{active_variant}.strategy"] = str(strategy)

    diagnostic_mode = str(claim_mode).strip().lower() in {DIAGNOSTIC, "diagnostic-only", "diagnostic_only"}
    numeric_violations = {key: value for key, value in checked.items() if int(value) > int(max_budget)}
    variant_violations = _variant_budget_violations(variant_names, max_budget=int(max_budget))

    _require(checked, f"config has no budget/count fields: {config_path}")
    _require(selected_evidence, f"config missing selected-count evidence: {config_path}")
    if not diagnostic_mode:
        _require(not numeric_violations, f"config exceeds claim budget {max_budget}: {numeric_violations}")
        _require(not variant_violations, f"config exceeds claim budget {max_budget}: {variant_violations}")
    return {
        "source_type": "config",
        "config_path": str(config_path),
        "source_path": str(config_path),
        "variant_fields": variant_names,
        "checked_budget_fields": checked,
        "selected_count_evidence_fields": selected_evidence,
        "max_budget": int(max_budget),
        "diagnostic_markers": [],
        "budget_violations_for_main_claim": numeric_violations,
        "variant_violations_for_main_claim": variant_violations,
    }


def build_claim_budget_report(
    *,
    configs: Sequence[str | Path] = (),
    specs: Sequence[str | Path] = (),
    ledger_summaries: Sequence[str | Path] = (),
    claim_manifests: Sequence[str | Path] = (),
    max_budget: int = DEFAULT_MAX_BUDGET,
    claim_mode: str = PAPER_MAIN,
) -> dict[str, Any]:
    diagnostic_mode = str(claim_mode).strip().lower() in {DIAGNOSTIC, "diagnostic-only", "diagnostic_only"}
    config_reports = [validate_config_budget(path, max_budget=max_budget, claim_mode=claim_mode) for path in configs]
    spec_reports = [validate_json_budget(path, source_type="spec", max_budget=max_budget, claim_mode=claim_mode) for path in specs]
    ledger_reports = [
        validate_ledger_summary_budget(path, max_budget=max_budget, claim_mode=claim_mode) for path in ledger_summaries
    ]
    manifest_reports = [
        validate_json_budget(path, source_type="claim_manifest", max_budget=max_budget, claim_mode=claim_mode)
        for path in claim_manifests
    ]
    _require(
        config_reports or spec_reports or ledger_reports or manifest_reports,
        "at least one config, spec, ledger summary, or claim manifest is required",
    )
    evidence_reports = [*config_reports, *spec_reports, *ledger_reports, *manifest_reports]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": DIAGNOSTIC_ONLY if diagnostic_mode else READY,
        "max_budget": int(max_budget),
        "claim_scope": "sparse_prebackbone_tad_main_claim",
        "claim_mode": DIAGNOSTIC if diagnostic_mode else PAPER_MAIN,
        "paper_main_claim_allowed": not diagnostic_mode,
        "diagnostic_only": bool(diagnostic_mode),
        "budget_contract": {
            "target_len_lte": int(max_budget),
            "selected_count_lte": int(max_budget),
            "max_selected_count_lte": int(max_budget),
            "dynamic_max_selected_count_lte": int(max_budget),
            "fixed_768_and_dynamic_768_are_diagnostic_only": True,
        },
        "evidence_reports": evidence_reports,
        "config_reports": config_reports,
        "spec_reports": spec_reports,
        "ledger_summary_reports": ledger_reports,
        "claim_manifest_reports": manifest_reports,
    }


def _failure_report(error: BaseException, *, max_budget: int, claim_mode: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": FAIL,
        "max_budget": int(max_budget),
        "claim_mode": str(claim_mode),
        "paper_main_claim_allowed": False,
        "diagnostic_only": False,
        "error": str(error),
        "error_type": type(error).__name__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed <=384 sparse TAD claim-budget validator.")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--ledger-summary", action="append", default=[])
    parser.add_argument("--claim-manifest", action="append", default=[])
    parser.add_argument("--claim-mode", choices=[PAPER_MAIN, DIAGNOSTIC], default=PAPER_MAIN)
    parser.add_argument("--max-budget", type=int, default=DEFAULT_MAX_BUDGET)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = build_claim_budget_report(
            configs=args.config,
            specs=args.spec,
            ledger_summaries=args.ledger_summary,
            claim_manifests=args.claim_manifest,
            max_budget=int(args.max_budget),
            claim_mode=args.claim_mode,
        )
        exit_code = 0
    except (AssertionError, ValueError, OSError) as exc:
        report = _failure_report(exc, max_budget=int(args.max_budget), claim_mode=args.claim_mode)
        exit_code = 1
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
