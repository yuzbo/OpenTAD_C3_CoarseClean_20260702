from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SELECTION_RELEVANT_KEYS = (
    "sample_id",
    "video_name",
    "window_start_frame",
    "dense_len",
    "valid_len",
    "frame_signals",
    "p_action",
    "strategy_selected_positions",
    "action_target",
    "gt_boundaries",
    "gt_segments",
)
TRUST_RELEVANT_KEYS = (
    "split",
    "subset",
    "paction_positive_provenance",
    "p_action_provenance",
    "p_action_source",
    "source_p_action",
    "diagnostic_only",
    "deploy_selection_ledger",
    "probe_model",
    "tcn_variant",
    "matrix_model_id",
    "official_action_seg_backend",
    "spatial_size",
    "probe_checkpoint_sha256",
    "probe_manifest_sha256",
    "uses_gt",
    "uses_gt_for_diagnostics",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
    "training_only",
    "diagnostic_only",
)
CANONICAL_SIGNATURE_KEYS = tuple(dict.fromkeys(SELECTION_RELEVANT_KEYS + TRUST_RELEVANT_KEYS))
STRICT_DEPLOY_FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_gt_for_diagnostics",
    "diagnostic_only",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
    "training_only",
)
SELECTION_SOURCE_FORBIDDEN_TRUE_FLAGS = STRICT_DEPLOY_FORBIDDEN_TRUE_FLAGS
STRICT_DEPLOY_PAYLOAD_KEYS = (
    "action_target",
    "action_labels",
    "gt_boundaries",
    "boundaries",
    "gt_segments",
    "gt_labels",
    "ground_truth",
    "teacher",
    "teacher_logits",
    "teacher_scores",
    "teacher_predictions",
    "oracle",
    "oracle_scores",
    "oracle_selected_positions",
    "prediction_cache",
    "prediction_cache_path",
    "raw_prediction",
    "raw_predictions",
    "raw_scores",
    "raw_logits",
    "boundary_support_r1",
    "boundary_support_r2",
    "boundary_support_r4",
    "boundary_support_r8",
    "action_coverage",
    "action_positive_coverage",
)
SELECTION_SOURCE_STRIP_KEYS = tuple(
    dict.fromkeys(STRICT_DEPLOY_PAYLOAD_KEYS + ("uses_gt_for_diagnostics", "diagnostic_only", "deploy_selection_ledger"))
)
SELECTION_SOURCE_STRIPPABLE_DIAGNOSTIC_TRUE_FLAGS = (
    "uses_gt_for_diagnostics",
    "diagnostic_only",
)
SELECTION_SOURCE_PRE_STRIP_FORBIDDEN_TRUE_FLAGS = tuple(
    key for key in SELECTION_SOURCE_FORBIDDEN_TRUE_FLAGS if key not in SELECTION_SOURCE_STRIPPABLE_DIAGNOSTIC_TRUE_FLAGS
)
PROVENANCE_FALSE_FLAGS = (
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
)
PROVENANCE_FORBIDDEN_TRUE_FLAGS = tuple(
    dict.fromkeys(
        STRICT_DEPLOY_FORBIDDEN_TRUE_FLAGS
        + (
            "uses_teacher",
            "uses_oracle",
            "uses_cache",
            "uses_prediction_cache",
            "uses_raw_prediction",
            "prediction_uses_gt",
        )
    )
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: sample row must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical_row_signature(row: Mapping[str, Any]) -> str:
    payload = {key: row.get(key) for key in CANONICAL_SIGNATURE_KEYS if key in row}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_paction_positive_provenance(provenance: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source_name}: p_action positive provenance is required")
    if not _nonempty_text(provenance.get("p_action_source")):
        raise ValueError(f"{source_name}: p_action positive provenance must include p_action_source")
    model_markers = (
        provenance.get("probe_model"),
        provenance.get("matrix_model_id"),
        provenance.get("official_action_seg_backend"),
        provenance.get("source_model"),
        provenance.get("model"),
        provenance.get("probe_checkpoint_sha256"),
        provenance.get("probe_manifest_sha256"),
    )
    if not any(_nonempty_text(item) for item in model_markers):
        raise ValueError(f"{source_name}: p_action positive provenance must include a model/backend/checkpoint marker")
    if provenance.get("no_gt_generation") is not True:
        raise ValueError(f"{source_name}: p_action positive provenance must set no_gt_generation=true")
    for key in PROVENANCE_FORBIDDEN_TRUE_FLAGS:
        if _is_true(provenance.get(key, False)):
            raise ValueError(f"{source_name}: p_action positive provenance forbidden flag {key}=true")
    for key in PROVENANCE_FALSE_FLAGS:
        if provenance.get(key) is not False:
            raise ValueError(f"{source_name}: p_action positive provenance must set {key}=false")
    return copy.deepcopy(dict(provenance))


def infer_paction_positive_provenance_from_row(row: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    for key in SELECTION_SOURCE_PRE_STRIP_FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"{source_name}: cannot infer p_action provenance with forbidden source flag {key}=true")
    p_action_source = row.get("p_action_source") or row.get("source_p_action") or "lowres_action_probe"
    provenance: dict[str, Any] = {
        "p_action_source": str(p_action_source),
        "no_gt_generation": True,
        "inferred_from_source_row": True,
        "uses_gt": False,
        "uses_gt_for_diagnostics": False,
        "diagnostic_only": False,
        "training_only": False,
    }
    for key in PROVENANCE_FALSE_FLAGS:
        provenance[key] = False
    for key in (
        "probe_model",
        "tcn_variant",
        "matrix_model_id",
        "official_action_seg_backend",
        "spatial_size",
        "probe_checkpoint_sha256",
        "probe_manifest_sha256",
    ):
        value = row.get(key)
        if _nonempty_text(value):
            provenance[key] = str(value)
        elif isinstance(value, int):
            provenance[key] = int(value)
    return validate_paction_positive_provenance(provenance, source_name=source_name)


def paction_positive_provenance_from_row(row: Mapping[str, Any], *, source_name: str, strict: bool = True) -> dict[str, Any]:
    raw = row.get("paction_positive_provenance")
    if raw is None:
        raw = row.get("p_action_provenance")
    if raw is None and strict:
        raise ValueError(f"{source_name}: p_action positive provenance is required")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source_name}: p_action positive provenance must be a JSON object")
    return validate_paction_positive_provenance(raw, source_name=source_name)


def reject_strict_deploy_source_row(
    row: Mapping[str, Any],
    *,
    source_name: str,
    reject_payload: bool = True,
) -> dict[str, Any]:
    for key in STRICT_DEPLOY_FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"{source_name}: forbidden strict deploy p_action source flag {key}=true")
    if reject_payload:
        for key in STRICT_DEPLOY_PAYLOAD_KEYS:
            if key in row:
                raise ValueError(f"{source_name}: forbidden strict deploy p_action source payload key {key}")
    return paction_positive_provenance_from_row(row, source_name=source_name, strict=True)


def _strip_selection_deploy_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(row))
    for key in SELECTION_SOURCE_STRIP_KEYS:
        out.pop(key, None)
    out["deploy_selection_source_stripped"] = True
    return out


def canonicalize_unique_sample_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    split: str = "",
    allow_identical_drop: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    seen: dict[str, tuple[str, int]] = {}
    out_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for line_no, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{input_jsonl}:{line_no}: sample row is missing sample_id")
        signature = canonical_row_signature(row)
        if sample_id in seen:
            old_signature, old_line = seen[sample_id]
            item = {
                "sample_id": sample_id,
                "first_line": int(old_line),
                "duplicate_line": int(line_no),
                "identical": bool(signature == old_signature),
            }
            duplicates.append(item)
            if signature == old_signature and allow_identical_drop:
                continue
            conflicts.append(item)
            raise ValueError(
                f"{input_jsonl}:{line_no}: conflicting duplicate sample_id={sample_id}; "
                f"first_line={old_line}"
            )
        seen[sample_id] = (signature, line_no)
        out_rows.append(dict(row))

    _write_jsonl(output_jsonl, out_rows)
    report = {
        "schema_version": "c3_paction_source_sample_canonicalization_v1",
        "split": str(split),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_jsonl_sha256": _sha256_file(input_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "canonical_signature_keys": list(CANONICAL_SIGNATURE_KEYS),
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "unique_sample_ids": len(seen),
        "duplicate_count": len(duplicates),
        "conflicting_duplicate_count": len(conflicts),
        "duplicates": duplicates,
    }
    if report_json is not None:
        _write_json(report_json, report)
    return report


def write_deploy_selection_source_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    split: str = "",
    allow_inferred_paction_positive_provenance: bool = False,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    out_rows: list[dict[str, Any]] = []
    stripped_key_counts: dict[str, int] = {}
    stripped_true_diagnostic_flag_counts: dict[str, int] = {}
    inferred_paction_positive_provenance_count = 0
    for line_no, row in enumerate(rows, start=1):
        source_name = f"{input_jsonl}:{line_no}"
        provenance = paction_positive_provenance_from_row(
            row,
            source_name=source_name,
            strict=not bool(allow_inferred_paction_positive_provenance),
        )
        if not provenance and allow_inferred_paction_positive_provenance:
            provenance = infer_paction_positive_provenance_from_row(row, source_name=source_name)
            inferred_paction_positive_provenance_count += 1
        for key in SELECTION_SOURCE_PRE_STRIP_FORBIDDEN_TRUE_FLAGS:
            if _is_true(row.get(key, False)):
                raise ValueError(f"{source_name}: forbidden strict deploy p_action source flag {key}=true")
        stripped = _strip_selection_deploy_payload(row)
        for key in SELECTION_SOURCE_STRIP_KEYS:
            if key in row:
                stripped_key_counts[key] = stripped_key_counts.get(key, 0) + 1
                if key in SELECTION_SOURCE_STRIPPABLE_DIAGNOSTIC_TRUE_FLAGS and _is_true(row.get(key, False)):
                    stripped_true_diagnostic_flag_counts[key] = stripped_true_diagnostic_flag_counts.get(key, 0) + 1
        stripped["paction_positive_provenance"] = provenance
        reject_strict_deploy_source_row(
            stripped,
            source_name=f"{source_name}:selection_deploy_source",
            reject_payload=True,
        )
        out_rows.append(stripped)

    _write_jsonl(output_jsonl, out_rows)
    report = {
        "schema_version": "c3_paction_deploy_selection_source_v1",
        "split": str(split),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_jsonl_sha256": _sha256_file(input_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "stripped_key_counts": stripped_key_counts,
        "stripped_true_diagnostic_flag_counts": stripped_true_diagnostic_flag_counts,
        "inferred_paction_positive_provenance_count": int(inferred_paction_positive_provenance_count),
        "rejects_true_gt_teacher_or_cache_flags": True,
        "selection_source_forbidden_true_flags": list(SELECTION_SOURCE_PRE_STRIP_FORBIDDEN_TRUE_FLAGS),
        "selection_source_strippable_diagnostic_true_flags": list(SELECTION_SOURCE_STRIPPABLE_DIAGNOSTIC_TRUE_FLAGS),
        "requires_paction_positive_provenance": True,
        "allow_inferred_paction_positive_provenance": bool(allow_inferred_paction_positive_provenance),
    }
    if report_json is not None:
        _write_json(report_json, report)
    return report
