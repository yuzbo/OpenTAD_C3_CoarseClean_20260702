from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402
from tools.bata import paction_budget_contract  # noqa: E402
from tools.bata import paction_source_samples  # noqa: E402


_BOUNDARY_ACQUISITION_PATH = ROOT / "opentad" / "datasets" / "transforms" / "boundary_acquisition.py"
_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "pc_ot_mras_lowres_probe_boundary_acquisition",
    _BOUNDARY_ACQUISITION_PATH,
)
_BOUNDARY_MODULE = importlib.util.module_from_spec(_BOUNDARY_SPEC)
assert _BOUNDARY_SPEC.loader is not None
sys.modules[_BOUNDARY_SPEC.name] = _BOUNDARY_MODULE
_BOUNDARY_SPEC.loader.exec_module(_BOUNDARY_MODULE)
validate_value_transport_selection_row = _BOUNDARY_MODULE.validate_value_transport_selection_row


OUTPUT_SCHEMA_VERSION = "pc_ot_mras_frontend_value_transport_ledger_v0"
SUMMARY_SCHEMA_VERSION = "c3_lowres_probe_value_transport_ledger_summary_v0"
READY = "C3_LOWRES_PROBE_LEDGER_READY"
NO_GO = "C3_LOWRES_PROBE_LEDGER_NO_GO"
CHECKPOINT_POLICY_SOURCE = "learned_paction_gap_loss_policy_checkpoint"
GAS_VT_CHECKPOINT_POLICY_SOURCE = "learned_paction_gas_vt_policy_checkpoint"
DEPLOY_CHECKPOINT_POLICY_SOURCES = {
    CHECKPOINT_POLICY_SOURCE,
    GAS_VT_CHECKPOINT_POLICY_SOURCE,
}
RADIUS_MOVE25_STRATEGY = "paction_lattice_radius_score_only_move25"
RADIUS_MOVE50_STRATEGY = "paction_lattice_radius_score_only_move50"
RADIUS_STRATEGIES = {
    RADIUS_MOVE25_STRATEGY,
    RADIUS_MOVE50_STRATEGY,
}
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
    "training_only",
)
WINDOW_SAMPLE_ID_RE = re.compile(r"^[^|\s][^|]*\|(?:0|[1-9][0-9]*)$")
INTEGER_TEXT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: probe sample row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"probe samples JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if INTEGER_TEXT_RE.fullmatch(text):
            return int(text)
        raise ValueError(f"{name} must be an integer") from None
    raise ValueError(f"{name} must be an integer")


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    return _strict_int(row[key], name=key)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_positions(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    out: list[int] = []
    for idx, item in enumerate(value):
        position = _strict_int(item, name=f"{name}[{idx}]")
        if position < 0:
            raise ValueError(f"{name}[{idx}] must be non-negative")
        out.append(position)
    if out != sorted(out):
        raise ValueError(f"{name} must be sorted")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique")
    return out


def _validate_sample_id(sample_id: str, *, line_no: int, require_window_sample_id: bool) -> None:
    if require_window_sample_id:
        if not WINDOW_SAMPLE_ID_RE.fullmatch(sample_id):
            raise ValueError(
                f"line {line_no}: sample_id must match video_name|non_negative_window_start"
            )
        return
    if "|" in sample_id and not WINDOW_SAMPLE_ID_RE.fullmatch(sample_id):
        raise ValueError(
            f"line {line_no}: sample_id with a window separator must match video_name|non_negative_window_start"
        )


def _uniform_fill_positions(
    selected: Sequence[int],
    *,
    valid_len: int,
    target_count: int,
) -> tuple[list[int], int]:
    selected_set = {int(item) for item in selected if 0 <= int(item) < int(valid_len)}
    if target_count <= len(selected_set):
        return sorted(selected_set)[: int(target_count)], 0
    candidates = [idx for idx in range(max(int(valid_len), 0)) if idx not in selected_set]
    if not candidates:
        return sorted(selected_set), 0
    need = min(int(target_count) - len(selected_set), len(candidates))
    if need <= 0:
        return sorted(selected_set), 0
    if need == 1:
        fill = [candidates[len(candidates) // 2]]
    else:
        fill = []
        last = len(candidates) - 1
        for rank in range(need):
            fill.append(candidates[round(rank * last / float(need - 1))])
    return sorted(selected_set.union(fill)), len(fill)


def _expected_required_count(
    *,
    require_selected_count: int | None,
    valid_len: int,
    dense_len: int | None,
    allow_short_valid_ratio_count: bool,
) -> int | None:
    return paction_budget_contract.expected_selected_count(
        require_selected_count,
        valid_len=int(valid_len),
        dense_len=int(dense_len or 0),
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
    )


def _selected_positions_from_sample(
    row: Mapping[str, Any],
    *,
    line_no: int,
    strategy: str,
    fallback_to_selected_positions: bool,
) -> list[int]:
    strategy_rows = row.get("strategy_selected_positions")
    if isinstance(strategy_rows, Mapping) and strategy in strategy_rows:
        return _int_positions(strategy_rows[strategy], name=f"line {line_no}: strategy_selected_positions.{strategy}")
    if fallback_to_selected_positions and "selected_positions" in row:
        return _int_positions(row["selected_positions"], name=f"line {line_no}: selected_positions")
    raise ValueError(f"line {line_no}: strategy '{strategy}' is missing from strategy_selected_positions")


def _policy_metadata(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    paction_policy = row.get("paction_policy")
    gas_vt_policy = row.get("gas_vt_policy")
    if isinstance(paction_policy, Mapping):
        return paction_policy
    if isinstance(gas_vt_policy, Mapping):
        return gas_vt_policy
    return None


def _budgeted_expanded_positions_from_policy(
    row: Mapping[str, Any],
    *,
    strategy: str,
    line_no: int,
) -> list[int] | None:
    policy_metadata = _policy_metadata(row)
    if not isinstance(policy_metadata, Mapping):
        return None
    by_strategy = policy_metadata.get("budgeted_expanded_positions_by_strategy")
    if not isinstance(by_strategy, Mapping) or str(strategy) not in by_strategy:
        return None
    return _int_positions(
        by_strategy[str(strategy)],
        name=f"line {line_no}: budgeted_expanded_positions_by_strategy.{strategy}",
    )


def _expanded_observation_contract(
    row: Mapping[str, Any],
    *,
    line_no: int,
    selected: Sequence[int],
    strategy: str,
    valid_len: int,
    expanded_budget: int | None,
) -> dict[str, Any]:
    policy_metadata = _policy_metadata(row)
    if not isinstance(policy_metadata, Mapping):
        return {}
    radii_by_strategy = policy_metadata.get("context_radius_by_strategy")
    if not isinstance(radii_by_strategy, Mapping) or str(strategy) not in radii_by_strategy:
        return {}
    dense_radii = radii_by_strategy[str(strategy)]
    if not isinstance(dense_radii, Sequence) or isinstance(dense_radii, (str, bytes, bytearray)):
        raise ValueError(f"context_radius_by_strategy.{strategy} must be a sequence")
    if len(dense_radii) < int(valid_len):
        raise ValueError(f"context_radius_by_strategy.{strategy} shorter than valid_len={valid_len}")
    radius_range = policy_metadata.get("context_radius_range")
    if not (
        isinstance(radius_range, Sequence)
        and not isinstance(radius_range, (str, bytes, bytearray))
        and len(radius_range) == 2
    ):
        radius_range = [0.0, 16.0]
    radius_min = float(radius_range[0])
    radius_max = float(radius_range[1])
    context_radius_float_by_position: list[float] = []
    context_radius_by_position: list[int] = []
    observations: list[dict[str, int]] = []
    expanded: set[int] = set()
    for center in selected:
        radius_float = max(radius_min, min(radius_max, float(dense_radii[int(center)])))
        radius = int(round(radius_float))
        radius = max(int(round(radius_min)), min(int(round(radius_max)), radius))
        start = max(0, int(center) - radius)
        end = min(int(valid_len) - 1, int(center) + radius)
        context_radius_float_by_position.append(float(radius_float))
        context_radius_by_position.append(int(radius))
        observations.append(
            {
                "center": int(center),
                "radius": int(radius),
                "expanded_start": int(start),
                "expanded_end": int(end),
            }
        )
        expanded.update(range(start, end + 1))
    budgeted_expanded_positions = _budgeted_expanded_positions_from_policy(
        row,
        strategy=strategy,
        line_no=line_no,
    )
    if budgeted_expanded_positions is not None:
        if any(position >= int(valid_len) for position in budgeted_expanded_positions):
            raise ValueError(f"line {line_no}: budgeted expanded positions exceed valid_len={valid_len}")
        if expanded_budget is not None and len(budgeted_expanded_positions) > int(expanded_budget):
            raise ValueError(
                f"line {line_no}: budgeted expanded count {len(budgeted_expanded_positions)} exceeds budget {int(expanded_budget)}"
            )
        expanded_positions = budgeted_expanded_positions
    else:
        expanded_positions = sorted(expanded)
        if expanded_budget is not None and str(strategy) in RADIUS_STRATEGIES and len(expanded_positions) > int(expanded_budget):
            raise ValueError(
                f"line {line_no}: adaptive radius expanded count {len(expanded_positions)} exceeds budget {int(expanded_budget)}"
            )
    budgeted_diagnostics_by_strategy = policy_metadata.get("budgeted_expanded_diagnostics_by_strategy")
    budgeted_diagnostics = (
        budgeted_diagnostics_by_strategy.get(str(strategy))
        if isinstance(budgeted_diagnostics_by_strategy, Mapping)
        else None
    )
    if not isinstance(budgeted_diagnostics, Mapping):
        budgeted_diagnostics = {}
    return {
        "selected_positions_are_centers": True,
        "context_radius_unit": str(policy_metadata.get("context_radius_unit") or "local_dense_snippet_index"),
        "context_radius_range": [float(radius_min), float(radius_max)],
        "context_radius_by_position": context_radius_by_position,
        "context_radius_float_by_position": context_radius_float_by_position,
        "selected_observations": observations,
        "expanded_selected_positions": expanded_positions,
        "expanded_selected_count": int(len(expanded_positions)),
        "budgeted_expanded_diagnostics": dict(budgeted_diagnostics),
    }


def sample_row_to_value_transport_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    strategy: str,
    target_len: int,
    require_selected_count: int | None = None,
    fill_to_target_count: bool = False,
    allow_short_valid_ratio_count: bool = False,
    fallback_to_selected_positions: bool = False,
    require_window_sample_id: bool = True,
    deploy_selection_ledger: bool = False,
    route_variant: str = "c3_lowres_probe_delta_p_action_original_adatad",
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
    _validate_sample_id(sample_id, line_no=line_no, require_window_sample_id=bool(require_window_sample_id))
    if deploy_selection_ledger and not require_window_sample_id:
        raise ValueError("deploy_selection_ledger requires strict video_name|window_start sample_id keys")
    if deploy_selection_ledger and fill_to_target_count:
        raise ValueError(
            "deploy_selection_ledger cannot use fill_to_target_count; "
            "gap control must come from the learned policy/loss, not uniform fill"
        )
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden source flag {key}=true")

    dense_len = _optional_int(row, "dense_len")
    valid_len = _optional_int(row, "valid_len")
    if valid_len is None:
        valid_len = dense_len
    if valid_len is None or valid_len <= 0:
        raise ValueError(f"line {line_no}: valid_len or dense_len must be positive")
    if dense_len is not None and valid_len > dense_len:
        raise ValueError(f"line {line_no}: valid_len cannot exceed dense_len")

    selected = _selected_positions_from_sample(
        row,
        line_no=line_no,
        strategy=strategy,
        fallback_to_selected_positions=bool(fallback_to_selected_positions),
    )
    if any(position >= valid_len for position in selected):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len={valid_len}")
    expected_required_count = _expected_required_count(
        require_selected_count=require_selected_count,
        valid_len=int(valid_len),
        dense_len=dense_len,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
    )
    has_budgeted_expanded_positions = _budgeted_expanded_positions_from_policy(
        row,
        strategy=strategy,
        line_no=line_no,
    ) is not None
    fill_count = 0
    if fill_to_target_count and expected_required_count is not None and len(selected) < int(expected_required_count):
        selected, fill_count = _uniform_fill_positions(
            selected,
            valid_len=int(valid_len),
            target_count=int(expected_required_count),
        )
    if (
        expected_required_count is not None
        and len(selected) != int(expected_required_count)
        and not has_budgeted_expanded_positions
    ):
        raise ValueError(
            f"line {line_no}: selected_count={len(selected)} does not match required count {int(expected_required_count)}"
        )

    diagnostics = {
        "source_probe_model": row.get("probe_model"),
        "source_tcn_variant": row.get("tcn_variant"),
        "source_spatial_size": row.get("spatial_size"),
        "source_strategy": str(strategy),
        "source_budget": row.get("budget"),
        "source_selected_count": len(_selected_positions_from_sample(
            row,
            line_no=line_no,
            strategy=strategy,
            fallback_to_selected_positions=bool(fallback_to_selected_positions),
        )),
        "uniform_visible_fill_count": int(fill_count),
        "required_selected_count": expected_required_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "fallback_to_selected_positions": bool(
            fallback_to_selected_positions
            and not (isinstance(row.get("strategy_selected_positions"), Mapping) and strategy in row["strategy_selected_positions"])
        ),
    }
    policy_metadata = _policy_metadata(row)
    if deploy_selection_ledger and not isinstance(policy_metadata, Mapping):
        raise ValueError(
            f"line {line_no}: paction_policy metadata is required for deploy selection ledger "
            "(gas_vt_policy is also accepted for GAS-VT rows)"
        )
    if isinstance(policy_metadata, Mapping):
        diagnostics["policy_family"] = policy_metadata.get("policy_family")
        diagnostics["policy_source"] = policy_metadata.get("source")
        diagnostics["policy_checkpoint_path"] = policy_metadata.get("checkpoint_path")
        diagnostics["policy_checkpoint_sha256"] = policy_metadata.get("checkpoint_sha256") or policy_metadata.get("policy_checkpoint_sha256")
        diagnostics["policy_fixed_budget"] = policy_metadata.get("fixed_budget") or policy_metadata.get("fixed_budgets")
        diagnostics["policy_dynamic_budget"] = policy_metadata.get("dynamic_budget")
        diagnostics["policy_uses_uniform_scaffold"] = policy_metadata.get("uses_uniform_scaffold")
        diagnostics["policy_uses_uniform_fill"] = policy_metadata.get("uses_uniform_fill")
        diagnostics["p_action_provenance"] = policy_metadata.get("p_action_provenance")
    observation_contract = _expanded_observation_contract(
        row,
        line_no=line_no,
        selected=selected,
        strategy=strategy,
        valid_len=int(valid_len),
        expanded_budget=expected_required_count,
    )
    if observation_contract:
        budgeted_diagnostics = observation_contract.pop("budgeted_expanded_diagnostics", {})
        if has_budgeted_expanded_positions and expected_required_count is not None:
            expanded_count = int(observation_contract["expanded_selected_count"])
            if expanded_count != int(expected_required_count):
                raise ValueError(
                    f"line {line_no}: expanded_selected_count={expanded_count} "
                    f"does not match required count {int(expected_required_count)}"
                )
        diagnostics.update(
            {
                "selected_positions_are_centers": True,
                "context_radius_unit": observation_contract["context_radius_unit"],
                "context_radius_range": observation_contract["context_radius_range"],
                "expanded_selected_count": observation_contract["expanded_selected_count"],
            }
        )
        for key in (
            "expanded_budget",
            "center_count",
            "budgeted_expanded_count",
            "budgeted_expanded_selection",
            "candidate_union_count",
            "expanded_score_source",
        ):
            if key in budgeted_diagnostics:
                diagnostics[key] = budgeted_diagnostics[key]
    if deploy_selection_ledger and isinstance(policy_metadata, Mapping):
        if policy_metadata.get("source") not in DEPLOY_CHECKPOINT_POLICY_SOURCES:
            raise ValueError(
                f"line {line_no}: deploy ledger requires checkpoint policy source "
                f"{sorted(DEPLOY_CHECKPOINT_POLICY_SOURCES)}, got {policy_metadata.get('source')}"
            )
        if not policy_metadata.get("checkpoint_path"):
            raise ValueError(f"line {line_no}: deploy ledger requires policy checkpoint_path")
        if not (policy_metadata.get("checkpoint_sha256") or policy_metadata.get("policy_checkpoint_sha256")):
            raise ValueError(f"line {line_no}: deploy ledger requires policy checkpoint_sha256")
        paction_source_samples.validate_paction_positive_provenance(
            policy_metadata.get("p_action_provenance"),
            source_name=f"line {line_no}: paction_policy/gas_vt_policy",
        )
    boundary_support = None if deploy_selection_ledger else _finite_float_or_none(row.get("boundary_support_r1"))
    if boundary_support is not None:
        diagnostics["diagnostic_boundary_support_r1_ignored_by_selection"] = boundary_support

    ledger_row = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "selected_positions_unit": "local_dense_index",
        "selected_positions": selected,
        "target_len": int(target_len),
        "selected_count": len(selected),
        "valid_len": int(valid_len),
        "dense_len": dense_len,
        "route": "C3_LOWRES_PROBE",
        "route_variant": str(route_variant),
        "policy": f"c3_lowres_probe_{strategy}",
        "policy_source": diagnostics.get("policy_source"),
        "policy_checkpoint_path": diagnostics.get("policy_checkpoint_path"),
        "policy_checkpoint_sha256": diagnostics.get("policy_checkpoint_sha256"),
        "source_schema_version": "c3_lowres_probe_samples_jsonl",
        "diagnostics": diagnostics,
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "diagnostic_only": not bool(deploy_selection_ledger),
        "training_only": False,
        "diagnostic_uses_train_utility_for_audit": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
    }
    ledger_row.update(observation_contract)
    validate_value_transport_selection_row(
        ledger_row,
        line_no=line_no,
        require_deployable=bool(deploy_selection_ledger),
    )
    return ledger_row


def run_conversion(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    strategy: str = "delta_p_action",
    target_len: int = 384,
    summary_json: str | Path | None = None,
    require_selected_count: int | None = None,
    fill_to_target_count: bool = False,
    allow_short_valid_ratio_count: bool = False,
    fallback_to_selected_positions: bool = False,
    require_window_sample_id: bool = True,
    deploy_selection_ledger: bool = False,
    deduplicate_sample_id: bool = False,
    route_variant: str = "c3_lowres_probe_delta_p_action_original_adatad",
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    if deploy_selection_ledger and not require_window_sample_id:
        raise ValueError("deploy_selection_ledger requires strict video_name|window_start sample_id keys")
    out_rows = [
        sample_row_to_value_transport_row(
            row,
            line_no=line_no,
            strategy=strategy,
            target_len=target_len,
            require_selected_count=require_selected_count,
            fill_to_target_count=fill_to_target_count,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            fallback_to_selected_positions=fallback_to_selected_positions,
            require_window_sample_id=require_window_sample_id,
            deploy_selection_ledger=deploy_selection_ledger,
            route_variant=route_variant,
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    duplicate_sample_id_count = 0
    if deduplicate_sample_id:
        deduped_rows: list[dict[str, Any]] = []
        by_sample_id: dict[str, dict[str, Any]] = {}
        for row in out_rows:
            sample_id = str(row["sample_id"])
            previous = by_sample_id.get(sample_id)
            if previous is None:
                by_sample_id[sample_id] = row
                deduped_rows.append(row)
                continue
            duplicate_sample_id_count += 1
            for key in (
                "selected_positions",
                "selected_count",
                "target_len",
                "valid_len",
                "dense_len",
                "policy_source",
                "policy_checkpoint_path",
                "policy_checkpoint_sha256",
            ):
                if previous.get(key) != row.get(key):
                    raise ValueError(f"duplicate sample_id {sample_id} has conflicting {key}")
            previous_provenance = (previous.get("diagnostics") or {}).get("p_action_provenance")
            current_provenance = (row.get("diagnostics") or {}).get("p_action_provenance")
            if previous_provenance != current_provenance:
                raise ValueError(f"duplicate sample_id {sample_id} has conflicting p_action_provenance")
        out_rows = deduped_rows
    sample_ids = [str(row["sample_id"]) for row in out_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("converted value-transport ledger has duplicate sample_id")
    _write_jsonl(output_jsonl, out_rows)
    counts = [int(row["selected_count"]) for row in out_rows]
    fill_counts = [int(row["diagnostics"]["uniform_visible_fill_count"]) for row in out_rows]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(out_rows),
        "strategy": str(strategy),
        "target_len": int(target_len),
        "require_selected_count": require_selected_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "fill_to_target_count": bool(fill_to_target_count),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "deduplicate_sample_id": bool(deduplicate_sample_id),
        "duplicate_sample_id_count": int(duplicate_sample_id_count),
        "route_variant": str(route_variant),
        "gap_control": "source_strategy_only_no_uniform_fill_for_deploy",
        "min_selected_count": min(counts),
        "max_selected_count": max(counts),
        "total_uniform_visible_fill_count": sum(fill_counts),
        "sample_ids_with_window_start": sum(1 for sample_id in sample_ids if "|" in sample_id),
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert C3 low-res probe samples to value-transport ledger rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--strategy", default="delta_p_action")
    parser.add_argument("--target-len", type=int, default=384)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--fill-to-target-count", action="store_true")
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--fallback-to-selected-positions", action="store_true")
    parser.add_argument("--allow-video-only-sample-id", action="store_true")
    parser.add_argument("--deploy-selection-ledger", action="store_true")
    parser.add_argument("--deduplicate-sample-id", action="store_true")
    parser.add_argument("--route-variant", default="c3_lowres_probe_delta_p_action_original_adatad")
    args = parser.parse_args(argv)

    try:
        if args.allow_video_only_sample_id and args.deploy_selection_ledger:
            raise ValueError("--allow-video-only-sample-id is diagnostic-only and cannot be used with --deploy-selection-ledger")
        summary = run_conversion(
            args.input_jsonl,
            args.output_jsonl,
            strategy=args.strategy,
            target_len=int(args.target_len),
            summary_json=args.summary_json,
            require_selected_count=args.require_selected_count,
            fill_to_target_count=bool(args.fill_to_target_count),
            allow_short_valid_ratio_count=bool(args.allow_short_valid_ratio_count),
            fallback_to_selected_positions=bool(args.fallback_to_selected_positions),
            require_window_sample_id=not bool(args.allow_video_only_sample_id),
            deploy_selection_ledger=bool(args.deploy_selection_ledger),
            deduplicate_sample_id=bool(args.deduplicate_sample_id),
            route_variant=args.route_variant,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
