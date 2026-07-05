from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = "c3_paction_learned_policy_ledger_validation_v1"
READY = "C3_PACTION_LEARNED_POLICY_LEDGER_VALIDATION_PASS"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
    "uses_gt_for_diagnostics",
    "training_only",
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
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


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


def _positions(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    out = [int(item) for item in value]
    if out != sorted(out):
        raise ValueError(f"{name} must be sorted")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique")
    if any(item < 0 for item in out):
        raise ValueError(f"{name} must be non-negative")
    return out


def _sample_map(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in sample_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample row is missing sample_id")
        if sample_id in out:
            raise ValueError(f"duplicate sample_id in sample rows: {sample_id}")
        out[sample_id] = row
    return out


def _gap_values(selected: Sequence[int], *, valid_len: int) -> list[int]:
    if not selected:
        return [int(valid_len)]
    gaps: list[int] = []
    previous = -1
    for item in selected:
        gaps.append(int(item) - int(previous))
        previous = int(item)
    gaps.append(int(valid_len) - int(previous))
    return gaps


def _p95(values: Sequence[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(int(item) for item in values)
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values))) - 1))
    return float(sorted_values[index])


def _boundaries(sample_row: Mapping[str, Any]) -> list[float]:
    raw = sample_row.get("gt_boundaries")
    if raw is None:
        raw = sample_row.get("boundaries")
    if isinstance(raw, list):
        return [float(item) for item in raw]
    out: list[float] = []
    if isinstance(sample_row.get("gt_segments"), list):
        for segment in sample_row["gt_segments"]:
            if isinstance(segment, list) and len(segment) >= 2:
                out.extend([float(segment[0]), float(segment[1])])
    return out


def _action_target(sample_row: Mapping[str, Any]) -> list[float]:
    raw = sample_row.get("action_target")
    if raw is None:
        raw = sample_row.get("action_labels")
    if isinstance(raw, list):
        return [float(item) for item in raw]
    return []


def validate_ledger(
    *,
    sample_jsonl: str | Path,
    ledger_jsonl: str | Path,
    strategy: str,
    metric_sample_jsonl: str | Path | None = None,
    expected_target_len: int | None = None,
    require_selected_count: int | None = None,
    allow_short_valid_ratio_count: bool = False,
    require_nonconstant_selected_count: bool = False,
    require_deployable: bool = True,
    boundary_radius: int = 1,
    min_boundary_support: float | None = None,
    min_action_coverage: float | None = None,
    max_max_gap: int | None = None,
    max_p95_gap: float | None = None,
    require_policy_source: str | None = None,
    require_checkpoint_path: str | Path | None = None,
    require_checkpoint_sha256: str | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    sample_by_id = _sample_map(_read_jsonl(sample_jsonl))
    metric_by_id = _sample_map(_read_jsonl(metric_sample_jsonl)) if metric_sample_jsonl is not None else sample_by_id
    ledger_rows = _read_jsonl(ledger_jsonl)
    seen: set[str] = set()
    selected_counts: list[int] = []
    all_gaps: list[int] = []
    max_gap = 0
    boundary_hits = 0
    boundary_total = 0
    action_selected = 0
    action_total = 0
    total_uniform_fill = 0
    for line_no, row in enumerate(ledger_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{ledger_jsonl}:{line_no}: missing sample_id")
        if sample_id in seen:
            raise ValueError(f"{ledger_jsonl}:{line_no}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        if sample_id not in sample_by_id:
            raise ValueError(f"{ledger_jsonl}:{line_no}: sample_id not found in sample_jsonl: {sample_id}")
        if sample_id not in metric_by_id:
            raise ValueError(f"{ledger_jsonl}:{line_no}: sample_id not found in metric_sample_jsonl: {sample_id}")
        if require_deployable:
            if row.get("deploy_selection_ledger") is not True:
                raise ValueError(f"{ledger_jsonl}:{line_no}: deploy_selection_ledger must be true")
            if row.get("diagnostic_only") is True:
                raise ValueError(f"{ledger_jsonl}:{line_no}: diagnostic_only ledger row is not deployable")
        for key in FORBIDDEN_TRUE_FLAGS:
            if _is_true(row.get(key, False)):
                raise ValueError(f"{ledger_jsonl}:{line_no}: forbidden flag {key}=true")
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), Mapping) else {}
        fill_count = int(diagnostics.get("uniform_visible_fill_count", 0) or 0)
        if fill_count != 0:
            raise ValueError(f"{ledger_jsonl}:{line_no}: uniform_visible_fill_count must be 0")
        total_uniform_fill += fill_count
        if str(diagnostics.get("source_strategy")) != str(strategy):
            raise ValueError(f"{ledger_jsonl}:{line_no}: source_strategy must be {strategy}")
        ledger_policy_source = row.get("policy_source", diagnostics.get("policy_source"))
        ledger_checkpoint_path = row.get("policy_checkpoint_path", diagnostics.get("policy_checkpoint_path"))
        ledger_checkpoint_sha256 = row.get("policy_checkpoint_sha256", diagnostics.get("policy_checkpoint_sha256"))
        if require_policy_source is not None and ledger_policy_source != str(require_policy_source):
            raise ValueError(f"{ledger_jsonl}:{line_no}: policy_source must be {require_policy_source}")
        if require_checkpoint_path is not None and str(ledger_checkpoint_path) != str(require_checkpoint_path):
            raise ValueError(f"{ledger_jsonl}:{line_no}: policy_checkpoint_path must be {require_checkpoint_path}")
        if require_checkpoint_sha256 is not None and ledger_checkpoint_sha256 != str(require_checkpoint_sha256):
            raise ValueError(f"{ledger_jsonl}:{line_no}: policy_checkpoint_sha256 mismatch")
        sample_row = sample_by_id[sample_id]
        metric_row = metric_by_id[sample_id]
        for key in FORBIDDEN_TRUE_FLAGS:
            if _is_true(sample_row.get(key, False)):
                raise ValueError(f"{sample_jsonl}:{sample_id}: forbidden sample flag {key}=true")
        paction_policy = sample_row.get("paction_policy")
        if isinstance(paction_policy, Mapping):
            if paction_policy.get("uses_uniform_fill") is not False:
                raise ValueError(f"{sample_jsonl}:{sample_id}: paction_policy uses_uniform_fill must be false")
            if paction_policy.get("uses_uniform_scaffold") is not False:
                raise ValueError(f"{sample_jsonl}:{sample_id}: paction_policy uses_uniform_scaffold must be false")
            if require_policy_source is not None and paction_policy.get("source") != str(require_policy_source):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.source must be {require_policy_source}"
                )
            if require_checkpoint_path is not None and str(paction_policy.get("checkpoint_path")) != str(require_checkpoint_path):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.checkpoint_path must be {require_checkpoint_path}"
                )
            if require_checkpoint_sha256 is not None and paction_policy.get("checkpoint_sha256") != str(require_checkpoint_sha256):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.checkpoint_sha256 mismatch"
                )
        elif require_policy_source is not None or require_checkpoint_path is not None:
            raise ValueError(f"{sample_jsonl}:{sample_id}: paction_policy metadata is required")
        selected = _positions(row.get("selected_positions"), name=f"{ledger_jsonl}:{line_no}: selected_positions")
        valid_len = int(row.get("valid_len"))
        dense_len = int(row.get("dense_len") or valid_len)
        if any(item >= valid_len for item in selected):
            raise ValueError(f"{ledger_jsonl}:{line_no}: selected position outside valid_len")
        if expected_target_len is not None and int(row.get("target_len")) != int(expected_target_len):
            raise ValueError(f"{ledger_jsonl}:{line_no}: target_len must be {expected_target_len}")
        expected_count = require_selected_count
        if expected_count is not None and allow_short_valid_ratio_count and valid_len < dense_len:
            expected_count = max(1, min(int(expected_count), valid_len, int(math.ceil(valid_len * float(expected_count) / float(dense_len)))))
        if expected_count is not None and len(selected) != int(expected_count):
            raise ValueError(f"{ledger_jsonl}:{line_no}: selected_count must be {expected_count}")
        if int(row.get("selected_count")) != len(selected):
            raise ValueError(f"{ledger_jsonl}:{line_no}: selected_count mismatch")
        sample_strategies = sample_row.get("strategy_selected_positions")
        if isinstance(sample_strategies, Mapping) and strategy in sample_strategies:
            source_selected = _positions(sample_strategies[strategy], name=f"{sample_jsonl}:{sample_id}: {strategy}")
            if source_selected != selected:
                raise ValueError(f"{ledger_jsonl}:{line_no}: ledger positions do not match source strategy")
        gaps = _gap_values(selected, valid_len=valid_len)
        all_gaps.extend(gaps)
        max_gap = max(max_gap, max(gaps) if gaps else 0)
        selected_counts.append(len(selected))
        boundaries = _boundaries(metric_row)
        boundary_total += len(boundaries)
        selected_float = [float(item) for item in selected]
        for boundary in boundaries:
            if any(abs(item - boundary) <= float(boundary_radius) for item in selected_float):
                boundary_hits += 1
        target = _action_target(metric_row)
        if target:
            valid_target = target[:valid_len]
            positive = {idx for idx, value in enumerate(valid_target) if float(value) >= 0.5}
            action_total += len(positive)
            action_selected += len(positive.intersection(set(selected)))
    boundary_support = None if boundary_total <= 0 else boundary_hits / float(boundary_total)
    action_coverage = None if action_total <= 0 else action_selected / float(action_total)
    if require_nonconstant_selected_count and len(set(selected_counts)) <= 1:
        raise ValueError("selected_count is constant; dynamic budget ledger is degenerate")
    if min_boundary_support is not None and boundary_support is not None and boundary_support < float(min_boundary_support):
        raise ValueError(f"boundary_support_r{int(boundary_radius)} below threshold: {boundary_support}")
    if min_action_coverage is not None and action_coverage is not None and action_coverage < float(min_action_coverage):
        raise ValueError(f"action_positive_coverage below threshold: {action_coverage}")
    p95_gap = _p95(all_gaps)
    if max_max_gap is not None and int(max_gap) > int(max_max_gap):
        raise ValueError(f"max_gap above threshold: {max_gap}")
    if max_p95_gap is not None and p95_gap is not None and float(p95_gap) > float(max_p95_gap):
        raise ValueError(f"p95_gap above threshold: {p95_gap}")
    checkpoint_sha256 = None
    if require_checkpoint_path is not None:
        checkpoint_path = Path(require_checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise ValueError(f"required checkpoint missing: {checkpoint_path}")
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if require_checkpoint_sha256 is not None and checkpoint_sha256 != str(require_checkpoint_sha256):
            raise ValueError("required checkpoint sha256 mismatch")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "sample_jsonl": str(sample_jsonl),
        "metric_sample_jsonl": None if metric_sample_jsonl is None else str(metric_sample_jsonl),
        "ledger_jsonl": str(ledger_jsonl),
        "strategy": str(strategy),
        "row_count": len(ledger_rows),
        "expected_target_len": expected_target_len,
        "require_selected_count": require_selected_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "require_nonconstant_selected_count": bool(require_nonconstant_selected_count),
        "require_deployable": bool(require_deployable),
        "min_selected_count": min(selected_counts),
        "max_selected_count": max(selected_counts),
        "mean_selected_count": sum(selected_counts) / float(len(selected_counts)),
        "max_gap": int(max_gap),
        "p95_gap": p95_gap,
        f"boundary_support_r{int(boundary_radius)}": boundary_support,
        "action_positive_coverage": action_coverage,
        "total_uniform_visible_fill_count": int(total_uniform_fill),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "required_policy_source": require_policy_source,
        "required_checkpoint_path": None if require_checkpoint_path is None else str(require_checkpoint_path),
        "required_checkpoint_sha256": require_checkpoint_sha256,
        "actual_checkpoint_sha256": checkpoint_sha256,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate no-uniform learned p_action policy value-transport ledgers.")
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--ledger-jsonl", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--metric-sample-jsonl")
    parser.add_argument("--expected-target-len", type=int)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--require-nonconstant-selected-count", action="store_true")
    parser.add_argument("--require-deployable", action="store_true")
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--min-boundary-support", type=float)
    parser.add_argument("--min-action-coverage", type=float)
    parser.add_argument("--max-max-gap", type=int)
    parser.add_argument("--max-p95-gap", type=float)
    parser.add_argument("--require-policy-source")
    parser.add_argument("--require-checkpoint-path")
    parser.add_argument("--require-checkpoint-sha256")
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)
    summary = validate_ledger(
        sample_jsonl=args.sample_jsonl,
        ledger_jsonl=args.ledger_jsonl,
        strategy=args.strategy,
        metric_sample_jsonl=args.metric_sample_jsonl,
        expected_target_len=args.expected_target_len,
        require_selected_count=args.require_selected_count,
        allow_short_valid_ratio_count=bool(args.allow_short_valid_ratio_count),
        require_nonconstant_selected_count=bool(args.require_nonconstant_selected_count),
        require_deployable=bool(args.require_deployable),
        boundary_radius=int(args.boundary_radius),
        min_boundary_support=args.min_boundary_support,
        min_action_coverage=args.min_action_coverage,
        max_max_gap=args.max_max_gap,
        max_p95_gap=args.max_p95_gap,
        require_policy_source=args.require_policy_source,
        require_checkpoint_path=args.require_checkpoint_path,
        require_checkpoint_sha256=args.require_checkpoint_sha256,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
