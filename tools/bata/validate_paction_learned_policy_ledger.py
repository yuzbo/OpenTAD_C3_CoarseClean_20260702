from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import paction_budget_contract
from tools.bata import paction_source_samples


SUMMARY_SCHEMA_VERSION = "c3_paction_learned_policy_ledger_validation_v1"
READY = "C3_PACTION_LEARNED_POLICY_LEDGER_VALIDATION_PASS"
GAS_VT_CHECKPOINT_POLICY_SOURCE = "learned_paction_gas_vt_policy_checkpoint"
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


def _unselected_holes(selected: Sequence[int], *, valid_len: int) -> list[int]:
    selected_set = {int(item) for item in selected}
    holes: list[int] = []
    current = 0
    for idx in range(max(0, int(valid_len))):
        if idx in selected_set:
            if current > 0:
                holes.append(current)
            current = 0
        else:
            current += 1
    if current > 0:
        holes.append(current)
    return holes or [0]


def _uniform_reference_positions(*, valid_len: int, selected_count: int) -> list[int]:
    valid_len = int(valid_len)
    selected_count = int(selected_count)
    if valid_len <= 0 or selected_count <= 0:
        return []
    if selected_count >= valid_len:
        return list(range(valid_len))
    step = float(valid_len) / float(selected_count)
    positions = [int(round(float(idx) * step)) for idx in range(selected_count)]
    positions = [max(0, min(valid_len - 1, item)) for item in positions]
    if len(set(positions)) == len(positions):
        return sorted(positions)
    out: list[int] = []
    used: set[int] = set()
    for item in positions:
        candidate = item
        while candidate in used and candidate + 1 < valid_len:
            candidate += 1
        while candidate in used and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in used:
            used.add(candidate)
            out.append(candidate)
    return sorted(out)


def _uniform_similarity(selected: Sequence[int], *, valid_len: int) -> float:
    selected_set = {int(item) for item in selected}
    if not selected_set:
        return 0.0
    reference = set(_uniform_reference_positions(valid_len=int(valid_len), selected_count=len(selected_set)))
    return len(selected_set.intersection(reference)) / float(len(selected_set))


def _p95(values: Sequence[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(int(item) for item in values)
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values))) - 1))
    return float(sorted_values[index])


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(item) for item in values) / float(len(values))


def _median(sorted_values: Sequence[float]) -> float:
    if not sorted_values:
        return 0.0
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return (float(sorted_values[mid - 1]) + float(sorted_values[mid])) / 2.0


def _iqr(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    sorted_values = [float(item) for item in sorted(values)]
    mid = len(sorted_values) // 2
    lower = sorted_values[:mid] or sorted_values
    upper = sorted_values[mid + (len(sorted_values) % 2) :] or sorted_values
    return _median(upper) - _median(lower)


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def _entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    total = float(len(values))
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / total
        entropy -= probability * math.log(probability, 2)
    return entropy


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


def _p_action_values(sample_row: Mapping[str, Any]) -> list[float]:
    frame_signals = sample_row.get("frame_signals")
    if isinstance(frame_signals, Mapping) and isinstance(frame_signals.get("p_action"), list):
        return [float(item) for item in frame_signals["p_action"]]
    if isinstance(sample_row.get("p_action"), list):
        return [float(item) for item in sample_row["p_action"]]
    return []


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), idx))
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        rank = (cursor + end - 1) / 2.0
        for idx in order[cursor:end]:
            ranks[idx] = rank
        cursor = end
    return ranks


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_ranks = _rankdata(x_values)
    y_ranks = _rankdata(y_values)
    x_mean = sum(x_ranks) / float(len(x_ranks))
    y_mean = sum(y_ranks) / float(len(y_ranks))
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks))
    x_var = sum((x - x_mean) ** 2 for x in x_ranks)
    y_var = sum((y - y_mean) ** 2 for y in y_ranks)
    denom = math.sqrt(x_var * y_var)
    if denom <= 0:
        return None
    return numerator / denom


def _action_interior_bin_coverage(action_target: Sequence[float], selected: Sequence[int], *, valid_len: int, bins: int = 4) -> tuple[int, int]:
    selected_set = {int(item) for item in selected}
    covered = 0
    total = 0
    idx = 0
    valid_target = [float(item) for item in action_target[: int(valid_len)]]
    while idx < len(valid_target):
        if valid_target[idx] < 0.5:
            idx += 1
            continue
        start = idx
        while idx < len(valid_target) and valid_target[idx] >= 0.5:
            idx += 1
        end = idx - 1
        width = end - start + 1
        bin_count = min(int(bins), width)
        for bin_idx in range(bin_count):
            left = start + int(math.floor(bin_idx * width / float(bin_count)))
            right = start + int(math.ceil((bin_idx + 1) * width / float(bin_count))) - 1
            total += 1
            if any(left <= pos <= right for pos in selected_set):
                covered += 1
    return covered, total


def _validate_paction_positive_provenance(
    paction_policy: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    provenance = paction_policy.get("p_action_provenance")
    paction_source_samples.validate_paction_positive_provenance(provenance, source_name=source_name)


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
    boundary_radii: Sequence[int] | None = None,
    min_boundary_support: float | None = None,
    min_action_coverage: float | None = None,
    max_max_gap: int | None = None,
    max_p95_gap: float | None = None,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    require_policy_source: str | None = None,
    require_checkpoint_path: str | Path | None = None,
    require_checkpoint_sha256: str | None = None,
    require_paction_provenance: bool = False,
    summary_json: str | Path | None = None,
    max_hole_top10_csv: str | Path | None = None,
) -> dict[str, Any]:
    sample_by_id = _sample_map(_read_jsonl(sample_jsonl))
    metric_by_id = _sample_map(_read_jsonl(metric_sample_jsonl)) if metric_sample_jsonl is not None else sample_by_id
    ledger_rows = _read_jsonl(ledger_jsonl)
    seen: set[str] = set()
    selected_counts: list[int] = []
    all_gaps: list[int] = []
    all_unselected_holes: list[int] = []
    uniform_similarities: list[float] = []
    max_gap = 0
    max_hole = 0
    boundary_hits = 0
    boundary_total = 0
    radii = [int(item) for item in (boundary_radii if boundary_radii is not None else [boundary_radius])]
    if not radii:
        radii = [int(boundary_radius)]
    radii = sorted(set(radii))
    boundary_hits_by_radius = {int(radius): 0 for radius in radii}
    boundary_bracket_hits_by_radius = {int(radius): 0 for radius in radii}
    action_selected = 0
    action_total = 0
    action_bin_selected = 0
    action_bin_total = 0
    spearman_values: list[float] = []
    p_action_topk_jaccards: list[float] = []
    p_action_topk_overlap_ratios: list[float] = []
    hole_rows: list[dict[str, Any]] = []
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
        gas_vt_policy = sample_row.get("gas_vt_policy")
        policy_metadata = paction_policy if isinstance(paction_policy, Mapping) else gas_vt_policy
        if isinstance(policy_metadata, Mapping):
            if policy_metadata.get("uses_uniform_fill") is not False:
                raise ValueError(f"{sample_jsonl}:{sample_id}: policy uses_uniform_fill must be false")
            if policy_metadata.get("uses_uniform_scaffold") is not False:
                raise ValueError(f"{sample_jsonl}:{sample_id}: policy uses_uniform_scaffold must be false")
            if require_policy_source is not None and policy_metadata.get("source") != str(require_policy_source):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.source must be {require_policy_source}"
                )
            if require_checkpoint_path is not None and str(policy_metadata.get("checkpoint_path")) != str(require_checkpoint_path):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.checkpoint_path must be {require_checkpoint_path}"
                )
            metadata_sha = policy_metadata.get("checkpoint_sha256") or policy_metadata.get("policy_checkpoint_sha256")
            if require_checkpoint_sha256 is not None and metadata_sha != str(require_checkpoint_sha256):
                raise ValueError(
                    f"{sample_jsonl}:{sample_id}: paction_policy.checkpoint_sha256 mismatch"
                )
            if require_paction_provenance:
                _validate_paction_positive_provenance(
                    policy_metadata,
                    source_name=f"{sample_jsonl}:{sample_id}",
                )
        elif require_policy_source is not None or require_checkpoint_path is not None or require_paction_provenance:
            raise ValueError(f"{sample_jsonl}:{sample_id}: paction_policy metadata is required")
        selected = _positions(row.get("selected_positions"), name=f"{ledger_jsonl}:{line_no}: selected_positions")
        valid_len = int(row.get("valid_len"))
        dense_len = int(row.get("dense_len") or valid_len)
        if any(item >= valid_len for item in selected):
            raise ValueError(f"{ledger_jsonl}:{line_no}: selected position outside valid_len")
        if expected_target_len is not None and int(row.get("target_len")) != int(expected_target_len):
            raise ValueError(f"{ledger_jsonl}:{line_no}: target_len must be {expected_target_len}")
        expected_count = paction_budget_contract.expected_selected_count(
            require_selected_count,
            valid_len=valid_len,
            dense_len=dense_len,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        )
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
        holes = _unselected_holes(selected, valid_len=valid_len)
        all_unselected_holes.extend(holes)
        max_hole = max(max_hole, max(holes) if holes else 0)
        row_max_hole = max(holes) if holes else 0
        video_name = str(sample_id).split("|", 1)[0]
        hole_rows.append(
            {
                "video_name": video_name,
                "sample_id": str(sample_id),
                "max_unselected_hole": int(row_max_hole),
                "selected_count": len(selected),
                "valid_len": int(valid_len),
            }
        )
        uniform_similarities.append(_uniform_similarity(selected, valid_len=valid_len))
        selected_counts.append(len(selected))
        boundaries = _boundaries(metric_row)
        boundary_total += len(boundaries)
        selected_float = [float(item) for item in selected]
        for boundary in boundaries:
            if any(abs(item - boundary) <= float(boundary_radius) for item in selected_float):
                boundary_hits += 1
            for radius in radii:
                if any(abs(item - boundary) <= float(radius) for item in selected_float):
                    boundary_hits_by_radius[int(radius)] += 1
                left_hit = any((boundary - float(radius)) <= item < boundary for item in selected_float)
                right_hit = any(boundary < item <= (boundary + float(radius)) for item in selected_float)
                if left_hit and right_hit:
                    boundary_bracket_hits_by_radius[int(radius)] += 1
        target = _action_target(metric_row)
        if target:
            valid_target = target[:valid_len]
            positive = {idx for idx, value in enumerate(valid_target) if float(value) >= 0.5}
            action_total += len(positive)
            action_selected += len(positive.intersection(set(selected)))
            bin_hit, bin_total = _action_interior_bin_coverage(valid_target, selected, valid_len=valid_len)
            action_bin_selected += bin_hit
            action_bin_total += bin_total
        p_values = _p_action_values(metric_row)
        if p_values:
            selected_mask = [1.0 if idx in set(selected) else 0.0 for idx in range(min(valid_len, len(p_values)))]
            corr = _spearman(p_values[: len(selected_mask)], selected_mask)
            if corr is not None:
                spearman_values.append(float(corr))
            if selected_mask and selected:
                topk_count = min(len(selected), len(selected_mask))
                ranked = sorted(
                    range(len(selected_mask)),
                    key=lambda idx: (float(p_values[idx]), -int(idx)),
                    reverse=True,
                )[:topk_count]
                topk_set = {int(idx) for idx in ranked}
                selected_set = {int(idx) for idx in selected if 0 <= int(idx) < len(selected_mask)}
                intersection = len(selected_set.intersection(topk_set))
                union = len(selected_set.union(topk_set))
                if union > 0:
                    p_action_topk_jaccards.append(intersection / float(union))
                if topk_count > 0:
                    p_action_topk_overlap_ratios.append(intersection / float(topk_count))
    boundary_support = None if boundary_total <= 0 else boundary_hits / float(boundary_total)
    boundary_support_by_radius = {
        int(radius): None if boundary_total <= 0 else hits / float(boundary_total)
        for radius, hits in boundary_hits_by_radius.items()
    }
    boundary_bracket_support_by_radius = {
        int(radius): None if boundary_total <= 0 else hits / float(boundary_total)
        for radius, hits in boundary_bracket_hits_by_radius.items()
    }
    action_coverage = None if action_total <= 0 else action_selected / float(action_total)
    action_interior_bin_coverage = None if action_bin_total <= 0 else action_bin_selected / float(action_bin_total)
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
    p95_unselected_hole = _p95(all_unselected_holes)
    if max_unselected_hole is not None and int(max_hole) > int(max_unselected_hole):
        raise ValueError(f"max_unselected_hole above threshold: {max_hole}")
    if (
        max_p95_unselected_hole is not None
        and p95_unselected_hole is not None
        and float(p95_unselected_hole) > float(max_p95_unselected_hole)
    ):
        raise ValueError(f"p95_unselected_hole above threshold: {p95_unselected_hole}")
    mean_uniform_similarity = _mean(uniform_similarities)
    max_observed_uniform_similarity = max(uniform_similarities) if uniform_similarities else None
    if (
        max_uniform_similarity is not None
        and max_observed_uniform_similarity is not None
        and float(max_observed_uniform_similarity) > float(max_uniform_similarity)
    ):
        raise ValueError(f"uniform similarity above threshold: {max_observed_uniform_similarity}")
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
        "selected_count_histogram": _count_histogram(selected_counts),
        "max_gap": int(max_gap),
        "p95_gap": p95_gap,
        "max_unselected_hole": int(max_hole),
        "p95_unselected_hole": p95_unselected_hole,
        "mean_uniform_similarity": mean_uniform_similarity,
        "meanK_matched_uniform_similarity": mean_uniform_similarity,
        "max_uniform_similarity": max_observed_uniform_similarity,
        f"boundary_support_r{int(boundary_radius)}": boundary_support,
        f"boundary_support@r{int(boundary_radius)}": boundary_support,
        "action_positive_coverage": action_coverage,
        "action_interior_bin_coverage": action_interior_bin_coverage,
        "p_action_rank_spearman": _mean(spearman_values),
        "p_action_topk_jaccard": _mean(p_action_topk_jaccards),
        "p_action_topk_overlap_ratio": _mean(p_action_topk_overlap_ratios),
        "dynamic_budget_entropy": _entropy(selected_counts),
        "dynamic_budget_iqr": _iqr(selected_counts),
        "max_hole_by_video_top10": sorted(
            hole_rows,
            key=lambda item: (int(item["max_unselected_hole"]), str(item["sample_id"])),
            reverse=True,
        )[:10],
        "total_uniform_visible_fill_count": int(total_uniform_fill),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "required_policy_source": require_policy_source,
        "required_checkpoint_path": None if require_checkpoint_path is None else str(require_checkpoint_path),
        "required_checkpoint_sha256": require_checkpoint_sha256,
        "actual_checkpoint_sha256": checkpoint_sha256,
        "require_paction_provenance": bool(require_paction_provenance),
        "paction_provenance_verified": bool(require_paction_provenance),
        "gap_metric_definition": "endpoint_inclusive_selected_stride_gap",
        "hole_metric_definition": "max_contiguous_unselected_dense_positions_between_selected_frames",
        "uniform_similarity_definition": "intersection_over_selected_count_against_round_i_times_valid_len_over_k",
        "p_action_topk_metric_definition": "selected_set_overlap_with_same_k_top_p_action_frames",
    }
    for radius, value in boundary_support_by_radius.items():
        summary[f"boundary_support_r{int(radius)}"] = value
        summary[f"boundary_support@r{int(radius)}"] = value
    for radius, value in boundary_bracket_support_by_radius.items():
        summary[f"boundary_bracket_support_r{int(radius)}"] = value
        summary[f"boundary_bracket_support@r{int(radius)}"] = value
    if max_hole_top10_csv is not None:
        csv_path = Path(max_hole_top10_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        top_rows = summary["max_hole_by_video_top10"]
        lines = ["video_name,sample_id,max_unselected_hole,selected_count,valid_len"]
        for item in top_rows:
            lines.append(
                f"{item['video_name']},{item['sample_id']},{item['max_unselected_hole']},{item['selected_count']},{item['valid_len']}"
            )
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--boundary-radii", type=int, nargs="+")
    parser.add_argument("--min-boundary-support", type=float)
    parser.add_argument("--min-action-coverage", type=float)
    parser.add_argument("--max-max-gap", type=int)
    parser.add_argument("--max-p95-gap", type=float)
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--require-policy-source")
    parser.add_argument("--require-checkpoint-path")
    parser.add_argument("--require-checkpoint-sha256")
    parser.add_argument("--require-paction-provenance", action="store_true")
    parser.add_argument("--summary-json")
    parser.add_argument("--max-hole-top10-csv")
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
        boundary_radii=args.boundary_radii,
        min_boundary_support=args.min_boundary_support,
        min_action_coverage=args.min_action_coverage,
        max_max_gap=args.max_max_gap,
        max_p95_gap=args.max_p95_gap,
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        require_policy_source=args.require_policy_source,
        require_checkpoint_path=args.require_checkpoint_path,
        require_checkpoint_sha256=args.require_checkpoint_sha256,
        require_paction_provenance=bool(args.require_paction_provenance),
        summary_json=args.summary_json,
        max_hole_top10_csv=args.max_hole_top10_csv,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
