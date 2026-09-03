from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_CSV = (
    Path("analysis_outputs")
    / "c3_completed_paction_candidate_benchmark_20260704"
    / "completed_paction_candidate_summary_20260704.csv"
)
DEFAULT_SELECTION_CSV = (
    Path("analysis_outputs")
    / "c3_completed_coarse_selection_oracle_style_20260704"
    / "completed_coarse_selection_all_strategies_20260704.csv"
)
DEFAULT_SAMPLE_JSON = (
    Path("analysis_outputs")
    / "c3_completed_coarse_selection_oracle_style_20260704"
    / "completed_selection_oracle_style_data.json"
)
DEFAULT_OUTPUT_DIR = Path("analysis_outputs") / "c3_probe_true_performance_figures_20260708"


def _short_label(value: Any) -> str:
    text = str(value)
    text = text.replace("temporal_tcn_", "")
    text = text.replace("separable_", "sep_")
    text = text.replace("causal_", "causal_")
    return text


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _scale01(values: Sequence[Any], length: int | None = None) -> np.ndarray:
    raw = list(values)
    if length is not None:
        raw = raw[: max(0, int(length))]
    arr = np.asarray([_float_or_nan(item) for item in raw], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _sample_signal(sample: Mapping[str, Any], key: str, *fallback_keys: str) -> Sequence[Any]:
    frame_signals = sample.get("frame_signals")
    for candidate in (key, *fallback_keys):
        values = sample.get(candidate)
        if values is None and isinstance(frame_signals, Mapping):
            values = frame_signals.get(candidate)
        if values is not None and not isinstance(values, (str, bytes)) and isinstance(values, Sequence):
            return values
    return []


def _load_dataframe(path: str | Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    return df


def _write_manifest(output_dir: Path, prefix: str, manifest: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_figure(manifest_figures: list[dict[str, Any]], path: Path, *, title: str, kind: str) -> None:
    manifest_figures.append(
        {
            "kind": kind,
            "title": title,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _top_candidate_models(candidate_df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("candidate_score", "top20_r4", "ap") if column in candidate_df.columns]
    if not sort_columns:
        return candidate_df.copy()
    return candidate_df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)


def _delta_selection_rows(selection_df: pd.DataFrame) -> pd.DataFrame:
    delta = selection_df[selection_df["strategy"].astype(str) == "delta_p_action"].copy()
    if delta.empty:
        raise ValueError("selection CSV contains no delta_p_action rows")
    for column in (
        "boundary_support_r1_global",
        "action_coverage_global",
        "p95_window_max_empty_gap",
        "mean_valid_len",
        "probe_ap",
    ):
        delta[column] = pd.to_numeric(delta[column], errors="coerce")
    gap_quality = 1.0 - (delta["p95_window_max_empty_gap"] / delta["mean_valid_len"].replace(0, np.nan))
    delta["gap_quality"] = gap_quality.clip(0.0, 1.0)
    delta["selection_score"] = (
        0.35 * delta["boundary_support_r1_global"]
        + 0.20 * delta["action_coverage_global"]
        + 0.30 * delta["gap_quality"]
        + 0.15 * delta["probe_ap"]
    )
    return delta.sort_values(["selection_score", "boundary_support_r1_global"], ascending=[False, False]).reset_index(
        drop=True
    )


def _plot_classifier_metrics(candidate_df: pd.DataFrame, output_dir: Path, prefix: str) -> Path:
    df = _top_candidate_models(candidate_df)
    labels = [_short_label(item) for item in df["model"]]
    x = np.arange(len(df))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=170)
    fig.suptitle("C3 coarse classifier signal quality", fontsize=15, fontweight="bold")

    ax = axes[0]
    for offset, column, label, color in (
        (-width, "ap", "AP", "#2563eb"),
        (0.0, "auc", "ROC-AUC", "#f97316"),
        (width, "best_f1", "best F1", "#16a34a"),
    ):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.012, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, max(0.75, float(np.nanmax(df[["ap", "auc", "best_f1"]].to_numpy(dtype=float))) + 0.08))
    ax.set_title("Frame-level action/background")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for offset, column, label, color in (
        (-width / 2.0, "action_bg_gap", "action-bg gap", "#0284c7"),
        (width / 2.0, "change_lift", "boundary change lift", "#dc2626"),
    ):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.88)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.004, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Signal separation")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ranked = df.sort_values("candidate_score", ascending=True)
    y = np.arange(len(ranked))
    values = pd.to_numeric(ranked["candidate_score"], errors="coerce").to_numpy(dtype=float)
    ax.barh(y, values, color=plt.cm.viridis(np.linspace(0.25, 0.85, len(ranked))))
    ax.set_yticks(y, [_short_label(item) for item in ranked["model"]], fontsize=8)
    ax.set_xlim(0.0, max(float(np.nanmax(values)) * 1.2, 0.1))
    ax.set_title("Candidate ranking")
    ax.set_xlabel("Composite candidate score")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        if np.isfinite(value):
            ax.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / f"{prefix}_coarse_classifier_metrics.png"
    _save(fig, path)
    return path


def _plot_delta_selection_quality(delta_df: pd.DataFrame, output_dir: Path, prefix: str) -> Path:
    df = delta_df.copy()
    labels = [_short_label(item) for item in df["model"]]
    x = np.arange(len(df))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=170)
    fig.suptitle("Indirect frame selection quality: delta_p_action", fontsize=15, fontweight="bold")

    ax = axes[0]
    for offset, column, label, color in (
        (-width / 2.0, "boundary_support_r1_global", "Boundary support r=1", "#7c3aed"),
        (width / 2.0, "action_coverage_global", "Action coverage", "#0f766e"),
    ):
        values = df[column].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, max(0.65, float(np.nanmax(df[["boundary_support_r1_global", "action_coverage_global"]].to_numpy())) + 0.08))
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage under matched budget")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    values = df["p95_window_max_empty_gap"].to_numpy(dtype=float)
    ax.bar(x, values, color="#ef4444", alpha=0.82)
    for idx, value in enumerate(values):
        if np.isfinite(value):
            ax.text(idx, value + max(values) * 0.015, f"{value:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Large-hole risk")
    ax.set_ylabel("p95 window max empty gap")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ranked = df.sort_values("selection_score", ascending=True)
    y = np.arange(len(ranked))
    values = ranked["selection_score"].to_numpy(dtype=float)
    ax.barh(y, values, color=plt.cm.cividis(np.linspace(0.25, 0.85, len(ranked))))
    ax.set_yticks(y, [_short_label(item) for item in ranked["model"]], fontsize=8)
    ax.set_xlim(0.0, max(float(np.nanmax(values)) * 1.22, 0.1))
    ax.set_title("Selection score")
    ax.set_xlabel("0.35 boundary + 0.20 action + 0.30 gap + 0.15 AP")
    ax.grid(axis="x", alpha=0.25)
    for idx, (_, row) in enumerate(ranked.iterrows()):
        ax.text(
            float(row["selection_score"]) + 0.005,
            idx,
            f"{float(row['selection_score']):.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / f"{prefix}_indirect_selection_quality.png"
    _save(fig, path)
    return path


def _plot_strategy_tradeoff(selection_df: pd.DataFrame, output_dir: Path, prefix: str) -> Path:
    df = selection_df.copy()
    for column in (
        "boundary_support_r1_global",
        "action_coverage_global",
        "p95_window_max_empty_gap",
        "probe_ap",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    strategies = sorted(df["strategy"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=170)
    fig.suptitle("Strategy trade-off across completed coarse classifiers", fontsize=15, fontweight="bold")

    ax = axes[0]
    cmap = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(strategies), 1)))
    for color, strategy in zip(cmap, strategies):
        part = df[df["strategy"].astype(str) == strategy]
        ax.scatter(
            part["action_coverage_global"],
            part["boundary_support_r1_global"],
            s=65,
            alpha=0.78,
            label=strategy,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("Action coverage")
    ax.set_ylabel("Boundary support r=1")
    ax.set_title("Boundary versus action coverage")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7, loc="best")

    ax = axes[1]
    aggregate = (
        df.groupby("strategy", as_index=False)
        .agg(
            boundary_support_r1_global=("boundary_support_r1_global", "mean"),
            action_coverage_global=("action_coverage_global", "mean"),
            p95_window_max_empty_gap=("p95_window_max_empty_gap", "mean"),
        )
        .sort_values("boundary_support_r1_global", ascending=False)
    )
    y = np.arange(len(aggregate))
    bar_h = 0.34
    ax.barh(
        y - bar_h / 2.0,
        aggregate["boundary_support_r1_global"],
        height=bar_h,
        color="#7c3aed",
        alpha=0.88,
        label="Boundary",
    )
    ax.barh(
        y + bar_h / 2.0,
        aggregate["action_coverage_global"],
        height=bar_h,
        color="#0f766e",
        alpha=0.82,
        label="Action",
    )
    ax.set_yticks(y, aggregate["strategy"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(0.7, float(np.nanmax(aggregate[["boundary_support_r1_global", "action_coverage_global"]].to_numpy())) + 0.08))
    ax.set_title("Mean coverage by strategy")
    ax.set_xlabel("Coverage")
    ax.grid(axis="x", alpha=0.24)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / f"{prefix}_strategy_tradeoff.png"
    _save(fig, path)
    return path


def _sample_models(sample_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    models = sample_payload.get("models")
    if isinstance(models, Mapping):
        return models
    raise ValueError("sample JSON must contain a 'models' object")


def _model_samples(model_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    samples = model_payload.get("samples")
    if isinstance(samples, Mapping):
        return samples
    sample_lines = model_payload.get("sample_lines")
    if isinstance(sample_lines, Mapping):
        return sample_lines
    return {}


def _first_sample_for_model(sample_payload: Mapping[str, Any], model: str) -> tuple[str, Mapping[str, Any]]:
    if "models" not in sample_payload and (
        "sample_id" in sample_payload or "valid_len" in sample_payload or "p_action" in sample_payload
    ):
        resolved = model or str(
            sample_payload.get("probe_model")
            or sample_payload.get("matrix_model_id")
            or sample_payload.get("route_variant")
            or "sample"
        )
        return resolved, sample_payload
    models = _sample_models(sample_payload)
    if model not in models:
        model = str(next(iter(models)))
    samples = _model_samples(models[model])
    if not samples:
        raise ValueError(f"sample JSON has no samples for model {model}")
    sample_id = str(next(iter(samples)))
    sample = samples[sample_id]
    if not isinstance(sample, Mapping):
        raise ValueError(f"sample payload for {sample_id} must be an object")
    return model, sample


def _sample_for_model_and_id(
    sample_payload: Mapping[str, Any],
    *,
    model: str,
    sample_id: str | None,
) -> tuple[str, Mapping[str, Any]]:
    if "models" not in sample_payload and (
        "sample_id" in sample_payload or "valid_len" in sample_payload or "p_action" in sample_payload
    ):
        resolved_model, sample = _first_sample_for_model(sample_payload, model)
        resolved_sample_id = str(sample.get("sample_id") or sample.get("name") or "sample")
        if sample_id is not None and sample_id != resolved_sample_id:
            raise ValueError(f"sample JSON contains {resolved_sample_id}, not requested sample_id={sample_id}")
        return resolved_model, sample

    models = _sample_models(sample_payload)
    if model not in models:
        model = str(next(iter(models)))
    samples = _model_samples(models[model])
    if not samples:
        raise ValueError(f"sample JSON has no samples for model {model}")
    if sample_id is None:
        sample_id = str(next(iter(samples)))
    if sample_id not in samples:
        raise ValueError(f"sample JSON has no sample_id={sample_id} for model {model}")
    sample = samples[sample_id]
    if not isinstance(sample, Mapping):
        raise ValueError(f"sample payload for {sample_id} must be an object")
    return model, sample


def _sample_length(sample: Mapping[str, Any]) -> int:
    for key in ("valid_len", "length", "dense_len"):
        if key in sample:
            return max(0, int(float(sample[key])))
    values = sample.get("p_action") or []
    return len(values) if isinstance(values, Sequence) else 0


def _value_for_key(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    strategies = payload.get("strategy_selected_positions")
    if isinstance(strategies, Mapping) and key in strategies:
        return strategies.get(key)
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _positions_for_key(
    sample: Mapping[str, Any],
    *,
    length: int,
    key: str,
    fallback_keys: Sequence[str] = (),
) -> list[int]:
    selected = _value_for_key(sample, key)
    if selected is None:
        for fallback in fallback_keys:
            selected = _value_for_key(sample, fallback)
            if selected is not None:
                break
    if selected is None or isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
        return []
    out: set[int] = set()
    for item in selected:
        try:
            position = int(float(item))
        except (TypeError, ValueError):
            continue
        if 0 <= position < int(length):
            out.add(position)
    return sorted(out)


def _selected_delta_positions(sample: Mapping[str, Any], *, length: int) -> list[int]:
    return _positions_for_key(
        sample,
        length=length,
        key="selected_delta_p_action",
        fallback_keys=("top_change_positions",),
    )


def _segments_from_action_target(sample: Mapping[str, Any], *, length: int) -> list[tuple[float, float]]:
    target = sample.get("action_target")
    if target is None or isinstance(target, (str, bytes)) or not isinstance(target, Sequence):
        return []
    max_len = min(int(length), len(target))
    segments: list[tuple[float, float]] = []
    start: int | None = None
    for idx in range(max_len):
        active = _float_or_nan(target[idx]) >= 0.5
        if active and start is None:
            start = idx
        if not active and start is not None:
            segments.append((float(start), float(idx - 1)))
            start = None
    if start is not None:
        segments.append((float(start), float(max_len - 1)))
    return segments


def _sample_segments(sample: Mapping[str, Any], *, length: int) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for key in ("segments", "gt_segments"):
        raw_segments = sample.get(key) or []
        if isinstance(raw_segments, (str, bytes)) or not isinstance(raw_segments, Sequence):
            continue
        for segment in raw_segments:
            if isinstance(segment, Sequence) and not isinstance(segment, (str, bytes)) and len(segment) >= 2:
                start = max(0.0, _float_or_nan(segment[0]))
                end = min(float(length - 1), _float_or_nan(segment[1]))
                if np.isfinite(start) and np.isfinite(end) and end >= start:
                    segments.append((start, end))
        if segments:
            return segments
    return _segments_from_action_target(sample, length=length)


def _sample_boundaries(
    sample: Mapping[str, Any],
    *,
    length: int,
    segments: Sequence[tuple[float, float]],
) -> list[float]:
    boundaries: list[float] = []
    for key in ("boundaries", "gt_boundaries"):
        raw = sample.get(key) or []
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            continue
        for item in raw:
            boundary = _float_or_nan(item)
            if 0 <= boundary < length:
                boundaries.append(float(boundary))
        if boundaries:
            return sorted(set(boundaries))
    for start, end in segments:
        boundaries.extend([float(start), float(end)])
    return sorted(set(boundaries))


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
        candidate = int(item)
        while candidate in used and candidate + 1 < valid_len:
            candidate += 1
        while candidate in used and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in used:
            used.add(candidate)
            out.append(candidate)
    return sorted(out)


def _mean_nearest_distance(source: Sequence[int], target: Sequence[int]) -> float | None:
    if not source or not target:
        return None
    target_arr = np.asarray(list(target), dtype=float)
    distances = [float(np.min(np.abs(target_arr - float(item)))) for item in source]
    return float(np.mean(distances)) if distances else None


def _mean_rank_aligned_abs_distance(left: Sequence[int], right: Sequence[int]) -> float | None:
    if not left or not right:
        return None
    count = min(len(left), len(right))
    if count <= 0:
        return None
    left_arr = np.asarray(sorted(left)[:count], dtype=float)
    right_arr = np.asarray(sorted(right)[:count], dtype=float)
    return float(np.mean(np.abs(left_arr - right_arr)))


def _uniform_delta_metrics(
    *,
    valid_len: int,
    delta_positions: Sequence[int],
    uniform_positions: Sequence[int],
) -> dict[str, Any]:
    return _uniform_selection_metrics(
        valid_len=valid_len,
        selection_positions=delta_positions,
        uniform_positions=uniform_positions,
        selection_label=f"delta_p_action_{len(set(int(item) for item in delta_positions))}",
        uniform_label=f"uniform_{len(set(int(item) for item in uniform_positions))}",
    )


def _uniform_selection_metrics(
    *,
    valid_len: int,
    selection_positions: Sequence[int],
    uniform_positions: Sequence[int],
    selection_label: str | None = None,
    uniform_label: str | None = None,
) -> dict[str, Any]:
    selection_set = set(int(item) for item in selection_positions)
    uniform_set = set(int(item) for item in uniform_positions)
    intersection = selection_set.intersection(uniform_set)
    union = selection_set.union(uniform_set)
    selection_to_uniform = _mean_nearest_distance(sorted(selection_set), sorted(uniform_set))
    uniform_to_selection = _mean_nearest_distance(sorted(uniform_set), sorted(selection_set))
    nearest_values = [value for value in (selection_to_uniform, uniform_to_selection) if value is not None]
    symmetric = float(np.mean(nearest_values)) if nearest_values else None
    selected_count = len(selection_set)
    resolved_selection_label = selection_label or f"selected_{selected_count}"
    resolved_uniform_label = uniform_label or f"uniform_{len(uniform_set)}"
    return {
        "valid_len": int(valid_len),
        "selected_count": int(selected_count),
        "selection_label": resolved_selection_label,
        "selected_label": resolved_selection_label,
        "delta_label": resolved_selection_label,
        "uniform_label": resolved_uniform_label,
        "selection_positions": sorted(selection_set),
        "delta_positions": sorted(selection_set),
        "uniform_positions": sorted(uniform_set),
        "overlap_count": int(len(intersection)),
        "union_count": int(len(union)),
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
        "overlap_fraction": float(len(intersection) / selected_count) if selected_count else 0.0,
        "mean_selection_to_uniform_nearest_distance": selection_to_uniform,
        "mean_uniform_to_selection_nearest_distance": uniform_to_selection,
        "mean_delta_to_uniform_nearest_distance": selection_to_uniform,
        "mean_uniform_to_delta_nearest_distance": uniform_to_selection,
        "mean_symmetric_nearest_distance": symmetric,
        "mean_rank_aligned_abs_distance": _mean_rank_aligned_abs_distance(sorted(selection_set), sorted(uniform_set)),
    }


def _plot_sample_timeline(
    sample_payload: Mapping[str, Any],
    *,
    model: str,
    output_dir: Path,
    prefix: str,
) -> Path:
    model, sample = _first_sample_for_model(sample_payload, model)
    sample_id = str(sample.get("sample_id") or sample.get("name") or "sample")
    length = _sample_length(sample)
    if length <= 0:
        raise ValueError(f"sample {sample_id} has no positive length")

    p_action = np.asarray(_sample_signal(sample, "p_action"), dtype=float)[:length]
    p_change = _scale01(_sample_signal(sample, "p_change", "p_change_scaled"), length)
    boundary_score = _scale01(_sample_signal(sample, "boundary_score", "transition_scaled"), length)
    selected = _selected_delta_positions(sample, length=length)
    segments = _sample_segments(sample, length=length)
    boundaries = _sample_boundaries(sample, length=length, segments=segments)

    fig, ax = plt.subplots(figsize=(16, 5.8), dpi=170)
    x = np.arange(length)
    for segment in segments:
        start = max(0.0, float(segment[0]))
        end = min(float(length - 1), float(segment[1]))
        if end >= start:
            ax.axvspan(start, end, color="#ef4444", alpha=0.12, linewidth=0)
    for boundary in boundaries:
        ax.axvline(boundary, color="#ef4444", alpha=0.78, linewidth=0.9)
    if p_action.size:
        ax.plot(x[: p_action.size], p_action, color="#2563eb", linewidth=1.8, label="p_action")
    if p_change.size:
        ax.plot(x[: p_change.size], p_change, color="#0f766e", linewidth=1.15, alpha=0.95, label="scaled |delta p_action|")
    if boundary_score.size:
        ax.plot(
            x[: boundary_score.size],
            boundary_score,
            color="#7c3aed",
            linewidth=1.05,
            alpha=0.88,
            label="scaled transition/boundary score",
        )
    if selected:
        y_values = [float(p_action[item]) if item < len(p_action) else 0.4 for item in selected]
        ax.scatter(selected, y_values, s=22, color="#f97316", edgecolor="white", linewidth=0.35, zorder=5, label="selected")

    boundary_text = sample.get("boundary_support_r1")
    action_text = sample.get("action_coverage")
    max_gap = sample.get("max_empty_gap")
    p95_gap = sample.get("p95_empty_gap")
    note = [
        f"model={_short_label(model)}",
        f"sample={sample_id}",
        f"selected={len(selected)}",
    ]
    if boundary_text is not None:
        note.append(f"boundary_r1={float(boundary_text):.3f}")
    if action_text is not None:
        note.append(f"action_cov={float(action_text):.3f}")
    if max_gap is not None:
        note.append(f"max_gap={float(max_gap):.0f}")
    if p95_gap is not None:
        note.append(f"p95_gap={float(p95_gap):.1f}")
    ax.text(
        0.99,
        0.95,
        "  ".join(note),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9, boxstyle="round,pad=0.25"),
    )
    ax.set_title("Sample timeline: p_action, transition signal, GT boundaries, and selected frames")
    ax.set_xlabel("Dense-window snippet index")
    ax.set_ylabel("Probability / scaled score")
    ax.set_xlim(0, max(1, length - 1))
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right")
    fig.tight_layout()
    path = output_dir / f"{prefix}_sample_timeline_{model}.png"
    _save(fig, path)
    return path


def _plot_uniform_vs_selection_rug(
    sample_payload: Mapping[str, Any],
    *,
    model: str,
    output_dir: Path,
    prefix: str,
    selection_key: str,
    selection_label: str | None = None,
    uniform_label: str | None = None,
    title: str | None = None,
    filename_suffix: str = "uniform_vs_selection_rug",
) -> tuple[Path, dict[str, Any]]:
    model, sample = _first_sample_for_model(sample_payload, model)
    sample_id = str(sample.get("sample_id") or sample.get("name") or "sample")
    length = _sample_length(sample)
    if length <= 0:
        raise ValueError(f"sample {sample_id} has no positive length")

    p_action = np.asarray(_sample_signal(sample, "p_action"), dtype=float)[:length]
    p_change = _scale01(_sample_signal(sample, "p_change", "p_change_scaled"), length)
    boundary_score = _scale01(_sample_signal(sample, "boundary_score", "transition_scaled"), length)
    selection_positions = _positions_for_key(sample, length=length, key=selection_key)
    if not selection_positions and selection_key == "selected_delta_p_action":
        selection_positions = _selected_delta_positions(sample, length=length)
    default_selection_name = "delta_p_action" if selection_key == "selected_delta_p_action" else selection_key
    resolved_selection_label = selection_label or f"{default_selection_name}_{len(selection_positions)}"
    resolved_uniform_label = uniform_label or f"uniform_{len(selection_positions)}"
    uniform_positions = _uniform_reference_positions(valid_len=length, selected_count=len(selection_positions))
    metrics = _uniform_selection_metrics(
        valid_len=length,
        selection_positions=selection_positions,
        uniform_positions=uniform_positions,
        selection_label=resolved_selection_label,
        uniform_label=resolved_uniform_label,
    )
    segments = _sample_segments(sample, length=length)
    boundaries = _sample_boundaries(sample, length=length, segments=segments)

    fig, (signal_ax, rug_ax) = plt.subplots(
        2,
        1,
        figsize=(16, 6.6),
        dpi=170,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08},
    )
    x = np.arange(length)
    for segment in segments:
        start = max(0.0, float(segment[0]))
        end = min(float(length - 1), float(segment[1]))
        if end >= start:
            signal_ax.axvspan(start, end, color="#ef4444", alpha=0.11, linewidth=0)
            rug_ax.axvspan(start, end, color="#ef4444", alpha=0.08, linewidth=0)
    for boundary in boundaries:
        signal_ax.axvline(boundary, color="#ef4444", alpha=0.62, linewidth=0.85)
        rug_ax.axvline(boundary, color="#ef4444", alpha=0.26, linewidth=0.75)
    if p_action.size:
        signal_ax.plot(x[: p_action.size], p_action, color="#2563eb", linewidth=1.75, label="p_action")
    if p_change.size:
        signal_ax.plot(x[: p_change.size], p_change, color="#0f766e", linewidth=1.05, alpha=0.92, label="scaled |delta p_action|")
    if boundary_score.size:
        signal_ax.plot(
            x[: boundary_score.size],
            boundary_score,
            color="#7c3aed",
            linewidth=1.0,
            alpha=0.82,
            label="scaled transition/boundary score",
        )
    if selection_positions:
        y_values = [float(p_action[item]) if item < len(p_action) else 0.4 for item in selection_positions]
        signal_ax.scatter(
            selection_positions,
            y_values,
            s=22,
            color="#f97316",
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
            label=metrics["selection_label"],
        )

    rug_ax.vlines(uniform_positions, 0.62, 1.12, color="#64748b", linewidth=0.85, alpha=0.55, label=metrics["uniform_label"])
    rug_ax.vlines(selection_positions, -0.12, 0.38, color="#f97316", linewidth=1.0, alpha=0.82, label=metrics["selection_label"])
    if uniform_positions:
        rug_ax.scatter(uniform_positions, [0.87] * len(uniform_positions), s=11, color="#64748b", alpha=0.55)
    if selection_positions:
        rug_ax.scatter(selection_positions, [0.13] * len(selection_positions), s=13, color="#f97316", alpha=0.88)

    metric_text = (
        f"Jaccard={metrics['jaccard']:.3f}  "
        f"overlap={metrics['overlap_fraction']:.3f}  "
        f"mean nearest={metrics['mean_symmetric_nearest_distance']:.2f}  "
        f"rank distance={metrics['mean_rank_aligned_abs_distance']:.2f}"
    )
    signal_ax.text(
        0.99,
        0.94,
        f"model={_short_label(model)}  sample={sample_id}  {metrics['selection_label']} vs {metrics['uniform_label']}\n{metric_text}",
        transform=signal_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9, boxstyle="round,pad=0.25"),
    )
    signal_ax.set_title(title or f"Same-window selected-frame rug: {metrics['selection_label']} versus uniform")
    signal_ax.set_ylabel("Probability / scaled score")
    signal_ax.set_ylim(-0.04, 1.04)
    signal_ax.grid(alpha=0.20)
    signal_ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right")

    rug_ax.set_yticks([0.13, 0.87], [metrics["selection_label"], metrics["uniform_label"]], fontsize=8)
    rug_ax.set_ylim(-0.28, 1.28)
    rug_ax.set_xlabel("Dense-window snippet index")
    rug_ax.set_xlim(0, max(1, length - 1))
    rug_ax.grid(axis="x", alpha=0.18)
    rug_ax.spines[["top", "right", "left"]].set_visible(False)
    rug_ax.tick_params(axis="y", length=0)

    path = output_dir / f"{prefix}_{filename_suffix}_{model}.png"
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.10, hspace=0.08)
    _save(fig, path)
    return path, metrics


def _plot_uniform_vs_delta_rug(
    sample_payload: Mapping[str, Any],
    *,
    model: str,
    output_dir: Path,
    prefix: str,
) -> tuple[Path, dict[str, Any]]:
    return _plot_uniform_vs_selection_rug(
        sample_payload,
        model=model,
        output_dir=output_dir,
        prefix=prefix,
        selection_key="selected_delta_p_action",
        title="Same-window selected-frame rug: delta_p_action versus uniform",
        filename_suffix="uniform_vs_delta_rug",
    )


def render_performance_figures(
    *,
    candidate_csv: str | Path,
    selection_csv: str | Path,
    sample_json: str | Path,
    output_dir: str | Path,
    prefix: str = "c3_probe",
) -> dict[str, Any]:
    candidate_df = _load_dataframe(
        candidate_csv,
        required_columns=(
            "model",
            "ap",
            "auc",
            "best_f1",
            "action_bg_gap",
            "change_lift",
            "top10_r4",
            "top20_r4",
            "candidate_score",
        ),
    )
    selection_df = _load_dataframe(
        selection_csv,
        required_columns=(
            "model",
            "strategy",
            "probe_ap",
            "probe_roc_auc",
            "boundary_support_r1_global",
            "action_coverage_global",
            "p95_window_max_empty_gap",
            "mean_valid_len",
            "max_empty_gap_global",
            "mean_selected",
            "windows",
        ),
    )
    sample_path = Path(sample_json)
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample JSON not found: {sample_path}")
    sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = str(prefix)

    figures: list[dict[str, Any]] = []
    classifier_path = _plot_classifier_metrics(candidate_df, output_path, prefix)
    _add_figure(figures, classifier_path, title="C3 coarse classifier metrics", kind="coarse_classifier_metrics")

    delta_df = _delta_selection_rows(selection_df)
    selection_path = _plot_delta_selection_quality(delta_df, output_path, prefix)
    _add_figure(figures, selection_path, title="delta_p_action indirect selection quality", kind="indirect_selection")

    tradeoff_path = _plot_strategy_tradeoff(selection_df, output_path, prefix)
    _add_figure(figures, tradeoff_path, title="Indirect selection strategy trade-off", kind="strategy_tradeoff")

    best_model = str(delta_df.iloc[0]["model"])
    timeline_path = _plot_sample_timeline(sample_payload, model=best_model, output_dir=output_path, prefix=prefix)
    _add_figure(figures, timeline_path, title=f"Sample timeline for {best_model}", kind="sample_timeline")
    rug_path, uniform_delta_comparison = _plot_uniform_vs_delta_rug(
        sample_payload,
        model=best_model,
        output_dir=output_path,
        prefix=prefix,
    )
    _add_figure(
        figures,
        rug_path,
        title=f"Uniform versus delta_p_action rug for {best_model}",
        kind="uniform_vs_delta_rug",
    )

    manifest = {
        "schema_version": "c3_probe_true_performance_figures_v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": {
            "candidate_csv": str(candidate_csv),
            "selection_csv": str(selection_csv),
            "sample_json": str(sample_json),
        },
        "output_dir": str(output_path),
        "prefix": prefix,
        "figure_count": len(figures),
        "figures": figures,
        "best_delta_selection_model": best_model,
        "uniform_delta_comparison": uniform_delta_comparison,
        "note": "Figures are diagnostics for p_action/coarse selection quality, not AdaTAD detector mAP.",
    }
    _write_manifest(output_path, prefix, manifest)
    return manifest


def _load_json_mapping(path: str | Path) -> Mapping[str, Any]:
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain an object: {json_path}")
    return payload


def _find_jsonl_sample(path: str | Path, *, sample_id: str) -> Mapping[str, Any]:
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, Mapping):
                raise ValueError(f"{jsonl_path}:{line_no}: row must be an object")
            if str(row.get("sample_id") or "") == sample_id:
                return row
    raise ValueError(f"{jsonl_path}: sample_id not found: {sample_id}")


def render_same_window_selection_rug(
    *,
    sample_json: str | Path,
    output_dir: str | Path,
    prefix: str = "selection",
    model: str = "sample",
    sample_id: str | None = None,
    selection_key: str = "selected_delta_p_action",
    selection_label: str | None = None,
    uniform_label: str | None = None,
    ledger_jsonl: str | Path | None = None,
    ledger_selection_key: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    sample_payload = _load_json_mapping(sample_json)
    model, sample = _sample_for_model_and_id(sample_payload, model=model, sample_id=sample_id)
    resolved_sample_id = str(sample.get("sample_id") or sample.get("name") or sample_id or "sample")
    sample_for_plot = dict(sample)

    source_files: dict[str, str | None] = {
        "sample_json": str(sample_json),
        "ledger_jsonl": None if ledger_jsonl is None else str(ledger_jsonl),
    }
    if ledger_jsonl is not None:
        ledger_row = _find_jsonl_sample(ledger_jsonl, sample_id=resolved_sample_id)
        source_key = ledger_selection_key or selection_key
        selected = _value_for_key(ledger_row, source_key)
        if selected is None:
            raise ValueError(f"{ledger_jsonl}:{resolved_sample_id}: missing selection key {source_key}")
        sample_for_plot[selection_key] = selected
        if "valid_len" not in sample_for_plot:
            for length_key in ("valid_len", "target_len", "dense_len"):
                if length_key in ledger_row:
                    sample_for_plot["valid_len"] = ledger_row[length_key]
                    break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    one_sample_payload = {"models": {model: {"samples": {resolved_sample_id: sample_for_plot}}}}
    rug_path, comparison = _plot_uniform_vs_selection_rug(
        one_sample_payload,
        model=model,
        output_dir=output_path,
        prefix=prefix,
        selection_key=selection_key,
        selection_label=selection_label,
        uniform_label=uniform_label,
        title=title,
        filename_suffix="same_window_rug",
    )

    manifest = {
        "schema_version": "same_window_selection_rug_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": source_files,
        "output_dir": str(output_path),
        "prefix": prefix,
        "model": model,
        "sample_id": resolved_sample_id,
        "selection_key": selection_key,
        "ledger_selection_key": ledger_selection_key,
        "selection_label": comparison["selection_label"],
        "uniform_label": comparison["uniform_label"],
        "figure_count": 1,
        "figures": [
            {
                "kind": "same_window_selection_rug",
                "title": title or f"{comparison['selection_label']} versus {comparison['uniform_label']}",
                "path": str(rug_path),
                "bytes": rug_path.stat().st_size,
            }
        ],
        "uniform_selection_comparison": comparison,
    }
    _write_manifest(output_path, prefix, manifest)
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render C3 coarse-classifier and indirect-frame-selection performance figures from local analysis tables."
    )
    parser.add_argument("--mode", choices=("performance", "same-window-rug"), default="performance")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--selection-csv", default=str(DEFAULT_SELECTION_CSV))
    parser.add_argument("--sample-json", default=str(DEFAULT_SAMPLE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="c3_probe")
    parser.add_argument("--model", default="sample")
    parser.add_argument("--sample-id")
    parser.add_argument("--rug-selection-key", default="selected_delta_p_action")
    parser.add_argument("--rug-selection-label")
    parser.add_argument("--rug-uniform-label")
    parser.add_argument("--ledger-jsonl")
    parser.add_argument("--ledger-selection-key")
    parser.add_argument("--rug-title")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.mode == "same-window-rug":
        manifest = render_same_window_selection_rug(
            sample_json=args.sample_json,
            output_dir=args.output_dir,
            prefix=args.prefix,
            model=args.model,
            sample_id=args.sample_id,
            selection_key=args.rug_selection_key,
            selection_label=args.rug_selection_label,
            uniform_label=args.rug_uniform_label,
            ledger_jsonl=args.ledger_jsonl,
            ledger_selection_key=args.ledger_selection_key,
            title=args.rug_title,
        )
    else:
        manifest = render_performance_figures(
            candidate_csv=args.candidate_csv,
            selection_csv=args.selection_csv,
            sample_json=args.sample_json,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
