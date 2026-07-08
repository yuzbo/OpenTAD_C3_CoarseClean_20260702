from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTPUT_SCHEMA_VERSION = "zero_shot_actionness_eval_v1"
SUMMARY_SCHEMA_VERSION = "zero_shot_actionness_eval_summary_v1"
READY = "ZERO_SHOT_ACTIONNESS_EVAL_READY"
SUPPORTED_SOURCE_MODES = {
    "motion",
    "feature_energy",
    "manual_jsonl",
    "video_text_mock",
    "xclip",
    "actionclip",
    "slowfast",
    "videomae",
}
RESERVED_EXTERNAL_PROVIDER_MODES = {"xclip", "actionclip", "slowfast", "videomae"}
FORBIDDEN_SOURCE_KEYS = (
    "teacher",
    "teacher_logits",
    "teacher_scores",
    "teacher_utility",
    "gt_segments",
    "gt_labels",
    "ground_truth",
    "oracle",
    "prediction_cache",
    "raw_prediction",
    "raw_predictions",
    "raw_logits",
    "raw_scores",
)
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _find_forbidden_paths(value: Any, *, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = key_text if not path else f"{path}.{key_text}"
            lower = key_text.lower()
            if lower in FORBIDDEN_TRUE_FLAGS:
                if _is_true(child):
                    hits.append(child_path)
                continue
            if (
                lower in FORBIDDEN_SOURCE_KEYS
                or lower.startswith("gt_")
                or any(token in lower for token in ("teacher", "raw_prediction", "prediction_cache", "cache", "oracle", "ground_truth"))
            ):
                hits.append(child_path)
                continue
            hits.extend(_find_forbidden_paths(child, path=child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_find_forbidden_paths(child, path=f"{path}[{idx}]"))
    return hits


def _reject_source_leakage(payload: Mapping[str, Any], *, source_name: str) -> None:
    hits = _find_forbidden_paths(payload)
    if hits:
        raise ValueError(f"{source_name}: forbidden source payload/provenance key {hits[0]}")


def annotation_segments(annotation: Mapping[str, Any]) -> dict[str, list[tuple[float, float, str | None]]]:
    database = annotation.get("database", annotation)
    if not isinstance(database, Mapping):
        raise ValueError("annotation must contain a database object or be a video mapping")
    out: dict[str, list[tuple[float, float, str | None]]] = {}
    for video_id, payload in database.items():
        if not isinstance(payload, Mapping):
            continue
        raw_annotations = payload.get("annotations") or payload.get("segments") or []
        segments: list[tuple[float, float, str | None]] = []
        if isinstance(raw_annotations, list):
            for item in raw_annotations:
                if isinstance(item, Mapping):
                    raw_segment = item.get("segment") or item.get("timestamps")
                    label = item.get("label")
                else:
                    raw_segment = item
                    label = None
                if isinstance(raw_segment, list) and len(raw_segment) >= 2:
                    start, end = float(raw_segment[0]), float(raw_segment[1])
                    if end > start:
                        segments.append((start, end, None if label is None else str(label)))
        out[str(video_id)] = segments
    return out


def _gt_action_for_time(segments: Sequence[tuple[float, float, str | None]], original_time: float) -> int:
    value = float(original_time)
    return int(any(start <= value < end for start, end, _label in segments))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str | None, int | None]:
    video_id = str(row.get("video_id") or row.get("video_name") or row.get("sample_id") or "")
    window = row.get("window_id")
    time_index = row.get("time_index")
    return (
        video_id,
        None if window is None else str(window),
        None if time_index is None else int(time_index),
    )


def _manual_score_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str | None, int | None], Mapping[str, Any]]:
    out: dict[tuple[str, str | None, int | None], Mapping[str, Any]] = {}
    for row in rows:
        out[_row_key(row)] = row
    return out


def _nested_numeric(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in row and isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
            return float(row[key])
    frame_signals = row.get("frame_signals")
    if isinstance(frame_signals, Mapping):
        for key in keys:
            if key in frame_signals and isinstance(frame_signals[key], (int, float)) and not isinstance(frame_signals[key], bool):
                return float(frame_signals[key])
    return None


def _feature_energy(row: Mapping[str, Any]) -> float:
    explicit = _nested_numeric(row, ("feature_energy", "energy", "motion_score", "motion"))
    if explicit is not None:
        return explicit
    for key in ("features", "feature", "feature_vector", "embedding"):
        value = row.get(key)
        if isinstance(value, list) and value:
            vals = [float(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
            if vals:
                return math.sqrt(sum(item * item for item in vals) / float(len(vals)))
    return 0.0


def _motion_score(row: Mapping[str, Any]) -> float:
    explicit = _nested_numeric(row, ("motion_score", "motion", "delta_feature_energy", "feature_delta"))
    if explicit is not None:
        return explicit
    return _feature_energy(row)


def _video_text_mock_score(row: Mapping[str, Any], *, action_prompts: Sequence[str], background_prompts: Sequence[str]) -> float:
    payload = {
        "video_id": row.get("video_id") or row.get("video_name"),
        "window_id": row.get("window_id"),
        "time_index": row.get("time_index"),
        "original_time": row.get("original_time"),
        "action_prompts": list(action_prompts),
        "background_prompts": list(background_prompts),
    }
    digest = int(_sha256_text(payload)[:12], 16)
    return float(digest % 10000) / 9999.0


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(float(item) for item in values)
    high = max(float(item) for item in values)
    if high <= low:
        return [0.5 for _item in values]
    return [(float(item) - low) / (high - low) for item in values]


def _logit(prob: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
    return math.log(clipped / (1.0 - clipped))


def _rank_pairs_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / float(total)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive_total = sum(1 for item in labels if int(item) == 1)
    if positive_total <= 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: float(item[0]), reverse=True)
    hit = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if int(label) == 1:
            hit += 1
            precision_sum += hit / float(rank)
    return precision_sum / float(positive_total)


def _recall_precision_at_k(labels: Sequence[int], scores: Sequence[float], k_values: Sequence[int]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    ranked = sorted(zip(scores, labels), key=lambda item: float(item[0]), reverse=True)
    positive_total = sum(1 for item in labels if int(item) == 1)
    recall: dict[str, float | None] = {}
    precision: dict[str, float | None] = {}
    for raw_k in k_values:
        k = max(1, int(raw_k))
        top = ranked[: min(k, len(ranked))]
        hits = sum(1 for _score, label in top if int(label) == 1)
        recall[str(k)] = None if positive_total <= 0 else hits / float(positive_total)
        precision[str(k)] = None if not top else hits / float(len(top))
    return recall, precision


def compute_metrics(rows: Sequence[Mapping[str, Any]], *, recall_k: Sequence[int]) -> dict[str, Any]:
    labels = [int(row["gt_action"]) for row in rows if row.get("valid") is True]
    scores = [float(row["p_action"]) for row in rows if row.get("valid") is True]
    recall, precision = _recall_precision_at_k(labels, scores, recall_k)
    action = sum(1 for item in labels if item == 1)
    background = sum(1 for item in labels if item == 0)
    total = action + background
    return {
        "auroc": _rank_pairs_auroc(labels, scores),
        "auprc": _average_precision(labels, scores),
        "recall_at_k": recall,
        "precision_at_k": precision,
        "action_background_balance": {
            "action": int(action),
            "background": int(background),
            "action_fraction": None if total <= 0 else action / float(total),
        },
    }


def _base_provenance(
    *,
    source_mode: str,
    prompt_hash: str | None,
    checkpoint_hash: str | None,
    thumos_trained: bool | None,
    uses_labels: bool,
    uses_teacher: bool,
    calibration_split: str | None,
) -> dict[str, Any]:
    return {
        "source_name": source_mode,
        "source_mode": source_mode,
        "prompt_hash": prompt_hash,
        "checkpoint_hash": checkpoint_hash,
        "thumos_trained": thumos_trained,
        "uses_labels": bool(uses_labels),
        "uses_teacher": bool(uses_teacher),
        "calibration_split": calibration_split,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "weights_downloaded": False,
    }


def _score_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    source_mode: str,
    manual_jsonl: str | Path | None,
    action_prompts: Sequence[str],
    background_prompts: Sequence[str],
) -> tuple[list[float], dict[str, Any]]:
    if source_mode in RESERVED_EXTERNAL_PROVIDER_MODES:
        raise ValueError(f"{source_mode} provider is reserved; tests/tools must inject local scores and must not download weights")
    prompt_hash = None
    checkpoint_hash = None
    thumos_trained: bool | None = False
    if source_mode == "manual_jsonl":
        if manual_jsonl is None:
            raise ValueError("manual_jsonl source mode requires manual_jsonl")
        manual_rows = _read_jsonl(manual_jsonl)
        for line_no, row in enumerate(manual_rows, start=1):
            _reject_source_leakage(row, source_name=f"{manual_jsonl}:{line_no}")
        manual_by_key = _manual_score_map(manual_rows)
        raw_scores: list[float] = []
        thumos_trained = None
        for source in sample_rows:
            key = _row_key(source)
            manual = manual_by_key.get(key)
            if manual is None and key[1] is not None:
                manual = manual_by_key.get((key[0], None, key[2]))
            if manual is None and key[1] is not None:
                manual = manual_by_key.get((key[0], key[1], None))
            if manual is None:
                raise ValueError(f"manual score missing for video_id={key[0]} window_id={key[1]} time_index={key[2]}")
            score = manual.get("p_action", manual.get("score"))
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"manual score must be numeric for {key}")
            raw_scores.append(float(score))
            provenance = manual.get("source_provenance")
            if isinstance(provenance, Mapping) and "thumos_trained" in provenance:
                thumos_trained = bool(provenance["thumos_trained"]) if provenance["thumos_trained"] is not None else None
        scores = [min(1.0, max(0.0, item)) for item in raw_scores]
    elif source_mode == "motion":
        scores = _minmax([_motion_score(row) for row in sample_rows])
    elif source_mode == "feature_energy":
        scores = _minmax([_feature_energy(row) for row in sample_rows])
    elif source_mode == "video_text_mock":
        prompt_hash = _sha256_text({"action": list(action_prompts), "background": list(background_prompts)})
        scores = [
            _video_text_mock_score(row, action_prompts=action_prompts, background_prompts=background_prompts)
            for row in sample_rows
        ]
    else:
        raise ValueError(f"unsupported source_mode: {source_mode}")
    provenance = _base_provenance(
        source_mode=source_mode,
        prompt_hash=prompt_hash,
        checkpoint_hash=checkpoint_hash,
        thumos_trained=thumos_trained,
        uses_labels=False,
        uses_teacher=False,
        calibration_split=None,
    )
    _reject_source_leakage(provenance, source_name=f"{source_mode}:source_provenance")
    return scores, provenance


def run_eval(
    *,
    annotation_json: str | Path,
    sample_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    source_mode: str = "motion",
    manual_jsonl: str | Path | None = None,
    recall_k: Sequence[int] = (1, 10, 100),
    action_prompts: Sequence[str] = ("a video clip showing a human action",),
    background_prompts: Sequence[str] = ("a video clip showing background or no action",),
) -> dict[str, Any]:
    if source_mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {sorted(SUPPORTED_SOURCE_MODES)}")
    annotation = _read_json(annotation_json)
    segments_by_video = annotation_segments(annotation)
    sample_rows = _read_jsonl(sample_jsonl)
    for line_no, row in enumerate(sample_rows, start=1):
        source_visible = {key: row[key] for key in row if key not in {"label", "annotations"}}
        _reject_source_leakage(source_visible, source_name=f"{sample_jsonl}:{line_no}")
    scores, provenance = _score_rows(
        sample_rows,
        source_mode=source_mode,
        manual_jsonl=manual_jsonl,
        action_prompts=action_prompts,
        background_prompts=background_prompts,
    )

    output_rows: list[dict[str, Any]] = []
    for line_no, (row, score) in enumerate(zip(sample_rows, scores), start=1):
        video_id = row.get("video_id") or row.get("video_name")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{sample_jsonl}:{line_no}: video_id is required")
        time_index = int(row.get("time_index", line_no - 1))
        original_time = float(row.get("original_time", time_index))
        valid = not ("valid" in row and _is_true(row.get("valid")) is False)
        p_action = min(1.0, max(0.0, float(score)))
        out = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "video_id": video_id,
            "window_id": str(row.get("window_id", f"{video_id}_{time_index}")),
            "time_index": time_index,
            "original_time": original_time,
            "p_action": p_action,
            "logit": _logit(p_action),
            "valid": bool(valid),
            "source_name": provenance["source_name"],
            "source_provenance": dict(provenance),
            "prompt_hash": provenance["prompt_hash"],
            "checkpoint_hash": provenance["checkpoint_hash"],
            "thumos_trained": provenance["thumos_trained"],
            "uses_labels": False,
            "uses_teacher": False,
            "calibration_split": provenance["calibration_split"],
            "gt_action": _gt_action_for_time(segments_by_video.get(video_id, []), original_time),
        }
        output_rows.append(out)
    metrics = compute_metrics(output_rows, recall_k=recall_k)
    _write_jsonl(output_jsonl, output_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "annotation_json": str(annotation_json),
        "sample_jsonl": str(sample_jsonl),
        "manual_jsonl": None if manual_jsonl is None else str(manual_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(output_rows),
        "source_mode": source_mode,
        "source_provenance": dict(provenance),
        "metrics": metrics,
        "primary_metric_policy": "threshold_free",
        "threshold_free_primary_metrics": ["auroc", "auprc", "recall_at_k", "precision_at_k"],
        "gt_labels_eval_only": True,
        "source_scoring_reads_gt": False,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot/no-target-label actionness sources.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--source-mode", default="motion", choices=sorted(SUPPORTED_SOURCE_MODES))
    parser.add_argument("--manual-jsonl")
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 10, 100])
    args = parser.parse_args(argv)
    summary = run_eval(
        annotation_json=args.annotation_json,
        sample_jsonl=args.sample_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source_mode=args.source_mode,
        manual_jsonl=args.manual_jsonl,
        recall_k=args.recall_k,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
