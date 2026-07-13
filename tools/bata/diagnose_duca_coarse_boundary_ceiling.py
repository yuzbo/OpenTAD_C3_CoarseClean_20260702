from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.analyze_duca_selection_quality import _boundaries, _boundary_labels, _validated_segments, binary_metrics
from tools.bata.duca_ceiling_utils import read_jsonl, sha256, write_csv, write_json


FEATURE_KEYS = ("p_action", "abs_delta_p_action", "uncertainty")


def _vector(row: Mapping[str, Any], key: str, length: int) -> list[float]:
    value = row.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < length:
        raise ValueError(f"{row.get('sample_id')}: missing deploy-visible vector {key}")
    return [float(item) for item in value[:length]]


def temporal_features(row: Mapping[str, Any], radius: int = 2) -> tuple[list[list[float]], list[int]]:
    length = int(row["valid_len"])
    signals = [_vector(row, key, length) for key in FEATURE_KEYS]
    segments = _validated_segments(row, length)
    labels = _boundary_labels(length, _boundaries(length, segments), radius)
    features: list[list[float]] = []
    for index in range(length):
        item = [index / max(1, length - 1)]
        for signal in signals:
            for offset in (-4, -2, -1, 0, 1, 2, 4):
                item.append(signal[min(length - 1, max(0, index + offset))])
        features.append(item)
    return features, labels


def _fit_logistic(features: Sequence[Sequence[float]], labels: Sequence[int], steps: int, learning_rate: float) -> dict[str, Any]:
    width = len(features[0])
    means = [sum(row[col] for row in features) / len(features) for col in range(width)]
    scales = [math.sqrt(sum((row[col] - means[col]) ** 2 for row in features) / len(features)) or 1.0 for col in range(width)]
    weights = [0.0] * width
    bias = 0.0
    positives = max(1, sum(labels))
    positive_weight = (len(labels) - positives) / positives
    for _ in range(int(steps)):
        grad = [0.0] * width
        grad_bias = 0.0
        for row, label in zip(features, labels):
            normalized = [(value - means[col]) / scales[col] for col, value in enumerate(row)]
            logit = bias + sum(weight * value for weight, value in zip(weights, normalized))
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            factor = positive_weight if label else 1.0
            error = factor * (probability - label)
            grad_bias += error
            for col, value in enumerate(normalized):
                grad[col] += error * value
        rate = float(learning_rate) / len(features)
        bias -= rate * grad_bias
        weights = [weight - rate * value for weight, value in zip(weights, grad)]
    return {"means": means, "scales": scales, "weights": weights, "bias": bias}


def _predict(model: Mapping[str, Any], features: Sequence[Sequence[float]]) -> list[float]:
    out = []
    for row in features:
        normalized = [(value - model["means"][col]) / model["scales"][col] for col, value in enumerate(row)]
        logit = model["bias"] + sum(weight * value for weight, value in zip(model["weights"], normalized))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))
    return out


def run(train_jsonl: str | Path, eval_jsonl: str | Path, output_dir: str | Path, radius: int = 2, steps: int = 80) -> dict[str, Any]:
    train_rows, eval_rows = read_jsonl(train_jsonl), read_jsonl(eval_jsonl)
    train_videos = {str(row.get("video_id")) for row in train_rows}
    eval_videos = {str(row.get("video_id")) for row in eval_rows}
    overlap = sorted(train_videos & eval_videos)
    if overlap:
        raise ValueError(f"train/eval video leakage: {overlap[:5]}")
    train_x: list[list[float]] = []
    train_y: list[int] = []
    for row in train_rows:
        features, labels = temporal_features(row, radius=radius)
        train_x.extend(features)
        train_y.extend(labels)
    model = _fit_logistic(train_x, train_y, steps=steps, learning_rate=0.2)
    pooled_y: list[int] = []
    pooled_scores: list[float] = []
    csv_rows = []
    for row in eval_rows:
        features, labels = temporal_features(row, radius=radius)
        scores = _predict(model, features)
        pooled_y.extend(labels)
        pooled_scores.extend(scores)
        metrics = binary_metrics(labels, scores)
        csv_rows.append({"sample_id": row.get("sample_id"), **{key: metrics[key] for key in ("auroc", "auprc", "ece", "prevalence")}})
    summary = {
        "schema_version": "duca_deploy_visible_boundary_probe_v2",
        "diagnostic_role": "held_out_deploy_visible_linear_probe_not_ceiling_not_oracle",
        "boundary_radius": int(radius),
        "feature_keys": list(FEATURE_KEYS),
        "train_video_count": len(train_videos),
        "eval_video_count": len(eval_videos),
        "video_overlap_count": 0,
        "metrics": binary_metrics(pooled_y, pooled_scores),
        "provenance": {"train_sha256": sha256(train_jsonl), "eval_sha256": sha256(eval_jsonl)},
        "leakage_contract": {"gt_used_for_training_label_only": True, "gt_used_as_model_input": False, "eval_gt_used_for_selection": False},
    }
    out = Path(output_dir)
    write_json(out / "coarse_boundary_probe.json", summary)
    write_csv(out / "coarse_boundary_probe_per_sample.csv", csv_rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out deploy-visible linear boundary probe; not a ceiling.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.train_jsonl, args.eval_jsonl, args.output_dir, args.radius, args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
