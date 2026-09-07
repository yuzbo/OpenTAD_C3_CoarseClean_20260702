"""Boundary risk with explicit misses, independent of official mAP code."""
import numpy as np


def class_agnostic_matches(gt, pred):
    matched = {}
    for prediction_index, p in enumerate(pred):
        candidates = []
        for index, event in enumerate(gt):
            if index in matched:
                continue
            start, end = event["segment"]
            left, right = p["segment"]
            intersection = max(0., min(end, right) - max(start, left))
            overlap = intersection / max(1e-12, end - start + right - left - intersection)
            candidates.append((overlap, -index, index))
        if candidates:
            overlap, _, index = max(candidates)
            if overlap >= .5:
                matched[index] = prediction_index
    return matched


def detection_risks(annotation, prediction, video_ids, short_seconds, score_threshold=0.):
    from opentad.evaluations.builder import remove_duplicate_annotations
    per_video = {}
    for name in video_ids:
        gt = remove_duplicate_annotations(annotation["database"][name].get("annotations", []))
        pred = sorted((p for p in prediction["results"].get(name, []) if p["score"] >= score_threshold),
                      key=lambda p: p["score"], reverse=True)[:100]
        row = dict(gt_count=len(gt), prediction_count=len(pred), matches=0, misses=0,
                   short_count=0, short_matches=0, start_abs_seconds=[], end_abs_seconds=[])
        used = set()
        for p in sorted(pred, key=lambda p: p["score"], reverse=True):
            candidates = [(i, a) for i, a in enumerate(gt) if i not in used and a["label"] == p["label"]]
            def iou(a):
                a0, a1 = a["segment"]
                p0, p1 = p["segment"]
                intersection = max(0., min(a1, p1) - max(a0, p0))
                return intersection / max(1e-12, a1 - a0 + p1 - p0 - intersection)
            if not candidates:
                continue
            i, a = max(candidates, key=lambda item: iou(item[1]))
            if iou(a) < .5:
                continue
            used.add(i)
            row["start_abs_seconds"].append(abs(p["segment"][0] - a["segment"][0]))
            row["end_abs_seconds"].append(abs(p["segment"][1] - a["segment"][1]))
        row["matches"], row["misses"] = len(used), len(gt) - len(used)
        row["unmatched_predictions"] = len(pred) - len(used)
        short = {i for i, a in enumerate(gt) if a["segment"][1] - a["segment"][0] <= short_seconds}
        row["short_count"], row["short_matches"] = len(short), len(short & used)
        agnostic = class_agnostic_matches(gt, pred)
        row.update(boundary_matches=len(agnostic), capped_start_sum_seconds=0., capped_end_sum_seconds=0.,
                   capped_start_sum_normalized=0., capped_end_sum_normalized=0., per_gt=[])
        for index, event in enumerate(gt):
            start, end = event["segment"]
            duration = end - start
            segment = pred[agnostic[index]]["segment"] if index in agnostic else None
            errors = [duration, duration] if segment is None else [min(duration, abs(segment[0] - start)), min(duration, abs(segment[1] - end))]
            for side, error in zip(("start", "end"), errors):
                row[f"capped_{side}_sum_seconds"] += error
                row[f"capped_{side}_sum_normalized"] += error / duration
            row["per_gt"].append(dict(gt_index=index, segment=event["segment"], label=event["label"],
                                     class_aware_matched=index in used, class_agnostic_match=agnostic.get(index),
                                     capped_start_seconds=errors[0], capped_end_seconds=errors[1]))
        per_video[name] = row
    total = lambda key: sum(v[key] for v in per_video.values())
    mean_error = lambda key: float(np.mean([x for v in per_video.values() for x in v[key]])) if total("matches") else None
    per_gt_mean = lambda key: total(key) / total("gt_count") if total("gt_count") else None
    return dict(matching="score-ordered, one-to-one tIoU >= 0.5; class-aware recall and class-agnostic capped boundaries",
                boundary_errors_conditioned_on_match=True, misses=total("misses"), gt_count=total("gt_count"),
                matches=total("matches"), prediction_count=total("prediction_count"),
                unmatched_predictions=total("unmatched_predictions"),
                short_recall=total("short_matches") / total("short_count") if total("short_count") else None,
                short_threshold_seconds=short_seconds, start_mae_seconds=mean_error("start_abs_seconds"),
                end_mae_seconds=mean_error("end_abs_seconds"), small_object_slice="NA: no object size labels",
                boundary_matched_fraction=per_gt_mean("boundary_matches"),
                capped_start_mae_seconds=per_gt_mean("capped_start_sum_seconds"),
                capped_end_mae_seconds=per_gt_mean("capped_end_sum_seconds"),
                capped_start_error_per_duration=per_gt_mean("capped_start_sum_normalized"),
                capped_end_error_per_duration=per_gt_mean("capped_end_sum_normalized"),
                capped_boundary_definition="each endpoint capped at GT duration; every miss contributes GT duration (normalized error 1)",
                proposal_cap_per_video=100, score_threshold=score_threshold, per_video=per_video)


def calibrate_risk_score(annotation, prediction, split, short_seconds):
    names = split["internal_dev"]
    if not names or set(prediction["results"]) - set(names):
        raise ValueError("risk score calibration accepts internal_dev predictions only")
    candidates = []
    for threshold in (0., .001, .005, .01, .02, .05, .1, .2, .3, .5):
        risk = detection_risks(annotation, prediction, names, short_seconds, threshold)
        denominator = risk["gt_count"] + risk["prediction_count"]
        candidates.append(dict(threshold=threshold, micro_f1=2 * risk["matches"] / denominator if denominator else 0.))
    best = max(candidates, key=lambda row: (row["micro_f1"], -row["threshold"]))
    return dict(score_threshold=best["threshold"], videos=names, candidates=candidates,
                definition="internal_dev micro F1 at class-aware tIoU .5; top 100; ties prefer lower threshold")


def paired_risk_bootstrap(left, right, samples=1000, seed=0):
    names = sorted(left["per_video"])
    if not names or set(names) != set(right["per_video"]):
        raise ValueError("paired bootstrap requires the same nonempty video set")
    pairs = dict(class_aware_recall=("matches", "gt_count"), short_recall=("short_matches", "short_count"),
                 capped_start_error=("capped_start_sum_normalized", "gt_count"),
                 capped_end_error=("capped_end_sum_normalized", "gt_count"))
    rng, draws = np.random.default_rng(seed), {key: [] for key in pairs}
    for _ in range(samples):
        selected = rng.choice(names, size=len(names), replace=True)
        for key, (numerator, denominator) in pairs.items():
            means = []
            for risk in (left, right):
                rows = [risk["per_video"][name] for name in selected]
                count = sum(row[denominator] for row in rows)
                means.append(sum(row[numerator] for row in rows) / count if count else None)
            if all(value is not None for value in means):
                draws[key].append(means[1] - means[0])
    return dict(unit="paired videos", direction="right minus left", samples=samples, seed=seed,
                metrics={key: dict(valid_draws=len(values), draws=values,
                                   ci95=np.quantile(values, [.025, .975]).tolist() if values else None)
                         for key, values in draws.items()})
