"""Evaluate each run's frozen model; calibrate risk thresholds before test use."""
import copy
from pathlib import Path
import subprocess
import sys
import numpy as np
from .metrics import detection_risks, calibrate_risk_score
from .records import load_json, save_json, require_same_checkpoint
from .prediction_export import training_source


def duration_slices(risk, annotation, train_videos):
    durations = [event["segment"][1] - event["segment"][0] for name in train_videos
                 for event in annotation["database"][name].get("annotations", []) if event["segment"][1] > event["segment"][0]]
    q1, q3 = risk["short_threshold_seconds"], float(np.quantile(durations, .75))
    rows = {}
    for video in risk["per_video"].values():
        for event in video["per_gt"]:
            duration = event["segment"][1] - event["segment"][0]
            relative = "short_q1" if duration <= q1 else "middle_q2_q3" if duration <= q3 else "long_q4"
            absolute = next(label for end, label in [(2., "0-2s"), (5., "2-5s"), (10., "5-10s"), (30., "10-30s"), (float("inf"), "30s+")] if duration <= end)
            for label in (relative, absolute):
                row = rows.setdefault(label, dict(gt_count=0, class_aware_matches=0, capped_start_sum=0., capped_end_sum=0.))
                row["gt_count"] += 1
                row["class_aware_matches"] += int(event["class_aware_matched"])
                row["capped_start_sum"] += event["capped_start_seconds"] / duration
                row["capped_end_sum"] += event["capped_end_seconds"] / duration
    for row in rows.values():
        row.update(class_aware_recall=row["class_aware_matches"] / row["gt_count"],
                   capped_start_error_per_duration=row["capped_start_sum"] / row["gt_count"],
                   capped_end_error_per_duration=row["capped_end_sum"] / row["gt_count"])
    return dict(training_duration_q1_seconds=q1, training_duration_q3_seconds=q3, strata=rows)


def epoch_selection_supplements(training, provenance):
    """Re-use scheduled full-test predictions; never retrain for these reports."""
    rows = []
    for completed in (50, 60):
        row = load_json(Path(training) / "intermediate_eval" / f"epoch_{completed:03d}" / "metrics.json")
        if (row.get("status") != "completed_validation" or row.get("provenance") != provenance
                or row.get("completed_epochs") != completed or row.get("subset") != "validation" or row.get("weights") != "ema"):
            raise ValueError("supplement requires the scheduled full-test EMA result for its exact epoch")
        rows.append(row)
    common = max(rows, key=lambda row: (row["metrics"]["average_mAP"], -row["completed_epochs"]))
    report = lambda row: dict(completed_epochs=row["completed_epochs"], checkpoint_epoch=row["completed_epochs"] - 1,
                              metrics=row["metrics"], predictions_path=row["predictions_path"], weights="ema")
    return dict(amendment="audit-repair-20260908; supplemental analysis, not original preregistration",
                fixed_completed_60=report(rows[1]), common_completed_50_60_best=report(common),
                selection_caveat="full-test checkpoint selection; seed0 is not training-seed uncertainty")


def evaluate(job, cfg, output, provenance, runtime, split, thresholds):
    from .runtime import require_gpu
    from opentad.evaluations.builder import build_evaluator
    require_gpu()
    training = Path(job["dependency_outputs"][job["source_train_id"]])
    _, training_receipt, training_provenance, parent = training_source(training)
    if any(job[key] != parent[key] for key in ("model", "dataset", "seed")):
        raise ValueError("evaluation differs from its registered training configuration")
    for key in ("split_sha256", "weights_sha256"):
        if provenance[key] != training_provenance[key]:
            raise ValueError(f"evaluation {key} differs from its training")
    exports = {}
    for subset in ("internal_diagnostic", "validation"):
        destination = Path(output) / subset
        command = [sys.executable, str(Path(__file__).with_name("prediction_export.py")),
                   "--training-run", str(training), "--output", str(destination), "--subset", subset]
        subprocess.run(command, check=True)
        exports[subset] = load_json(destination / "predictions.json")
        require_same_checkpoint(training_receipt.get("checkpoint_identity"),
                                load_json(destination / "prediction_receipt.json").get("checkpoint_identity"))
        if set(exports[subset]["results"]) != set(split[subset]):
            raise ValueError("prediction export does not cover the exact split")
        if subset == "internal_diagnostic":
            annotation = load_json(cfg.evaluation.ground_truth_filename)
            calibration = calibrate_risk_score(annotation, exports[subset], split, thresholds["short_action_seconds"])
            # Freeze before the validation predictions are generated.
            save_json(Path(output) / "risk_calibration.json", calibration)
    parameters = copy.deepcopy(dict(cfg.evaluation))
    parameters["thread"] = 8
    official = build_evaluator(dict(prediction_filename=exports["validation"], **parameters)).evaluate()
    extra = dict(parameters, tiou_thresholds=[.8, .9])
    high = build_evaluator(dict(prediction_filename=exports["validation"], **extra)).evaluate()
    high.pop("average_mAP", None)  # Preserve the official .3-.7 average separately.
    risk = detection_risks(annotation, exports["validation"], split["validation"], thresholds["short_action_seconds"],
                           score_threshold=calibration["score_threshold"])
    metrics = dict(official={key: float(value) for key, value in official.items()},
                   checkpoint_identity=training_receipt["checkpoint_identity"],
                   epoch_selection_supplements=epoch_selection_supplements(training, training_provenance),
                   high_tiou={key: float(value) for key, value in high.items()}, risk=risk,
                   duration_slices=duration_slices(risk, annotation, split["training"]), metric_scale="0 to 1",
                   checkpoint_selection=training_receipt.get("checkpoint_selection", "final_epoch"),
                   selected_checkpoint_epoch=training_receipt.get("selected_checkpoint_epoch", 59),
                   selection_score=training_receipt.get("best_validation_score"))
    save_json(Path(output) / "metrics.json", metrics)
    return dict(metrics_path=str(Path(output) / "metrics.json"), predictions_path=str(Path(output) / "validation/predictions.json"),
                raw_predictions_path=str(Path(output) / "validation/raw_predictions"),
                risk_calibration_path=str(Path(output) / "risk_calibration.json"),
                model_source_commit=training_receipt["source_commit"], training_provenance=training_provenance,
                checkpoint_identity=training_receipt["checkpoint_identity"],
                checkpoint_selection=metrics["checkpoint_selection"], selected_checkpoint_epoch=metrics["selected_checkpoint_epoch"],
                measurement_source_commit=provenance["source_commit"])
