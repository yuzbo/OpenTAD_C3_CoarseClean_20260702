"""Run as a file: fixed-q inference of a completed dynamic-A EMA, with same-pass cost.

This is an intervention on an existing checkpoint, not a trained constrained
allocator. The frozen training snapshot supplies every model and evaluator import.
"""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


DIAGNOSTIC_ID = "sci3-a-dynamic-best-fixed-q050-20260908"
OVERRIDE = {"budget_mode": "fixed", "budget": 0.5}


@contextmanager
def fixed_half_budget(model):
    config = model.geosparse.config
    if model.training or config["route"] != "A" or config["budget_mode"] != "dynamic":
        raise ValueError("this diagnostic requires an eval-mode dynamic A model")
    if config["selector"] not in {"hybrid", "pg_only"}:
        raise ValueError("fixed-q intervention requires the learned source selector")
    original = {key: config[key] for key in OVERRIDE}
    try:
        config.update(OVERRIDE)
        yield
    finally:
        config.update(original)


def add_eligible_heavy_reference(row):
    """A's same-resolution full Heavy on eligible tubelets, alongside padded full."""
    if row["route"] != "A":
        raise ValueError("this denominator helper is for route A")
    temporal, height, width = row["native_shape"]
    valid = row["valid_tubelets"]
    parent_t = row["heavy_reference"]["parent_shape"][0]
    if len(valid) != temporal or temporal % parent_t:
        raise ValueError("native validity does not match the parent grid")
    d, depth = row["heavy_reference"]["width"], row["heavy_reference"]["depth"]
    counts = [sum(valid[start:start + parent_t]) * height * width
              for start in range(0, temporal, parent_t)]
    reference = depth * sum(12 * k * d * d + 2 * k * k * d for k in counts)
    row.update(heavy_eligible_full_macs=reference,
               heavy_eligible_full_ratio=row["heavy_macs"] / reference if reference else None,
               eligible_parent_tokens=counts,
               heavy_padded_full_ratio=row["heavy_mac_ratio"])
    return row


def cost_summary(rows):
    import numpy as np
    if not rows or any(row["requested_budget"] != 0.5 for row in rows):
        raise ValueError("every measured window must execute fixed q=.5")
    total = lambda key: sum(row[key] for row in rows)
    distribution = {}
    for key in ("selected_native_ratio", "heavy_eligible_full_ratio", "heavy_padded_full_ratio"):
        values = [row[key] for row in rows if row[key] is not None]
        distribution[key] = dict(unit="window", count=len(values),
            mean=float(np.mean(values)) if values else None,
            quantiles=dict(zip(("min", "p25", "p50", "p75", "p95", "max"),
                              np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist())) if values else {})
    heavy = total("heavy_macs")
    eligible, padded = total("heavy_eligible_full_macs"), total("heavy_reference_macs")
    return dict(windows=len(rows), requested_q=.5, heavy_macs=heavy,
        heavy_eligible_full_macs=eligible, heavy_padded_full_macs=padded,
        ratio_of_sums_eligible_full=heavy / eligible if eligible else None,
        ratio_of_sums_padded_full=heavy / padded if padded else None,
        model_conv_linear_matmul_macs_counted=total("model_macs_counted"),
        distributions=distribution,
        scope="Heavy formula from actual per-layer/per-parent QKV counts; positive attention quadratic term; q is not a MAC ratio; instrumented passes are not latency measurements")


def run(args):
    # No geosparse_ext imports before the recorded model root is selected.
    training = args.training_run.resolve()
    bindings = json.loads((training / "bindings.json").read_text())
    root = Path(bindings["repo_root"]).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from mmengine.config import Config
    from geosparse_ext.prediction_export import training_source, evaluation_dataset
    from geosparse_ext.records import load_json, save_json, source_commit, content_id, file_id, require_same_checkpoint
    from geosparse_ext.runtime import require_gpu, seed_all, loader, load_trained_model, merge_windows
    from geosparse_ext.analysis_capture import SelectionObserver, OperationMacCounter
    from geosparse_ext.metrics import detection_risks
    from geosparse_ext.evaluation import duration_slices
    from opentad.evaluations.builder import build_evaluator
    import geosparse_ext.detector as detector_module

    bindings, receipt, provenance, training_job = training_source(training)
    if (source_commit(root) != receipt["source_commit"]
            or Path(detector_module.__file__).resolve().parents[1] != root):
        raise ValueError("model imports escaped the frozen training snapshot")
    measurement_commit = source_commit(Path(__file__).resolve().parents[1])
    if training_job["route"] != "A" or training_job["model"]["budget_mode"] != "dynamic":
        raise ValueError("expected the completed dynamic-A training")
    require_gpu()
    seed_all(training_job["seed"])
    resolved = load_json(training / "resolved_config.json")
    cfg = Config.fromfile(str(training / "resolved_opentad.py"))
    if (content_id(resolved) != provenance["resolved_config_sha256"]
            or json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]
            or file_id(cfg.model.recognition_checkpoint) != provenance["weights_sha256"]):
        raise ValueError("typed configuration or initialization differs from training")
    protocol = Path(bindings["protocol_root"]) / training_job["dataset"]
    split = load_json(protocol / "split.json")
    if content_id(split) != provenance["split_sha256"] or len(split["training"]) != 200:
        raise ValueError("diagnostic requires the recorded full200 training split")
    dataset_cfg, physical_subset = evaluation_dataset(cfg, "validation")
    annotation = load_json(dataset_cfg.ann_file)
    names = split["validation"]
    if len(names) != 211 or set(names) != {name for name, row in annotation["database"].items() if row["subset"] == physical_subset}:
        raise ValueError("diagnostic requires all 211 official test videos")

    # Reuse the already frozen training-derived risk threshold. No test fitting.
    reference = args.reference_evaluation.resolve()
    reference_receipt = load_json(reference / "result.json")
    reference_metrics = load_json(reference / "metrics.json")
    if reference_receipt.get("status") != "completed" or reference_receipt.get("is_mock") is not False:
        raise ValueError("reference independent evaluation is incomplete")
    require_same_checkpoint(receipt["checkpoint_identity"], reference_receipt["checkpoint_identity"])
    require_same_checkpoint(receipt["checkpoint_identity"], reference_metrics["checkpoint_identity"])
    calibration = load_json(reference / "risk_calibration.json")
    if set(calibration["videos"]) != set(split["training"]):
        raise ValueError("risk threshold was not calibrated on the full training pool")
    thresholds = load_json(protocol / "risk_thresholds.json")
    data = loader(dataset_cfg, 1, bindings["runtime"]["num_workers"], training_job["seed"])
    if len(data.dataset) != 792 or len(data) != 792:
        raise ValueError("diagnostic requires 792 individual test windows")

    args.output.mkdir(parents=True, exist_ok=False)
    job = dict(training_job, kind="evaluate", depends_on=[training_job["job_id"]],
               dependency_outputs={training_job["job_id"]: str(training)})
    model = load_trained_model(job, cfg, args.output, provenance)
    model.inference_video_index = {name: i for i, name in enumerate(sorted(names))}
    identity = dict(diagnostic_id=DIAGNOSTIC_ID, is_mock=False, training_run=str(training),
        source_train_id=training_job["job_id"], checkpoint_identity=model.checkpoint_identity,
        selected_checkpoint_epoch=model.epoch, training_provenance=provenance,
        model_source_commit=receipt["source_commit"], measurement_source_commit=measurement_commit,
        original_model_config=copy.deepcopy(model.geosparse.config), measurement_override=OVERRIDE,
        reference_evaluation=str(reference), reference_official=reference_metrics["official"],
        risk_calibration=calibration, subset="validation", expected_videos=211, expected_windows=792,
        seed=training_job["seed"], weights="ema", new_training=False,
        interpretation="same trained dynamic-A weights with inference fixed q=.5; not a learned constrained budget method or a MAC=.5 result")
    save_json(args.output / "measurement.json", identity)
    save_json(args.output / "risk_calibration.json", calibration)
    post = copy.deepcopy(cfg.post_processing)
    post.sliding_window = True
    results = {name: [] for name in names}
    rows, seen, windows = [], set(), set()
    raw = args.output / "raw_predictions"
    raw.mkdir()
    plans = args.output / "plans"
    plans.mkdir()
    observer = SelectionObserver(model)
    try:
        with fixed_half_budget(model), torch.no_grad(), (args.output / "windows.jsonl").open("w") as stream:
            for step, batch in enumerate(data):
                with OperationMacCounter(model) as counter, torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                    predictions = model.forward_test(**batch)
                row, arrays = observer.record(training_job, counter.report(), step)
                add_eligible_heavy_reference(row)
                window = (row["video_id"], row["window_id"])
                if window in windows or row["video_id"] not in results or row["requested_budget"] != .5:
                    raise ValueError("duplicate/foreign window or unapplied budget override")
                windows.add(window)
                seen.add(row["video_id"])
                row["measurement_override"] = OVERRIDE
                np.savez_compressed(plans / f"window_{step:06d}.npz", **arrays)
                plan = model.latest_route_plan
                torch.save(dict(proposals=[x.cpu() for x in predictions[0]], scores=[x.cpu() for x in predictions[1]],
                    metas=batch["metas"], selected_native=plan.selected_native.cpu(),
                    requested_budget=plan.requested_budget.cpu(), heavy_trace=row["layers"]),
                    raw / f"batch_{step:06d}.pth")
                if args.precheck_only:
                    # Same actual GPU inputs, with/without observers/counter.
                    observer.close()
                    selected = plan.selected_native.clone()
                    with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                        unobserved = model.forward_test(**batch)
                    for left, right in zip(predictions[0] + predictions[1], unobserved[0] + unobserved[1]):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    torch.testing.assert_close(selected, model.latest_route_plan.selected_native, atol=0, rtol=0)
                for name, records in model.post_processing(predictions, batch["metas"], post, data.dataset.class_map).items():
                    if name not in results:
                        raise ValueError("prediction video escaped the test split")
                    results[name].extend(records)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if step % 20 == 0:
                    print(json.dumps(dict(completed_windows=step + 1, expected_windows=792, q=.5)), flush=True)
                if args.precheck_only:
                    break
        if args.precheck_only:
            save_json(args.output / "result.json", dict(identity, status="completed_gpu_precheck",
                instrumented_forward_equal=True, measured_windows=1, cost=cost_summary(rows),
                scientific_evaluation_complete=False))
            return
        if seen != set(names) or len(windows) != 792:
            raise ValueError("full diagnostic coverage incomplete")
        prediction = dict(results=merge_windows(results, cfg.post_processing.nms))
        save_json(args.output / "predictions.json", prediction)
        parameters = copy.deepcopy(dict(cfg.evaluation))
        parameters["thread"] = 8
        official = build_evaluator(dict(prediction_filename=prediction, **parameters)).evaluate()
        high = build_evaluator(dict(prediction_filename=prediction, **dict(parameters, tiou_thresholds=[.8, .9]))).evaluate()
        high.pop("average_mAP", None)
        risk = detection_risks(annotation, prediction, names, thresholds["short_action_seconds"], calibration["score_threshold"])
        metrics = dict(official={k: float(v) for k, v in official.items()}, high_tiou={k: float(v) for k, v in high.items()},
            risk=risk, duration_slices=duration_slices(risk, annotation, split["training"]),
            checkpoint_identity=model.checkpoint_identity, measurement_override=OVERRIDE, metric_scale="0 to 1",
            difference_from_same_weight_dynamic={k: float(v) - reference_metrics["official"][k] for k, v in official.items()})
        save_json(args.output / "metrics.json", metrics)
        save_json(args.output / "cost_summary.json", cost_summary(rows))
        save_json(args.output / "result.json", dict(identity, status="completed_diagnostic", scientific_evaluation_complete=True,
            measured_videos=sorted(seen), measured_windows=len(windows),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            metrics_path=str(args.output / "metrics.json"), cost_path=str(args.output / "cost_summary.json"),
            prediction_path=str(args.output / "predictions.json"), cost_and_predictions_same_forward=True))
    except Exception as error:
        save_json(args.output / "failure.json", dict(identity, status="failed", completed_windows=len(rows),
            error_type=type(error).__name__, error=str(error), scientific_evaluation_complete=False))
        raise
    finally:
        observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precheck-only", action="store_true")
    run(parser.parse_args())
