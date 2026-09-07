"""Full validation and best-checkpoint selection, with training state restored."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from .records import load_json, save_json, file_id, checkpoint_identity
from .prediction_export import evaluation_dataset


@contextmanager
def ema_evaluation(model, ema, seed, video_names):
    # EMA overwrites frozen weights too; restore every parameter it can touch.
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    missing = object()
    attributes = [(model, name) for name in ("epoch", "minibatch", "latest_route_plan",
                  "latest_acquisition", "pending_cost", "inference_seed", "inference_video_index")]
    attributes.append((model.geosparse, "execution_rng"))
    if model.geosparse.encoder is not None:
        attributes.append((model.geosparse.encoder, "trace"))
    previous = [(module, name, getattr(module, name, missing)) for module, name in attributes]
    try:
        with torch.no_grad():
            state = model.state_dict()
            for name, value in ema.shadow.items():
                state[name].copy_(value)
        model.eval()
        model.inference_seed = seed
        model.inference_video_index = {name: i for i, name in enumerate(sorted(video_names))}
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in parameters:
                    value.copy_(parameters[name])
            for name, value in buffers.items():
                parent, _, leaf = name.rpartition(".")
                module = model.get_submodule(parent) if parent else model
                module._buffers[leaf] = value
        for module, training in modes:
            module.training = training
        for module, name, value in previous:
            if value is missing:
                if hasattr(module, name):
                    delattr(module, name)
            else:
                setattr(module, name, value)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def validate_dataset(model, ema, cfg, runtime, job, provenance, output, completed_epochs, subset="validation"):
    from opentad.evaluations.builder import build_evaluator
    from .runtime import loader, merge_windows

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset_cfg, physical_subset = evaluation_dataset(cfg, subset)
    annotation = load_json(dataset_cfg.ann_file)
    names = sorted(name for name, row in annotation["database"].items() if row["subset"] == physical_subset)
    if not names:
        raise ValueError("periodic validation requires a nonempty registered split")
    row = dict(status="completed_validation", is_mock=False, is_final_result=False,
               source_train_id=job["job_id"], seed=job["seed"], completed_epochs=completed_epochs,
               subset=subset, weights="ema", videos=names, provenance=provenance,
               metric_scale="0 to 1")
    # An inference/evaluator failure is recorded after state restoration and does
    # not turn a measurement into a training promotion gate.
    with ema_evaluation(model, ema, job["seed"], names):
        try:
            data = loader(dataset_cfg, runtime["evaluation_batch"], runtime["num_workers"], job["seed"])
            post = copy.deepcopy(cfg.post_processing)
            post.sliding_window = True
            results, seen = {name: [] for name in names}, set()
            for step, batch in enumerate(data):
                seen.update(meta["video_name"] for meta in batch["metas"])
                with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                    predictions = model.forward_test(**batch)
                for name, records in model.post_processing(predictions, batch["metas"], post, data.dataset.class_map).items():
                    if name not in results:
                        raise ValueError("periodic validation prediction escaped its registered split")
                    results[name].extend(records)
                if step % 20 == 0:
                    print(json.dumps(dict(event="validation_progress", subset=subset, completed_epochs=completed_epochs,
                                          batch=step, total_batches=len(data))), flush=True)
            if seen != set(names):
                raise ValueError(f"periodic validation did not cover its exact split: seen={sorted(seen)}")
            predictions = dict(results=merge_windows(results, cfg.post_processing.nms))
            save_json(output / "predictions.json", predictions)
            parameters = copy.deepcopy(dict(cfg.evaluation))
            parameters.update(ground_truth_filename=dataset_cfg.ann_file, subset=physical_subset, thread=8)
            evaluator = build_evaluator(dict(prediction_filename=predictions, **parameters))
            metrics = {key: float(value) for key, value in evaluator.evaluate().items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("periodic validation returned nonfinite metrics")
            row.update(metrics=metrics, evaluated_classes=len(evaluator.activity_index),
                       windows=len(data.dataset),
                       predictions_path=str(output / "predictions.json"))
        except Exception as exc:
            row.update(status="failed_validation", reason=str(exc), traceback=traceback.format_exc())
    row["seconds"] = time.perf_counter() - started
    save_json(output / "metrics.json", row)
    print(json.dumps({key: value for key, value in row.items() if key not in {"videos", "provenance", "traceback"}}), flush=True)
    return row


def periodic_validation(model, ema, cfg, runtime, job, provenance, output, completed_epochs):
    if not completed_epochs or completed_epochs % 5:
        return None
    destination = Path(output) / "intermediate_eval" / f"epoch_{completed_epochs:03d}"
    receipt = destination / "metrics.json"
    if receipt.is_file():
        previous = load_json(receipt)
        if previous.get("provenance") != provenance or previous.get("completed_epochs") != completed_epochs:
            raise ValueError("intermediate validation belongs to another training run")
        if previous["status"] == "completed_validation":
            return previous
    return validate_dataset(model, ema, cfg, runtime, job, provenance, destination, completed_epochs)


def missing_validations(output, provenance):
    missing = []
    for completed in range(5, 61, 5):
        path = Path(output) / "intermediate_eval" / f"epoch_{completed:03d}" / "metrics.json"
        row = load_json(path) if path.exists() else {}
        if not (row.get("status") == "completed_validation" and row.get("provenance") == provenance
                and row.get("completed_epochs") == completed and row.get("subset") == "validation"):
            missing.append(completed)
    return missing


def repair_missing_validations(model, ema, cfg, runtime, job, provenance, output):
    """After optimization, retry only missing measurements of saved epoch weights.

    Re-entering the train job at epoch 60 invokes this path with no optimizer
    steps. Persistent evaluation failures remain selection_pending downstream.
    """
    from .runtime import restore_mutable_state
    for completed in missing_validations(output, provenance):
        path = Path(output) / "checkpoint" / f"epoch_{completed - 1}.pth"
        if not path.is_file():
            continue
        saved = torch.load(path, map_location="cpu")
        if saved["epoch"] != completed - 1 or saved["provenance"] != provenance:
            raise ValueError("validation repair checkpoint belongs to another epoch or run")
        restore_mutable_state(model, saved, "state_dict")
        device = next(model.parameters()).device
        ema.shadow = {n: saved["state_dict_ema"][n].to(device) for n in ema.names}
        model.epoch = saved["epoch"]
        row = periodic_validation(model, ema, cfg, runtime, job, provenance, output, completed)
        update_best_checkpoint(row, path, output)
    return missing_validations(output, provenance)


def update_best_checkpoint(validation, checkpoint, output):
    if validation is None or validation["status"] != "completed_validation":
        return
    if validation["subset"] != "validation" or validation["completed_epochs"] <= 0:
        raise ValueError("best selection requires full official validation during training")
    output = Path(output)
    score = validation["metrics"]["average_mAP"]
    record = output / "best.json"
    if record.is_file():
        previous = load_json(record)
        if previous["provenance"] != validation["provenance"]:
            raise ValueError("best checkpoint belongs to another training run")
        # A tie retains the earlier evaluated checkpoint.
        if previous["score"] >= score:
            return
    best = output / "checkpoint" / "best.pth"
    temporary = best.with_suffix(".tmp")
    shutil.copyfile(checkpoint, temporary)
    os.replace(temporary, best)
    save_json(record, dict(checkpoint_path=str(best.resolve()), completed_epochs=validation["completed_epochs"],
              checkpoint_epoch=validation["completed_epochs"] - 1, score=score,
              selection_metric="average_mAP@0.3:0.1:0.7", selection_subset="validation", weights="ema",
              source_train_id=validation["source_train_id"], seed=validation["seed"], provenance=validation["provenance"],
              checkpoint_identity=checkpoint_identity(validation["source_train_id"], validation["completed_epochs"] - 1,
                                                      validation["provenance"], file_id(best))))
