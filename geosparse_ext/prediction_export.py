"""Run as a file: model imports come from the training run's frozen snapshot."""
import argparse
import json
from pathlib import Path
import sys


def training_source(training):
    read = lambda name: json.loads((Path(training) / name).read_text())
    bindings, receipt, provenance = read("bindings.json"), read("result.json"), read("source_commits.json")
    if receipt.get("status") != "completed" or receipt.get("is_mock") is not False or receipt.get("completed_epochs") != 60:
        raise ValueError("prediction export requires its own completed 60-epoch training")
    if receipt.get("checkpoint_selection") == "best_full_validation_average_mAP" and not receipt.get("best_checkpoint_selection_complete"):
        raise ValueError("best checkpoint selection requires every scheduled full validation to finish")
    if bindings["source_commit"] != receipt["source_commit"] or any(receipt.get(key) != value for key, value in provenance.items()):
        raise ValueError("training receipt and source provenance differ")
    return bindings, receipt, provenance, read("job.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subset", choices=["internal_dev", "validation"], required=True)
    args = parser.parse_args()
    bindings, receipt, provenance, training_job = training_source(args.training_run)
    root = Path(bindings["repo_root"])
    sys.path.insert(0, str(root))
    # Executing this file directly keeps the measurement package out of model
    # imports. An evaluator update cannot silently replace the trained network.
    import copy
    import torch
    from mmengine.config import Config
    from geosparse_ext.records import source_commit, load_json, save_json, content_id, file_id
    from geosparse_ext.runtime import require_gpu, seed_all, loader, load_trained_model, merge_windows
    import geosparse_ext.detector as detector_module
    if source_commit(root) != receipt["source_commit"] or Path(detector_module.__file__).resolve().parents[1] != root.resolve():
        raise ValueError("model imports escaped the training snapshot")
    require_gpu()
    seed_all(training_job["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = load_json(args.training_run / "resolved_config.json")
    if content_id(resolved) != provenance["resolved_config_sha256"]:
        raise ValueError("training configuration changed")
    # The Python config preserves required tuple types (e.g. MMAction Resize).
    # Compare its JSON form with the run's checked resolved configuration.
    cfg = Config.fromfile(str(args.training_run / "resolved_opentad.py"))
    if json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]:
        raise ValueError("typed training configuration differs from its resolved record")
    if file_id(cfg.model.recognition_checkpoint) != provenance["weights_sha256"]:
        raise ValueError("recognition initialization asset changed")
    job = dict(training_job, kind="evaluate", depends_on=[training_job["job_id"]],
               dependency_outputs={training_job["job_id"]: str(args.training_run)})
    model = load_trained_model(job, cfg, args.output, provenance)
    split = load_json(Path(bindings["protocol_root"]) / training_job["dataset"] / "split.json")
    names = split[args.subset]
    model.inference_video_index = {name: i for i, name in enumerate(sorted(names))}
    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.subset_name = args.subset
    dataset_cfg.data_path = bindings["video_roots"]["training" if args.subset == "internal_dev" else "validation"]
    data = loader(dataset_cfg, bindings["runtime"]["evaluation_batch"], bindings["runtime"]["num_workers"], training_job["seed"])
    post = copy.deepcopy(cfg.post_processing)
    post.sliding_window = True
    raw = args.output / "raw_predictions"
    raw.mkdir(exist_ok=True)
    results = {name: [] for name in names}
    seen = set()
    with torch.no_grad():
        for step, batch in enumerate(data):
            seen.update(meta["video_name"] for meta in batch["metas"])
            with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                predictions = model.forward_test(**batch)
            plan = model.latest_route_plan
            torch.save(dict(proposals=[x.cpu() for x in predictions[0]], scores=[x.cpu() for x in predictions[1]],
                            metas=batch["metas"], selected_native=plan.selected_native.cpu(),
                            requested_budget=plan.requested_budget.cpu(),
                            heavy_trace=[] if model.geosparse.encoder is None else model.geosparse.encoder.trace),
                       raw / f"batch_{step:06d}.pth")
            for name, records in model.post_processing(predictions, batch["metas"], post, data.dataset.class_map).items():
                if name not in results:
                    raise ValueError("prediction video is outside the requested split")
                results[name].extend(records)
            if step % 20 == 0:
                print(json.dumps(dict(subset=args.subset, batch=step, total_batches=len(data))), flush=True)
    if seen != set(names):
        raise ValueError(f"prediction loader missed split videos: {sorted(set(names) - seen)}")
    save_json(args.output / "predictions.json", dict(results=merge_windows(results, cfg.post_processing.nms)))
    save_json(args.output / "prediction_receipt.json", dict(status="completed_prediction_export", is_mock=False,
              source_train_id=training_job["job_id"], model_source_commit=receipt["source_commit"], subset=args.subset,
              videos=names, training_checkpoint=receipt["checkpoint_path"], selected_checkpoint_epoch=model.epoch,
              training_provenance=provenance))


if __name__ == "__main__":
    main()
