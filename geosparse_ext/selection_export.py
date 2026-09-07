"""Replay a checkpoint from its frozen model source; export selections without changing training.

Run this file directly, not with -m. A profiling forward is never a latency sample.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def thumbnails(meta, row, output):
    import decord
    import numpy as np
    from PIL import Image
    reader = decord.VideoReader(meta["geosparse"]["source_path"], num_threads=1)
    positions = np.flatnonzero(row["valid_tubelets"])
    positions = positions[np.unique(np.linspace(0, len(positions) - 1, min(8, len(positions))).astype(int))]
    images = []
    for position in positions:
        pair = []
        for member, source_id in enumerate(row["source_frame_id"][position]):
            image = Image.fromarray(reader[source_id].asnumpy())
            image.thumbnail((768, 432))
            path = output / "frames" / f"w{row['window_index']:06d}_t{position:03d}_{member}.jpg"
            path.parent.mkdir(exist_ok=True)
            image.save(path, quality=88)
            pair.append(str(path.relative_to(output)))
        images.append(dict(tubelet=int(position), paths=pair))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch", type=int, help="zero-based immutable epoch; explicitly interim, excluded from main figures")
    parser.add_argument("--videos", nargs="+", help="explicit qualitative subset; omitted means the entire official evaluation split")
    parser.add_argument("--thumbnail-videos", type=int, default=3, help="first sorted videos, fixed independently of scores")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("use an empty output directory so checkpoints cannot be mixed")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("geosparse_observer", root / "geosparse_ext/analysis_capture.py")
    observer_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer_code)
    record_spec = importlib.util.spec_from_file_location("measurement_records", root / "geosparse_ext/records.py")
    measurement_records = importlib.util.module_from_spec(record_spec)
    record_spec.loader.exec_module(measurement_records)
    bindings = read(args.training_run / "bindings.json")
    provenance = read(args.training_run / "source_commits.json")
    training_job = read(args.training_run / "job.json")
    frozen = Path(bindings["repo_root"])
    sys.path.insert(0, str(frozen))
    import numpy as np
    import torch
    from mmengine.config import Config
    from opentad.models.builder import build_detector
    from geosparse_ext.records import source_commit, save_json, content_id, file_id
    from geosparse_ext.runtime import require_gpu, seed_all, loader, load_trained_model, restore_mutable_state, merge_windows
    from geosparse_ext.prediction_export import training_source
    import geosparse_ext.detector as detector_code
    if source_commit(frozen) != provenance["source_commit"] or Path(detector_code.__file__).resolve().parents[1] != frozen.resolve():
        raise ValueError("model source differs from its training snapshot")
    require_gpu()
    seed_all(training_job["seed"])
    resolved = read(args.training_run / "resolved_config.json")
    cfg = Config.fromfile(str(args.training_run / "resolved_opentad.py"))
    if content_id(resolved) != provenance["resolved_config_sha256"] or json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]:
        raise ValueError("typed configuration differs from training")
    if args.checkpoint_epoch is None:
        _, receipt, _, _ = training_source(args.training_run)
        job = dict(training_job, kind="evaluate", depends_on=[training_job["job_id"]],
                   dependency_outputs={training_job["job_id"]: str(args.training_run)})
        model = load_trained_model(job, cfg, args.output, provenance)
        checkpoint_path = receipt["checkpoint_path"]
        final = True
    else:
        checkpoint_path = str(args.training_run / "checkpoint" / f"epoch_{args.checkpoint_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["epoch"] != args.checkpoint_epoch or checkpoint["provenance"] != provenance:
            raise ValueError("interim checkpoint provenance or epoch differs")
        model = build_detector(cfg.model).cuda().eval()
        restore_mutable_state(model, checkpoint, "state_dict_ema")
        model.epoch, model.inference_seed = checkpoint["epoch"], training_job["seed"]
        final = False
    split = read(Path(bindings["protocol_root"]) / training_job["dataset"] / "split.json")
    identity = measurement_records.checkpoint_identity(training_job["job_id"], model.epoch, provenance, file_id(checkpoint_path))
    all_names = sorted(split["validation"])
    names = sorted(args.videos) if args.videos else all_names
    if not set(names) <= set(all_names) or not names:
        raise ValueError("videos must belong to the registered official evaluation split")
    # Retain global indices even for a qualitative subset: indices seed routing.
    model.inference_video_index = {name: i for i, name in enumerate(all_names)}
    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.subset_name = "validation"
    dataset_cfg.data_path = bindings["video_roots"]["validation"]
    data = loader(dataset_cfg, 1, bindings["runtime"]["num_workers"], training_job["seed"])
    data.dataset.data_list = [row for row in data.dataset.data_list if row[0] in set(names)]
    post = copy.deepcopy(cfg.post_processing)
    post.sliding_window = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plans").mkdir()
    observer = observer_code.SelectionObserver(model)
    results, seen, pictured = {name: [] for name in names}, set(), set()
    try:
        with (args.output / "windows.jsonl").open("w", encoding="utf-8", buffering=1) as stream, torch.no_grad():
            for index, batch in enumerate(data):
                counter = observer_code.OperationMacCounter(model)
                with counter, torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                    predictions = model.forward_test(**batch)
                row, arrays = observer.record(training_job, counter.report(), index)
                meta = batch["metas"][0]
                name = meta["video_name"]
                if name in names[:max(0, args.thumbnail_videos)] and name not in pictured:
                    row["thumbnails"] = thumbnails(meta, row, args.output)
                    pictured.add(name)
                seen.add(name)
                path = f"plans/window_{index:06d}.npz"
                np.savez_compressed(args.output / path, **arrays)
                row["plan_path"] = path
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                for video, records in model.post_processing(predictions, batch["metas"], post, data.dataset.class_map).items():
                    results[video].extend(records)
                if index % 20 == 0:
                    print(json.dumps(dict(event="selection_export", windows=index + 1, total=len(data))), flush=True)
    finally:
        observer.close()
    if seen != set(names):
        raise ValueError("export did not cover every requested video")
    save_json(args.output / "predictions.json", dict(results=merge_windows(results, cfg.post_processing.nms)))
    annotations = read(dataset_cfg.ann_file)["database"]
    save_json(args.output / "annotations.json", {name: annotations[name] for name in names})
    save_json(args.output / "job.json", training_job)
    save_json(args.output / "export_receipt.json", dict(status="completed_selection_export", is_mock=False,
        is_final_checkpoint=final, source_train_id=training_job["job_id"], seed=training_job["seed"],
        model_source_commit=provenance["source_commit"], measurement_source_commit=source_commit(root),
        checkpoint_path=checkpoint_path, selected_checkpoint_epoch=model.epoch, weights="ema", videos=names,
        checkpoint_identity=identity,
        official_split_videos=len(all_names), full_split=names == all_names, windows=len(data),
        thumbnail_policy="first sorted videos; first window; eight evenly spaced native tubelets, both source frames",
        timing_scope="profiling pass; NOT a latency benchmark", training_provenance=provenance))


if __name__ == "__main__":
    main()
