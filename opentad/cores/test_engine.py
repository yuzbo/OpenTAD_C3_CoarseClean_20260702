import os
import copy
import hashlib
import inspect
import json
import time
from collections import Counter

import numpy as np
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def summarize_duca_execution_cost(
    rows,
    *,
    peak_cuda_memory_mb,
    final_postprocess_ms_by_video=None,
    full_population_wall_ms=None,
):
    """Aggregate label-free per-window execution records into per-video cost."""

    if not rows:
        raise ValueError("DUCA execution cost requires at least one inference window")
    per_video = {}
    budget_counts = Counter()
    for row in rows:
        video_name = str(row["video_name"])
        requested = int(row["requested_budget"])
        budget_counts[requested] += 1
        aggregate = per_video.setdefault(
            video_name,
            {
                "window_count": 0,
                "actual_observations": 0,
                "execution_slots": 0,
                "wall_ms": 0.0,
                "data_wait_ms": 0.0,
                "model_and_window_postprocess_ms": 0.0,
                "final_video_postprocess_ms": 0.0,
                "selector_ms": 0.0,
                "videomae_ms": 0.0,
                "detector_ms": 0.0,
            },
        )
        aggregate["window_count"] += 1
        aggregate["actual_observations"] += int(row["actual_observations"])
        aggregate["execution_slots"] += int(row["execution_slots"])
        if "data_wait_ms" in row or "model_and_window_postprocess_ms" in row:
            data_wait_ms = float(row.get("data_wait_ms", 0.0))
            model_ms = float(row.get("model_and_window_postprocess_ms", 0.0))
            aggregate["data_wait_ms"] += data_wait_ms
            aggregate["model_and_window_postprocess_ms"] += model_ms
            aggregate["wall_ms"] += data_wait_ms + model_ms
        else:
            # Preserve the small pure-aggregation helper's legacy input shape.
            aggregate["model_and_window_postprocess_ms"] += float(row["wall_ms"])
            aggregate["wall_ms"] += float(row["wall_ms"])
        aggregate["selector_ms"] += float(row["selector_ms"])
        aggregate["videomae_ms"] += float(row["videomae_ms"])
        aggregate["detector_ms"] += float(row["detector_ms"])
    final_postprocess_ms_by_video = final_postprocess_ms_by_video or {}
    unknown_postprocess_videos = set(final_postprocess_ms_by_video) - set(per_video)
    if unknown_postprocess_videos:
        raise ValueError("final post-processing contains a video absent from inference")
    for video_name, value in per_video.items():
        final_ms = float(final_postprocess_ms_by_video.get(video_name, 0.0))
        value["final_video_postprocess_ms"] = final_ms
        value["wall_ms"] += final_ms
    wall = np.asarray([value["wall_ms"] for value in per_video.values()], dtype=np.float64)
    attributed_wall_ms = float(wall.sum())

    def _component_percentiles(field):
        values = np.asarray(
            [value[field] for value in per_video.values()], dtype=np.float64
        )
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "schema_version": "duca_h65_system_multibudget_execution_cost_v2",
        "held_out_semantics_read": False,
        "measurement_scope": (
            "dataloader_consumer_wait_plus_model_window_postprocess_plus_final_video_nms"
        ),
        "data_wait_is_realized_prefetch_consumer_wait": True,
        "window_count": len(rows),
        "video_count": len(per_video),
        "requested_budget_window_counts": {
            str(key): int(value) for key, value in sorted(budget_counts.items())
        },
        "total_actual_observations": int(
            sum(value["actual_observations"] for value in per_video.values())
        ),
        "total_execution_slots": int(
            sum(value["execution_slots"] for value in per_video.values())
        ),
        "per_video_wall_ms_p50": float(np.percentile(wall, 50)),
        "per_video_wall_ms_p95": float(np.percentile(wall, 95)),
        "per_video_component_ms": {
            field: _component_percentiles(field)
            for field in (
                "data_wait_ms",
                "selector_ms",
                "videomae_ms",
                "detector_ms",
                "model_and_window_postprocess_ms",
                "final_video_postprocess_ms",
                "wall_ms",
            )
        },
        "attributed_per_video_wall_ms_sum": attributed_wall_ms,
        "full_population_wall_ms": None
        if full_population_wall_ms is None
        else float(full_population_wall_ms),
        "unattributed_framework_wall_ms": None
        if full_population_wall_ms is None
        else float(max(float(full_population_wall_ms) - attributed_wall_ms, 0.0)),
        "peak_cuda_memory_mb": float(peak_cuda_memory_mb),
        "per_video": {key: per_video[key] for key in sorted(per_video)},
        "per_window": rows,
    }


def _write_json_atomic(path, value):
    target = os.path.abspath(os.path.expanduser(str(path)))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    inference_cfg = copy.deepcopy(cfg.inference)
    post_processing_cfg = copy.deepcopy(cfg.post_processing)
    execution_cost_path = inference_cfg.get("execution_cost_json", None)
    if execution_cost_path and world_size != 1:
        raise ValueError("DUCA execution-cost sealing requires one inference process")
    inference_cfg["folder"] = os.path.join(cfg.work_dir, "outputs")
    if inference_cfg.save_raw_prediction:
        create_folder(inference_cfg["folder"])

    # external classifier
    external_cls = test_loader.dataset.class_map
    if (
        "external_cls" in post_processing_cfg
        and post_processing_cfg.external_cls is not None
    ):
        external_cls = build_classifier(post_processing_cfg.external_cls)

    # whether the testing dataset is sliding window
    post_processing_cfg.sliding_window = isinstance(
        test_loader.dataset, SlidingWindowDataset
    )

    # model forward
    model.eval()
    if execution_cost_path and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result_dict = {}
    execution_rows = []
    execution_run_start = time.perf_counter() if execution_cost_path else None
    progress = iter(tqdm.tqdm(test_loader, disable=(rank != 0)))
    while True:
        data_wait_start = time.perf_counter()
        try:
            data_dict = next(progress)
        except StopIteration:
            break
        if execution_cost_path:
            torch.cuda.synchronize()
            batch_ready = time.perf_counter()
            data_wait_ms = (batch_ready - data_wait_start) * 1000.0
            batch_start = batch_ready
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=inference_cfg,
                    post_cfg=post_processing_cfg,
                    ext_cls=external_cls,
                )
        if execution_cost_path:
            torch.cuda.synchronize()
            model_and_window_postprocess_ms = (
                time.perf_counter() - batch_start
            ) * 1000.0
            root_model = getattr(model, "module", model)
            runtime_metas = getattr(root_model, "_last_forward_test_metas", None)
            if not isinstance(runtime_metas, list) or len(runtime_metas) != 1:
                raise RuntimeError(
                    "execution-cost inference is frozen to batch_size=1 with one metadata row"
                )
            meta = runtime_metas[0]
            runtime = meta.get("duca_runtime_profile")
            if not isinstance(runtime, dict):
                raise RuntimeError("DUCA runtime profiling must be enabled for cost sealing")
            execution_rows.append(
                {
                    "video_name": str(meta.get("video_name", "")),
                    "window_start_frame": int(meta.get("window_start_frame", 0)),
                    "requested_budget": int(meta["duca_online_requested_budget"]),
                    "effective_budget": int(meta["duca_online_effective_budget"]),
                    "actual_observations": int(meta["duca_actual_heavy_observations"]),
                    "execution_slots": int(meta["duca_heavy_execution_slots"]),
                    "data_wait_ms": float(data_wait_ms),
                    "model_and_window_postprocess_ms": float(
                        model_and_window_postprocess_ms
                    ),
                    "selector_ms": float(runtime["selector_ms"]),
                    "videomae_ms": float(runtime["videomae_ms"]),
                    "detector_ms": float(runtime["detector_ms"]),
                }
            )
        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    if execution_cost_path:
        result_dict, final_postprocess_ms_by_video = gather_ddp_results(
            world_size,
            result_dict,
            post_processing_cfg,
            return_postprocess_timings=True,
        )
        torch.cuda.synchronize()
        full_population_wall_ms = (
            time.perf_counter() - execution_run_start
        ) * 1000.0
    else:
        result_dict = gather_ddp_results(world_size, result_dict, post_processing_cfg)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        result_path = None
        if post_processing_cfg.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w", encoding="utf-8") as out:
                json.dump(result_eval, out)

        if execution_cost_path:
            if result_path is None:
                raise RuntimeError("execution-cost sealing requires post_processing.save_dict=True")
            cost_summary = summarize_duca_execution_cost(
                execution_rows,
                peak_cuda_memory_mb=float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                final_postprocess_ms_by_video=final_postprocess_ms_by_video,
                full_population_wall_ms=full_population_wall_ms,
            )
            cost_summary["prediction_path"] = os.path.abspath(result_path)
            cost_summary["prediction_sha256"] = _sha256_file(result_path)
            _write_json_atomic(execution_cost_path, cost_summary)

        metrics_dict = None
        evaluator_identity = None
        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)
            source_path = inspect.getsourcefile(evaluator.__class__)
            evaluator_identity = {
                "module": evaluator.__class__.__module__,
                "class_name": evaluator.__class__.__qualname__,
                "source_path": None
                if source_path is None
                else os.path.abspath(source_path),
                "source_sha256": None
                if source_path is None
                else _sha256_file(source_path),
            }
        return {
            "metrics": metrics_dict,
            "result_path": None
            if result_path is None
            else os.path.abspath(result_path),
            "result_count": int(sum(len(items) for items in result_dict.values())),
            "video_count": int(len(result_dict)),
            "evaluator": evaluator_identity,
        }
    return None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gather_ddp_results(
    world_size, result_dict, post_cfg, *, return_postprocess_timings=False
):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # do nms for sliding window, if needed
    postprocess_timings = {}
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            if return_postprocess_timings and torch.cuda.is_available():
                torch.cuda.synchronize()
            postprocess_start = time.perf_counter()
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
            if return_postprocess_timings and torch.cuda.is_available():
                torch.cuda.synchronize()
            if return_postprocess_timings:
                postprocess_timings[str(k)] = (
                    time.perf_counter() - postprocess_start
                ) * 1000.0
        result_dict = tmp_result_dict
    if return_postprocess_timings:
        return result_dict, postprocess_timings
    return result_dict
