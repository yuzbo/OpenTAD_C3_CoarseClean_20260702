import os
import copy
import hashlib
import inspect
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


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
    result_dict = {}
    local_window_counts = {}
    local_model_forward_batch_count = 0
    local_input_sample_count = 0
    local_window_metadata_count = 0
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        batch_size = int(data_dict["masks"].shape[0])
        local_input_sample_count += batch_size
        metas = data_dict.get("metas", ())
        if isinstance(metas, (list, tuple)):
            for meta in metas:
                video_name = meta.get("video_name") if hasattr(meta, "get") else None
                if video_name is not None:
                    video_name = str(video_name)
                    local_window_counts[video_name] = (
                        int(local_window_counts.get(video_name, 0)) + 1
                    )
                    local_window_metadata_count += 1
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=inference_cfg,
                    post_cfg=post_processing_cfg,
                    ext_cls=external_cls,
                )
        local_model_forward_batch_count += 1

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict, post_processing_execution = gather_ddp_results(
        world_size,
        result_dict,
        post_processing_cfg,
        local_execution_stats={
            "window_counts": local_window_counts,
            "model_forward_batch_count": local_model_forward_batch_count,
            "input_sample_count": local_input_sample_count,
            "window_metadata_count": local_window_metadata_count,
        },
        return_execution_receipt=True,
    )

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        result_path = None
        post_processing_execution["pipeline_events"].insert(
            0, "model_forward_loop_complete"
        )
        dataset_source_identity = _source_identity(test_loader.dataset.__class__)
        dataset_length = int(len(test_loader.dataset))
        post_processing_execution.update(
            {
                "dataset_class": (
                    f"{test_loader.dataset.__class__.__module__}."
                    f"{test_loader.dataset.__class__.__qualname__}"
                ),
                "dataset_class_source_identity": dataset_source_identity,
                "dataset_length": dataset_length,
                "dataset_is_sliding_window": bool(
                    isinstance(test_loader.dataset, SlidingWindowDataset)
                ),
            }
        )
        if post_processing_cfg.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w", encoding="utf-8") as out:
                json.dump(result_eval, out)
            post_processing_execution["pipeline_events"].append(
                "post_nms_prediction_saved"
            )
        post_processing_execution.update(
            {
                "result_saved_after_nms": result_path is not None,
                "result_path": (
                    None if result_path is None else os.path.abspath(result_path)
                ),
                "result_sha256": (
                    None if result_path is None else _sha256_file(result_path)
                ),
            }
        )

        metrics_dict = None
        evaluator_identity = None
        evaluator_evaluate_called = False
        evaluator_evaluate_succeeded = False
        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            evaluator_evaluate_called = True
            post_processing_execution["pipeline_events"].append(
                "official_evaluator_evaluate_called"
            )
            metrics_dict = evaluator.evaluate()
            evaluator_evaluate_succeeded = True
            post_processing_execution["pipeline_events"].append(
                "official_evaluator_evaluate_returned"
            )
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
        evaluation_config = _jsonable(cfg.evaluation)
        post_processing_execution.update(
            {
                "evaluator_prediction_source": (
                    "in_memory_post_nms_result_object"
                ),
                "evaluation_config": evaluation_config,
                "evaluation_config_sha256": _canonical_sha256(
                    evaluation_config
                ),
                "evaluator": evaluator_identity,
                "evaluator_evaluate_called": evaluator_evaluate_called,
                "evaluator_evaluate_succeeded": evaluator_evaluate_succeeded,
            }
        )
        pipeline_identity = {
            "schema_version": "opentad_window_merge_nms_pipeline_identity_v1",
            "engine_source_identity": _source_identity(eval_one_epoch),
            "gather_source_identity": _source_identity(gather_ddp_results),
            "nms_callable_identity": post_processing_execution[
                "nms_callable_identity"
            ],
            "dataset_class": post_processing_execution["dataset_class"],
            "dataset_class_source_identity": dataset_source_identity,
            "dataset_is_sliding_window": post_processing_execution[
                "dataset_is_sliding_window"
            ],
            "nms_config": post_processing_execution["nms_config"],
            "nms_config_sha256": post_processing_execution[
                "nms_config_sha256"
            ],
            "evaluation_config": evaluation_config,
            "evaluation_config_sha256": post_processing_execution[
                "evaluation_config_sha256"
            ],
            "evaluator": evaluator_identity,
            "world_size": int(post_processing_execution["world_size"]),
        }
        post_processing_execution["pipeline_identity"] = pipeline_identity
        post_processing_execution["pipeline_identity_sha256"] = (
            _canonical_sha256(pipeline_identity)
        )
        post_processing_execution[
            "full_detector_window_merge_nms_evaluation_completed"
        ] = bool(
            dataset_length > 0
            and post_processing_execution["dataset_is_sliding_window"]
            and post_processing_execution["input_sample_count"] == dataset_length
            and post_processing_execution["window_metadata_count"]
            == dataset_length
            and sum(post_processing_execution["window_counts"].values())
            == dataset_length
            and post_processing_execution[
                "cross_window_result_aggregation_executed"
            ]
            and post_processing_execution["nms_applied"]
            and post_processing_execution["nms_call_count"]
            == post_processing_execution["pre_nms_video_count"]
            == post_processing_execution["post_nms_video_count"]
            == len(post_processing_execution["window_counts"])
            and post_processing_execution["result_saved_after_nms"]
            and evaluator_evaluate_called
            and evaluator_evaluate_succeeded
        )
        post_processing_execution["content_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in post_processing_execution.items()
                if key != "content_sha256"
            }
        )
        return {
            "metrics": metrics_dict,
            "result_path": None
            if result_path is None
            else os.path.abspath(result_path),
            "result_count": int(sum(len(items) for items in result_dict.values())),
            "video_count": int(len(result_dict)),
            "evaluator": evaluator_identity,
            "post_processing_execution": post_processing_execution,
        }
    return None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _source_identity(value):
    source_path = inspect.getsourcefile(value)
    return {
        "module": value.__module__,
        "qualname": value.__qualname__,
        "source_path": (
            None if source_path is None else os.path.abspath(source_path)
        ),
        "source_sha256": (
            None if source_path is None else _sha256_file(source_path)
        ),
    }


def _result_count(result_dict):
    return int(sum(len(items) for items in result_dict.values()))


def gather_ddp_results(
    world_size,
    result_dict,
    post_cfg,
    *,
    local_execution_stats=None,
    return_execution_receipt=False,
):
    world_size = int(world_size)
    if world_size <= 0:
        world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
    distributed = bool(dist.is_initialized())
    if world_size > 1 and not distributed:
        raise RuntimeError(
            "multi-process result gathering requires an initialized process group"
        )
    gather_dict_list = [None for _ in range(world_size)]
    if distributed:
        dist.all_gather_object(gather_dict_list, result_dict)
    else:
        gather_dict_list[0] = result_dict
    gathered_result_count_by_rank = [
        _result_count(value) for value in gather_dict_list
    ]
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    execution_stats_by_rank = None
    if local_execution_stats is not None:
        execution_stats_by_rank = [None for _ in range(world_size)]
        if distributed:
            dist.all_gather_object(
                execution_stats_by_rank,
                _jsonable(local_execution_stats),
            )
        else:
            execution_stats_by_rank[0] = _jsonable(local_execution_stats)
    merged_window_counts = {}
    model_forward_batch_count = 0
    input_sample_count = 0
    window_metadata_count = 0
    for stats in execution_stats_by_rank or ():
        model_forward_batch_count += int(stats["model_forward_batch_count"])
        input_sample_count += int(stats["input_sample_count"])
        window_metadata_count += int(stats["window_metadata_count"])
        for video_name, count in stats["window_counts"].items():
            merged_window_counts[str(video_name)] = (
                int(merged_window_counts.get(str(video_name), 0)) + int(count)
            )

    pre_nms_counts = {
        str(video_name): int(len(items))
        for video_name, items in sorted(result_dict.items())
    }
    nms_config = _jsonable(post_cfg.nms)
    nms_applied = bool(
        post_cfg.sliding_window is True and post_cfg.nms is not None
    )
    nms_call_count = 0
    # do nms for sliding window, if needed
    if nms_applied:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            labels = []
            class_idx = []
            if v:
                segments = torch.tensor(
                    [data["segment"] for data in v],
                    dtype=torch.float32,
                ).reshape(-1, 2)
                scores = torch.tensor(
                    [data["score"] for data in v],
                    dtype=torch.float32,
                )
                for data in v:
                    if data["label"] not in class_idx:
                        class_idx.append(data["label"])
                    labels.append(class_idx.index(data["label"]))
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                segments = torch.empty((0, 2), dtype=torch.float32)
                scores = torch.empty((0,), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.long)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
            nms_call_count += 1

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
        result_dict = tmp_result_dict
    post_nms_counts = {
        str(video_name): int(len(items))
        for video_name, items in sorted(result_dict.items())
    }
    receipt = {
        "schema_version": "opentad_window_merge_nms_execution_v1",
        "world_size": int(world_size),
        "model_forward_batch_count": int(model_forward_batch_count),
        "input_sample_count": int(input_sample_count),
        "window_metadata_count": int(window_metadata_count),
        "window_counts": dict(sorted(merged_window_counts.items())),
        "gathered_result_count_by_rank": gathered_result_count_by_rank,
        "cross_window_result_aggregation_executed": True,
        "pre_nms_video_count": int(len(pre_nms_counts)),
        "pre_nms_result_count": int(sum(pre_nms_counts.values())),
        "pre_nms_result_count_by_video": pre_nms_counts,
        "nms_config": nms_config,
        "nms_config_sha256": _canonical_sha256(nms_config),
        "nms_callable_identity": _source_identity(batched_nms),
        "nms_applied": nms_applied,
        "nms_call_count": int(nms_call_count),
        "post_nms_video_count": int(len(post_nms_counts)),
        "post_nms_result_count": int(sum(post_nms_counts.values())),
        "post_nms_result_count_by_video": post_nms_counts,
        "pipeline_events": [
            "ddp_result_gather_complete",
            "cross_window_result_aggregation_complete",
            *(
                ["sliding_window_nms_complete"]
                if nms_applied
                else []
            ),
        ],
    }
    if return_execution_receipt:
        return result_dict, receipt
    return result_dict
