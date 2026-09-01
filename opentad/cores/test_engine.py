import os
import copy
import hashlib
import json
import time
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def iter_limited_batches(iterable, max_batches=None):
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    for batch_idx, batch in enumerate(iterable):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


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
    max_batches=None,
    epoch=None,
):
    """Inference and Evaluation the model"""

    # GeoRoute uses this narrow, opt-in development profiler only for matched
    # route screening.  It deliberately records component timings rather than
    # claiming a paper-grade system cost; formal whole-system cost remains a
    # later frozen protocol.
    georoute_profile_cfg = cfg.get("georoute_development_profile", {})
    georoute_profile_enabled = bool(georoute_profile_cfg.get("enabled", False))
    if georoute_profile_enabled and not torch.cuda.is_available():
        raise ValueError("GeoRoute development profiling requires CUDA")
    georoute_telemetry_cfg = cfg.get("georoute_diagnostic_telemetry", {})
    georoute_telemetry_enabled = bool(
        georoute_telemetry_cfg.get("enabled", False)
    )
    formal_georoute_binding = cfg.get(
        "georoute_official_development_binding",
        cfg.get("georoute_telemetry_binding", None),
    )
    georoute_telemetry_sampler_indices = []
    if georoute_telemetry_enabled:
        if max_batches is not None:
            raise ValueError("GeoRoute diagnostic telemetry requires complete inference")
        expected_world_size = (
            int(formal_georoute_binding["world_size"])
            if formal_georoute_binding is not None
            else 1
        )
        if int(world_size) != expected_world_size:
            raise ValueError(
                "GeoRoute diagnostic telemetry world size differs from its "
                "registered protocol"
            )
        # ``solver.test.batch_size`` is job-global; the dataloader divides it
        # by world size.  Route telemetry is sample-level only when the local
        # batch is exactly one.
        if int(test_loader.batch_size) != 1:
            raise ValueError(
                "GeoRoute diagnostic telemetry requires one sample per rank"
            )
        georoute_telemetry_sampler_indices = list(iter(test_loader.sampler))
        if len(georoute_telemetry_sampler_indices) != len(test_loader):
            raise ValueError(
                "GeoRoute telemetry requires one sampler index per local batch"
            )
    georoute_samples = []
    georoute_telemetry_records = []
    previous_window_end = time.perf_counter()

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
    if "external_cls" in post_processing_cfg:
        if post_processing_cfg.external_cls != None:
            external_cls = build_classifier(post_processing_cfg.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    post_processing_cfg.sliding_window = isinstance(
        test_loader.dataset, SlidingWindowDataset
    )

    # model forward
    model.eval()
    result_dict = {}
    for batch_idx, data_dict in enumerate(
        tqdm.tqdm(
            iter_limited_batches(test_loader, max_batches=max_batches),
            disable=(rank != 0),
        )
    ):
        loader_ready = time.perf_counter()
        if georoute_profile_enabled:
            torch.cuda.synchronize()
            model_start = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=inference_cfg,
                    post_cfg=post_processing_cfg,
                    ext_cls=external_cls,
                )
        if georoute_profile_enabled:
            torch.cuda.synchronize()
            window_end = time.perf_counter()
            georoute_samples.append(
                {
                    "loader_wait_ms": (loader_ready - previous_window_end) * 1000.0,
                    "model_and_postprocess_ms": (window_end - model_start) * 1000.0,
                    "window_wall_ms": (window_end - previous_window_end) * 1000.0,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                }
            )
            previous_window_end = window_end
        if georoute_telemetry_enabled:
            unwrapped_model = getattr(model, "module", model)
            backbone = getattr(unwrapped_model, "backbone", None)
            latest_audit = getattr(backbone, "latest_georoute_audit", None)
            telemetry = (
                latest_audit.get("diagnostic_telemetry")
                if isinstance(latest_audit, dict)
                else None
            )
            if not isinstance(telemetry, dict):
                raise RuntimeError(
                    "GeoRoute diagnostic replay did not emit window telemetry"
                )
            dataset_index = int(
                georoute_telemetry_sampler_indices[batch_idx]
            )
            dataset_row = test_loader.dataset.data_list[dataset_index]
            video_id = str(dataset_row[0])
            centers = dataset_row[3]
            descriptor = {
                "dataset_index": dataset_index,
                "video_id": video_id,
                "window_center_count": int(len(centers)),
                "window_center_first": (
                    float(centers[0]) if len(centers) else None
                ),
                "window_center_last": (
                    float(centers[-1]) if len(centers) else None
                ),
            }
            if formal_georoute_binding is not None:
                descriptor.update(
                    rank=int(rank),
                    local_batch_index=int(batch_idx),
                )
            descriptor_bytes = json.dumps(
                descriptor, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            georoute_telemetry_records.append(
                {
                    **descriptor,
                    "window_descriptor_sha256": hashlib.sha256(
                        descriptor_bytes
                    ).hexdigest(),
                    "route": telemetry,
                }
            )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, post_processing_cfg)
    if georoute_telemetry_enabled and int(world_size) > 1:
        gathered_telemetry = [None for _ in range(int(world_size))]
        dist.all_gather_object(
            gathered_telemetry,
            georoute_telemetry_records,
        )
        georoute_telemetry_records = sorted(
            [
                record
                for rank_records in gathered_telemetry
                for record in rank_records
            ],
            key=lambda record: (
                int(record["dataset_index"]),
                int(record["rank"]),
                int(record["local_batch_index"]),
            ),
        )

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    metrics_dict = None
    if rank == 0:
        result_eval = dict(results=result_dict)
        evaluated_video_ids = sorted(
            {str(row[0]) for row in test_loader.dataset.data_list}
        )
        if "spatial_zoom_s1_test_binding" in cfg:
            if epoch is None or max_batches is not None:
                raise ValueError("formal S1 test evidence requires complete inference")
            from tools.bata.spatial_zoom_s1_evidence import write_s1_test_evidence

            write_s1_test_evidence(
                result_dict=result_dict,
                evaluated_video_ids=evaluated_video_ids,
                cfg=cfg,
                epoch=int(epoch),
            )
        elif "spatial_zoom_s1_runtime_binding" in cfg:
            if epoch is None or max_batches is not None:
                raise ValueError(
                    "formal S1 gate evidence requires a complete epoch evaluation"
                )
            from tools.bata.spatial_zoom_s1_evidence import write_s1_gate_evidence

            write_s1_gate_evidence(
                result_dict=result_dict,
                evaluated_video_ids=evaluated_video_ids,
                cfg=cfg,
                epoch=int(epoch),
            )
        if post_processing_cfg.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(
                dict(prediction_filename=result_eval, **cfg.evaluation)
            )
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)
        if georoute_profile_enabled:
            if not georoute_samples:
                raise RuntimeError("GeoRoute development profiler observed no inference windows")
            import numpy as np

            # The first item includes loader startup and is kept in the raw
            # list but excluded from the steady-state percentile whenever a
            # later window exists.  The report labels this scope explicitly.
            steady_samples = georoute_samples[1:] or georoute_samples
            profile_payload = {
                "schema_version": "georoute_development_profile_v1",
                "scope": {
                    "development_only": True,
                    "same_process_loader_wait": True,
                    "model_and_postprocess": True,
                    "evaluator_excluded": True,
                    "paper_grade_end_to_end_claim_allowed": False,
                    "diagnostic_route_telemetry_inside_timed_forward": bool(
                        cfg.model.backbone.custom.get(
                            "georoute_diagnostic_telemetry_enabled",
                            False,
                        )
                    ),
                    "separate_from_accuracy_evaluation": bool(
                        georoute_profile_cfg.get(
                            "separate_from_accuracy_evaluation",
                            False,
                        )
                    ),
                },
                "sample_count": len(georoute_samples),
                "steady_sample_count": len(steady_samples),
                "loader_wait_p50_ms": float(np.percentile([item["loader_wait_ms"] for item in steady_samples], 50)),
                "loader_wait_p95_ms": float(np.percentile([item["loader_wait_ms"] for item in steady_samples], 95)),
                "model_and_postprocess_p50_ms": float(np.percentile([item["model_and_postprocess_ms"] for item in steady_samples], 50)),
                "model_and_postprocess_p95_ms": float(np.percentile([item["model_and_postprocess_ms"] for item in steady_samples], 95)),
                "window_wall_p50_ms": float(np.percentile([item["window_wall_ms"] for item in steady_samples], 50)),
                "window_wall_p95_ms": float(np.percentile([item["window_wall_ms"] for item in steady_samples], 95)),
                "peak_allocated_mb": float(max(item["peak_allocated_mb"] for item in georoute_samples)),
                "samples": georoute_samples,
            }
            unwrapped_model = getattr(model, "module", model)
            backbone = getattr(unwrapped_model, "backbone", None)
            latest_audit = getattr(backbone, "latest_georoute_audit", None)
            if latest_audit is not None:
                profile_payload["last_georoute_audit"] = latest_audit
            profile_path = os.path.join(cfg.work_dir, "georoute_development_profile.json")
            with open(profile_path, "w", encoding="utf-8") as handle:
                json.dump(profile_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        if georoute_telemetry_enabled:
            dataset_count = len(test_loader.dataset.data_list)
            unique_dataset_indices = {
                int(record["dataset_index"])
                for record in georoute_telemetry_records
            }
            if unique_dataset_indices != set(range(dataset_count)):
                raise RuntimeError(
                    "GeoRoute diagnostic telemetry population is incomplete"
                )
            formal_world2 = formal_georoute_binding is not None
            if not formal_world2 and len(georoute_telemetry_records) != dataset_count:
                raise RuntimeError(
                    "single-rank GeoRoute telemetry repeated the population"
                )
            descriptor_keys = (
                (
                    "dataset_index",
                    "rank",
                    "local_batch_index",
                    "video_id",
                    "window_center_count",
                    "window_center_first",
                    "window_center_last",
                    "window_descriptor_sha256",
                )
                if formal_world2
                else (
                    "dataset_index",
                    "video_id",
                    "window_center_count",
                    "window_center_first",
                    "window_center_last",
                    "window_descriptor_sha256",
                )
            )
            population_bytes = json.dumps(
                [
                    {
                        key: record[key] for key in descriptor_keys
                    }
                    for record in georoute_telemetry_records
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            telemetry_payload = {
                "schema_version": (
                    "georoute_formal_development_telemetry_v1"
                    if formal_world2
                    else "georoute_diagnostic_telemetry_v1"
                ),
                "development_only": True,
                "official_test_opened": False,
                "gt_for_route_used": False,
                "teacher_for_route_used": False,
                "oracle_used": False,
                "raw_prediction_cache_used": False,
                "world_size": int(world_size),
                "local_batch_size": int(test_loader.batch_size),
                "dataset_count": dataset_count,
                "record_count": len(georoute_telemetry_records),
                "unique_dataset_count": len(unique_dataset_indices),
                "sampler_padding_count": (
                    len(georoute_telemetry_records) - dataset_count
                ),
                "population_sha256": hashlib.sha256(
                    population_bytes
                ).hexdigest(),
                "phase_m_binding": dict(
                    cfg.get("georoute_phase_m_binding", {})
                ),
                "records": georoute_telemetry_records,
            }
            telemetry_path = os.path.join(
                cfg.work_dir, "georoute_diagnostic_telemetry.json"
            )
            with open(telemetry_path, "w", encoding="utf-8") as handle:
                json.dump(telemetry_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    return metrics_dict


def gather_ddp_results(world_size, result_dict, post_cfg):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
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
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(
                segments, scores, labels, **post_cfg.nms
            )

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
    return result_dict
