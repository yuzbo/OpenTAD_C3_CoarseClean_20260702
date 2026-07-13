import os
import copy
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
    evaluation_epoch=None,
):
    """Inference and Evaluation the model"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict, evaluation_epoch=evaluation_epoch)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)
            metrics_path = cfg.evaluation.get("output_metrics_path", None)
            if metrics_path is None and cfg.post_processing.save_dict:
                metrics_path = os.path.join(cfg.work_dir, "evaluation_metrics.json")
            if metrics_path is not None:
                create_folder(os.path.dirname(metrics_path))
                metrics_payload = dict(metrics_dict, evaluation_epoch=evaluation_epoch)
                with open(metrics_path, "w", encoding="utf-8") as out:
                    json.dump(metrics_payload, out, indent=2, default=lambda value: value.item())


def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    return apply_sliding_window_nms(result_dict, post_cfg)


def apply_sliding_window_nms(result_dict, post_cfg):
    """Apply the same cross-window NMS used by the production evaluation loop."""
    if post_cfg.sliding_window is not True or post_cfg.nms is None:
        return result_dict
    merged = {}
    for video_name, detections in result_dict.items():
        if not detections:
            merged[video_name] = []
            continue
        segments = torch.tensor([item["segment"] for item in detections], dtype=torch.float32)
        scores = torch.tensor([item["score"] for item in detections], dtype=torch.float32)
        class_names = []
        labels = []
        for item in detections:
            if item["label"] not in class_names:
                class_names.append(item["label"])
            labels.append(class_names.index(item["label"]))
        labels = torch.tensor(labels, dtype=torch.float32)
        segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
        merged[video_name] = [
            {
                "segment": [round(value.item(), 2) for value in segment],
                "label": class_names[int(label.item())],
                "score": round(score.item(), 4),
            }
            for segment, label, score in zip(segments, labels, scores)
        ]
    return merged
