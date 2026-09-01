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
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=inference_cfg,
                    post_cfg=post_processing_cfg,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

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
        result_dict = tmp_result_dict
    return result_dict
