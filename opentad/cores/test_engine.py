import os
import copy
import gzip
import hashlib
import json
import math
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


class InvalidProposalError(RuntimeError):
    """Raised before NMS when an unfiltered proposal set is not numerically valid."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


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
                result_dict[k] = list(v)

    save_pre_cross_window = bool(
        _post_cfg_get(cfg.post_processing, "save_pre_cross_window_detections", False)
    )
    save_post_processing_audit = bool(
        _post_cfg_get(cfg.post_processing, "save_post_processing_audit", False)
    )
    if save_pre_cross_window or save_post_processing_audit:
        result_dict, post_processing_audit, pre_cross_window_result_dict = gather_ddp_results(
            world_size,
            result_dict,
            cfg.post_processing,
            return_audit=True,
            return_pre_cross_window=True,
        )
    else:
        result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)
        post_processing_audit = None
        pre_cross_window_result_dict = None

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        pre_cross_window_artifact = None
        if save_pre_cross_window:
            pre_cross_window_path = _post_cfg_get(
                cfg.post_processing,
                "pre_cross_window_detections_path",
                os.path.join(cfg.work_dir, "pre_cross_window_detections.json.gz"),
            )
            pre_cross_window_payload = {
                "schema_version": "opentad_pre_cross_window_detections_v1",
                "artifact_kind": "pre_cross_window_nms_full_precision_detections",
                "evaluation_epoch": evaluation_epoch,
                "git_commit": os.environ.get("PHYSTIME_EXPECTED_COMMIT"),
                "git_tree": os.environ.get("PHYSTIME_EXPECTED_TREE"),
                "results": pre_cross_window_result_dict,
            }
            _atomic_write_json_gzip(pre_cross_window_path, pre_cross_window_payload)
            pre_cross_window_artifact = {
                "path": os.path.abspath(pre_cross_window_path),
                "sha256": _sha256_file(pre_cross_window_path),
            }

        if save_post_processing_audit:
            audit_path = _post_cfg_get(
                cfg.post_processing,
                "post_processing_audit_path",
                os.path.join(cfg.work_dir, "post_processing_audit.json"),
            )
            audit_payload = {
                "schema_version": "opentad_post_processing_audit_v1",
                "evaluation_epoch": evaluation_epoch,
                "git_commit": os.environ.get("PHYSTIME_EXPECTED_COMMIT"),
                "git_tree": os.environ.get("PHYSTIME_EXPECTED_TREE"),
                "pre_cross_window_artifact": pre_cross_window_artifact,
                "post_processing": post_processing_audit,
            }
            _atomic_write_json(audit_path, audit_payload)

        result_eval = dict(results=result_dict, evaluation_epoch=evaluation_epoch)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            _atomic_write_json(result_path, result_eval)

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
                metrics_payload = dict(metrics_dict, evaluation_epoch=evaluation_epoch)
                _atomic_write_json(metrics_path, metrics_payload)


def gather_ddp_results(
    world_size,
    result_dict,
    post_cfg,
    *,
    return_audit=False,
    return_pre_cross_window=False,
):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = list(v)

    if return_audit or return_pre_cross_window:
        merged, audit = apply_sliding_window_nms(result_dict, post_cfg, return_audit=True)
        outputs = [merged]
        if return_audit:
            outputs.append(audit)
        if return_pre_cross_window:
            outputs.append(result_dict)
        return tuple(outputs)
    return apply_sliding_window_nms(result_dict, post_cfg)


def apply_sliding_window_nms(result_dict, post_cfg, *, return_audit=False):
    """Apply the same cross-window NMS used by the production evaluation loop."""
    policy = {
        "filter_invalid_proposals": bool(
            _post_cfg_get(post_cfg, "filter_invalid_proposals", True)
        ),
        "proposal_min_duration": float(
            _post_cfg_get(post_cfg, "proposal_min_duration", 0.0)
        ),
        "round_before_cross_window_nms": bool(
            _post_cfg_get(post_cfg, "round_before_cross_window_nms", True)
        ),
        "round_after_cross_window_nms": bool(
            _post_cfg_get(post_cfg, "round_after_cross_window_nms", True)
        ),
        "segment_round_digits": int(
            _post_cfg_get(post_cfg, "segment_round_digits", 2)
        ),
        "score_round_digits": int(
            _post_cfg_get(post_cfg, "score_round_digits", 4)
        ),
    }
    if (
        not math.isfinite(policy["proposal_min_duration"])
        or policy["proposal_min_duration"] < 0
    ):
        raise ValueError("proposal_min_duration must be finite and non-negative")
    if policy["segment_round_digits"] < 0 or policy["score_round_digits"] < 0:
        raise ValueError("rounding digits must be non-negative integers")
    audit = _build_proposal_audit(policy)
    if post_cfg.sliding_window is not True or post_cfg.nms is None:
        audit["nms_applied"] = False
        audit["aggregate"]["input_detections"] = sum(
            len(detections) for detections in result_dict.values()
        )
        audit["aggregate"]["kept_for_nms"] = audit["aggregate"]["input_detections"]
        audit["aggregate"]["post_nms_detections"] = audit["aggregate"]["input_detections"]
        if return_audit:
            return result_dict, audit
        return result_dict

    prepared = {}
    for video_name, detections in result_dict.items():
        raw_valid_detections, raw_validation = _validate_video_detections(
            video_name,
            detections,
            min_duration=policy["proposal_min_duration"],
        )
        segment_values = [item["segment"] for item in raw_valid_detections]
        score_values = [item["score"] for item in raw_valid_detections]
        rounded_segment_values = segment_values
        rounded_score_values = score_values
        changed_segment_values = 0
        changed_scores = 0
        if policy["round_before_cross_window_nms"]:
            rounded_segment_values = [
                [
                    round(value, policy["segment_round_digits"])
                    for value in segment
                ]
                for segment in segment_values
            ]
            rounded_score_values = [
                round(value, policy["score_round_digits"]) for value in score_values
            ]
            changed_segment_values = sum(
                rounded != original
                for rounded_segment, original_segment in zip(
                    rounded_segment_values, segment_values
                )
                for rounded, original in zip(rounded_segment, original_segment)
            )
            changed_scores = sum(
                rounded != original
                for rounded, original in zip(rounded_score_values, score_values)
            )

        effective_detections = [
            {
                "segment": segment,
                "label": item["label"],
                "score": score,
            }
            for item, segment, score in zip(
                raw_valid_detections,
                rounded_segment_values,
                rounded_score_values,
            )
        ]
        effective_valid_detections, effective_validation = _validate_video_detections(
            video_name,
            effective_detections,
            min_duration=policy["proposal_min_duration"],
        )
        video_audit = _combine_validation_audits(
            video_name=video_name,
            raw_validation=raw_validation,
            effective_validation=effective_validation,
            filter_invalid=policy["filter_invalid_proposals"],
            changed_segment_values=changed_segment_values,
            changed_scores=changed_scores,
        )
        prepared[video_name] = effective_valid_detections
        audit["videos"][video_name] = video_audit
        _accumulate_video_audit(audit["aggregate"], video_audit)

    if (
        not policy["filter_invalid_proposals"]
        and audit["aggregate"]["invalid_detections"] > 0
    ):
        raise InvalidProposalError(
            "invalid proposals reached an unfiltered cross-window NMS replay; "
            "the mode was stopped before NMS",
            audit,
        )

    merged = {}
    for video_name, detections in prepared.items():
        if not detections:
            merged[video_name] = []
            continue

        segment_values = [item["segment"] for item in detections]
        score_values = [item["score"] for item in detections]

        segments = torch.tensor(segment_values, dtype=torch.float32)
        scores = torch.tensor(score_values, dtype=torch.float32)
        class_names = []
        class_indices = {}
        labels = []
        for item in detections:
            if item["label"] not in class_indices:
                class_indices[item["label"]] = len(class_names)
                class_names.append(item["label"])
            labels.append(class_indices[item["label"]])
        labels = torch.tensor(labels, dtype=torch.long)
        segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
        post_nms_valid = (
            torch.isfinite(segments).all(dim=1)
            & torch.isfinite(scores)
            & (
                segments[:, 1] - segments[:, 0]
                > policy["proposal_min_duration"]
            )
        )
        post_nms_invalid_count = int((~post_nms_valid).sum().item())
        audit["videos"][video_name][
            "post_nms_invalid_detections"
        ] = post_nms_invalid_count
        audit["aggregate"][
            "post_nms_invalid_detections"
        ] += post_nms_invalid_count
        if post_nms_invalid_count:
            raise RuntimeError(
                f"cross-window NMS produced {post_nms_invalid_count} invalid "
                f"detections for {video_name}"
            )

        merged_detections = []
        for segment, label, score in zip(segments, labels, scores):
            segment_output = [float(value.item()) for value in segment]
            score_output = float(score.item())
            if policy["round_after_cross_window_nms"]:
                segment_output = [
                    round(value, policy["segment_round_digits"])
                    for value in segment_output
                ]
                score_output = round(score_output, policy["score_round_digits"])
            merged_detections.append(
                {
                    "segment": segment_output,
                    "label": class_names[int(label.item())],
                    "score": score_output,
                }
            )
        valid_outputs, output_validation = _validate_video_detections(
            video_name,
            merged_detections,
            min_duration=policy["proposal_min_duration"],
        )
        output_invalid_count = output_validation["invalid_detections"]
        output_filtered_count = (
            output_invalid_count if policy["filter_invalid_proposals"] else 0
        )
        audit["videos"][video_name]["output_validation"] = output_validation
        audit["videos"][video_name][
            "post_nms_invalid_detections"
        ] = output_invalid_count
        audit["videos"][video_name][
            "post_nms_filtered_detections"
        ] = output_filtered_count
        audit["aggregate"][
            "post_nms_invalid_detections"
        ] += output_invalid_count
        audit["aggregate"][
            "post_nms_filtered_detections"
        ] += output_filtered_count
        if output_invalid_count and not policy["filter_invalid_proposals"]:
            raise InvalidProposalError(
                "rounding produced invalid unfiltered detections after "
                f"cross-window NMS for {video_name}",
                audit,
            )
        merged[video_name] = valid_outputs
        audit["videos"][video_name]["post_nms_detections"] = len(valid_outputs)
        audit["aggregate"]["post_nms_detections"] += len(valid_outputs)

    audit["nms_applied"] = True
    if return_audit:
        return merged, audit
    return merged


def _post_cfg_get(post_cfg, key, default):
    if hasattr(post_cfg, "get"):
        return post_cfg.get(key, default)
    return getattr(post_cfg, key, default)


def _build_proposal_audit(policy):
    invalid_keys = (
        "malformed_detection",
        "malformed_segment",
        "non_finite_segment",
        "non_finite_score",
        "non_positive_duration",
        "invalid_label",
    )
    return {
        "schema_version": "opentad_cross_window_nms_audit_v1",
        "nms_applied": None,
        "policy": dict(policy),
        "aggregate": {
            "videos": 0,
            "videos_with_invalid_detections": 0,
            "input_detections": 0,
            "valid_detections": 0,
            "invalid_detections": 0,
            "filtered_detections": 0,
            "raw_invalid_detections": 0,
            "raw_filtered_detections": 0,
            "effective_input_detections": 0,
            "effective_invalid_detections": 0,
            "effective_filtered_detections": 0,
            "rounding_induced_invalid_detections": 0,
            "kept_for_nms": 0,
            "post_nms_detections": 0,
            "post_nms_invalid_detections": 0,
            "post_nms_filtered_detections": 0,
            "pre_nms_rounding_changed_segment_values": 0,
            "pre_nms_rounding_changed_scores": 0,
            "invalid_reason_counts": {key: 0 for key in invalid_keys},
            "raw_invalid_reason_counts": {key: 0 for key in invalid_keys},
            "effective_invalid_reason_counts": {key: 0 for key in invalid_keys},
        },
        "videos": {},
    }


def _validate_video_detections(
    video_name,
    detections,
    *,
    min_duration,
):
    valid_detections = []
    invalid_reason_counts = {
        "malformed_detection": 0,
        "malformed_segment": 0,
        "non_finite_segment": 0,
        "non_finite_score": 0,
        "non_positive_duration": 0,
        "invalid_label": 0,
    }
    invalid_samples = []
    for index, item in enumerate(detections):
        reasons = []
        parsed = None
        if not isinstance(item, dict):
            reasons.append("malformed_detection")
        else:
            segment = item.get("segment")
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                reasons.append("malformed_segment")
            else:
                try:
                    start = float(segment[0])
                    end = float(segment[1])
                except (TypeError, ValueError, OverflowError):
                    reasons.append("malformed_segment")
                else:
                    if not math.isfinite(start) or not math.isfinite(end):
                        reasons.append("non_finite_segment")
                    elif end - start <= min_duration:
                        reasons.append("non_positive_duration")

            try:
                score = float(item.get("score"))
            except (TypeError, ValueError, OverflowError):
                reasons.append("non_finite_score")
            else:
                if not math.isfinite(score):
                    reasons.append("non_finite_score")

            label = item.get("label")
            try:
                label_is_valid = label is not None and hash(label) is not None
            except TypeError:
                label_is_valid = False
            if not label_is_valid:
                reasons.append("invalid_label")

            if not reasons:
                parsed = {
                    "segment": [start, end],
                    "label": label,
                    "score": score,
                }

        unique_reasons = sorted(set(reasons))
        if unique_reasons:
            for reason in unique_reasons:
                invalid_reason_counts[reason] += 1
            if len(invalid_samples) < 20:
                invalid_samples.append(
                    {
                        "index": index,
                        "reasons": unique_reasons,
                        "detection_repr": repr(item)[:512],
                    }
                )
        else:
            valid_detections.append(parsed)

    invalid_count = len(detections) - len(valid_detections)
    video_audit = {
        "video_name": video_name,
        "input_detections": len(detections),
        "valid_detections": len(valid_detections),
        "invalid_detections": invalid_count,
        "invalid_reason_counts": invalid_reason_counts,
        "invalid_samples": invalid_samples,
    }
    return valid_detections, video_audit


def _combine_validation_audits(
    *,
    video_name,
    raw_validation,
    effective_validation,
    filter_invalid,
    changed_segment_values,
    changed_scores,
):
    raw_invalid = raw_validation["invalid_detections"]
    effective_invalid = effective_validation["invalid_detections"]
    combined_reasons = {
        key: raw_validation["invalid_reason_counts"][key]
        + effective_validation["invalid_reason_counts"][key]
        for key in raw_validation["invalid_reason_counts"]
    }
    return {
        "video_name": video_name,
        "input_detections": raw_validation["input_detections"],
        "valid_detections": effective_validation["valid_detections"],
        "invalid_detections": raw_invalid + effective_invalid,
        "filtered_detections": (
            raw_invalid + effective_invalid if filter_invalid else 0
        ),
        "raw_invalid_detections": raw_invalid,
        "raw_filtered_detections": raw_invalid if filter_invalid else 0,
        "effective_input_detections": effective_validation["input_detections"],
        "effective_invalid_detections": effective_invalid,
        "effective_filtered_detections": (
            effective_invalid if filter_invalid else 0
        ),
        "rounding_induced_invalid_detections": effective_invalid,
        "kept_for_nms": effective_validation["valid_detections"],
        "post_nms_detections": 0,
        "post_nms_invalid_detections": 0,
        "post_nms_filtered_detections": 0,
        "pre_nms_rounding_changed_segment_values": changed_segment_values,
        "pre_nms_rounding_changed_scores": changed_scores,
        "invalid_reason_counts": combined_reasons,
        "raw_validation": raw_validation,
        "effective_validation": effective_validation,
        "output_validation": None,
    }


def _accumulate_video_audit(aggregate, video_audit):
    aggregate["videos"] += 1
    aggregate["input_detections"] += video_audit["input_detections"]
    aggregate["valid_detections"] += video_audit["valid_detections"]
    aggregate["invalid_detections"] += video_audit["invalid_detections"]
    aggregate["filtered_detections"] += video_audit["filtered_detections"]
    aggregate["raw_invalid_detections"] += video_audit[
        "raw_invalid_detections"
    ]
    aggregate["raw_filtered_detections"] += video_audit[
        "raw_filtered_detections"
    ]
    aggregate["effective_input_detections"] += video_audit[
        "effective_input_detections"
    ]
    aggregate["effective_invalid_detections"] += video_audit[
        "effective_invalid_detections"
    ]
    aggregate["effective_filtered_detections"] += video_audit[
        "effective_filtered_detections"
    ]
    aggregate["rounding_induced_invalid_detections"] += video_audit[
        "rounding_induced_invalid_detections"
    ]
    aggregate["kept_for_nms"] += video_audit["kept_for_nms"]
    aggregate["pre_nms_rounding_changed_segment_values"] += video_audit[
        "pre_nms_rounding_changed_segment_values"
    ]
    aggregate["pre_nms_rounding_changed_scores"] += video_audit[
        "pre_nms_rounding_changed_scores"
    ]
    if video_audit["invalid_detections"] > 0:
        aggregate["videos_with_invalid_detections"] += 1
    for key, count in video_audit["invalid_reason_counts"].items():
        aggregate["invalid_reason_counts"][key] += count
    for key, count in video_audit["raw_validation"][
        "invalid_reason_counts"
    ].items():
        aggregate["raw_invalid_reason_counts"][key] += count
    for key, count in video_audit["effective_validation"][
        "invalid_reason_counts"
    ].items():
        aggregate["effective_invalid_reason_counts"][key] += count


def _atomic_write_json(path, payload):
    path = os.path.abspath(path)
    create_folder(os.path.dirname(path))
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as out:
        json.dump(
            payload,
            out,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item(),
        )
        out.write("\n")
    os.replace(temporary_path, path)


def _atomic_write_json_gzip(path, payload):
    path = os.path.abspath(path)
    create_folder(os.path.dirname(path))
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as out:
        json.dump(
            payload,
            out,
            separators=(",", ":"),
            sort_keys=True,
            default=lambda value: value.item(),
        )
        out.write("\n")
    os.replace(temporary_path, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
