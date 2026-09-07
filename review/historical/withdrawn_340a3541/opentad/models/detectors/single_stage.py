import inspect
from collections.abc import Mapping
import torch
from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck, build_selector
from .base import BaseDetector
from ..utils.post_processing import batched_nms, convert_to_seconds
from ..utils.truetime_geometry import (
    SELECTED_AXIS,
    TRUE_TIME_AXIS,
    remap_selected_axis_segments_to_true_time,
)


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None, frame_selector=None):
        super(SingleStageDetector, self).__init__()

        if frame_selector is not None:
            self.frame_selector = build_selector(frame_selector)

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_frame_selector(self):
        """bool: whether the detector has a pre-backbone frame selector"""
        return hasattr(self, "frame_selector") and self.frame_selector is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def after_optimizer_step(self):
        if self.with_frame_selector:
            hook = getattr(self.frame_selector, "after_optimizer_step", None)
            if callable(hook):
                return hook()
        return None

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        selector_loss_keys = set()
        if self.with_frame_selector:
            selector_outputs = self.frame_selector.forward_train(
                inputs=inputs,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            gt_segments = selector_outputs["gt_segments"]
            gt_labels = selector_outputs["gt_labels"]
            self._merge_selector_losses(losses, selector_outputs.get("losses", {}))
            selector_loss_keys = set(losses)

        if self.with_backbone:
            backbone_kwargs = selector_outputs.get("extra_backbone_kwargs", {}) if self.with_frame_selector else {}
            x = self.backbone(inputs, masks, **backbone_kwargs)
        else:
            x = inputs

        if self.with_frame_selector and hasattr(self.frame_selector, "recover_features"):
            backbone_support_meta = getattr(self.backbone, "latest_support_metadata", None) if self.with_backbone else None
            x, masks = self.frame_selector.recover_features(
                x, masks, selector_outputs, backbone_support_metadata=backbone_support_meta
            )
            metas = selector_outputs.get("metas", metas)
            gt_segments = selector_outputs.get("gt_segments", gt_segments)
            gt_labels = selector_outputs.get("gt_labels", gt_labels)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks, metas = self._call_neck_forward(x, masks, metas=metas)

        if self.with_rpn_head:
            rpn_losses = self._call_rpn_head_forward_train(
                x,
                masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            self._merge_detector_losses(
                losses,
                rpn_losses,
                source_name="rpn_head",
                protected_keys=selector_loss_keys,
            )

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_frame_selector:
            selector_outputs = self.frame_selector.forward_test(
                inputs=inputs,
                masks=masks,
                metas=metas,
                **kwargs,
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            self._require_selector_remap_metadata(metas)

        if self.with_backbone:
            backbone_kwargs = selector_outputs.get("extra_backbone_kwargs", {}) if self.with_frame_selector else {}
            x = self.backbone(inputs, masks, **backbone_kwargs)
        else:
            x = inputs

        if self.with_frame_selector and hasattr(self.frame_selector, "recover_features"):
            backbone_support_meta = getattr(self.backbone, "latest_support_metadata", None) if self.with_backbone else None
            x, masks = self.frame_selector.recover_features(
                x, masks, selector_outputs, backbone_support_metadata=backbone_support_meta
            )
            metas = selector_outputs.get("metas", metas)
            self._require_selector_remap_metadata(metas)


        if self.with_projection:
            x, masks = self.projection(x, masks)


        if self.with_neck:
            x, masks, metas = self._call_neck_forward(x, masks, metas=metas)

        if self.with_rpn_head:
            rpn_proposals, rpn_scores = self._call_rpn_head_forward_test(x, masks, metas=metas)
        else:
            rpn_proposals = rpn_scores = None

        self._last_forward_test_metas = metas
        predictions = rpn_proposals, rpn_scores
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        rpn_proposals, rpn_scores = predictions
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K,num_classes] after sigmoid

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            meta = metas[i]
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0], dtype=torch.long).contiguous()
            else:
                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            segments, meta = self._remap_selector_segments_for_post_processing(segments, meta)

            # if not sliding window, do nms
            sliding_window = post_cfg.get("sliding_window", False) if hasattr(post_cfg, "get") else getattr(post_cfg, "sliding_window", False)
            nms_cfg = post_cfg.get("nms", None) if hasattr(post_cfg, "get") else getattr(post_cfg, "nms", None)
            if not sliding_window and nms_cfg is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **nms_cfg)


            video_id = meta["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, meta)

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def _call_neck_forward(self, feat_list, mask_list, metas):
        if self._callable_accepts_metas(self.neck.forward):
            out = self.neck(feat_list, mask_list, metas=metas)
        else:
            out = self.neck(feat_list, mask_list)
        if not isinstance(out, (tuple, list)):
            raise TypeError("neck forward must return (features, masks) or (features, masks, metas)")
        if len(out) == 2:
            feat_out, mask_out = out
            return feat_out, mask_out, metas
        if len(out) == 3:
            feat_out, mask_out, meta_out = out
            return feat_out, mask_out, meta_out
        raise ValueError("neck forward must return (features, masks) or (features, masks, metas)")

    def _call_rpn_head_forward_train(self, feat_list, mask_list, metas, gt_segments, gt_labels, **kwargs):
        call_kwargs = dict(kwargs)
        if self._callable_accepts_metas(self.rpn_head.forward_train):
            call_kwargs["metas"] = metas
        return self.rpn_head.forward_train(
            feat_list,
            mask_list,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **call_kwargs,
        )

    def _call_rpn_head_forward_test(self, feat_list, mask_list, metas, **kwargs):
        call_kwargs = dict(kwargs)
        if self._callable_accepts_metas(self.rpn_head.forward_test):
            call_kwargs["metas"] = metas
        return self.rpn_head.forward_test(feat_list, mask_list, **call_kwargs)

    @staticmethod
    def _validate_selector_losses(selector_losses):
        for key, value in selector_losses.items():
            if key in {"cost", "total_loss", "detector_utility_distribution_loss"}:
                raise ValueError(f"frame_selector aggregate or alias loss is forbidden: {key}")
            if not str(key).endswith("_loss"):
                raise ValueError(f"frame_selector loss key must name a leaf loss: {key}")
            if not torch.is_tensor(value):
                raise ValueError(f"frame_selector loss value for {key} must be a tensor")
            if value.ndim != 0:
                raise ValueError(f"frame_selector loss value for {key} must be scalar")
            if torch.is_complex(value) or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"frame_selector loss value for {key} must be finite and real-valued")

    @staticmethod
    def _merge_selector_losses(losses, selector_losses):
        SingleStageDetector._validate_selector_losses(selector_losses)
        for key, value in selector_losses.items():
            prefixed_key = key if str(key).startswith("selector_") else f"selector_{key}"
            if prefixed_key in losses:
                raise ValueError(f"frame_selector loss key collision: {prefixed_key}")
            losses[prefixed_key] = value

    @staticmethod
    def _merge_detector_losses(losses, detector_losses, *, source_name, protected_keys):
        for key, value in detector_losses.items():
            if key in protected_keys:
                raise ValueError(f"{source_name} loss key collision with frame_selector: {key}")
            losses[key] = value

    @staticmethod
    def _require_selector_remap_metadata(metas):
        if metas is None:
            return
        holders = [metas] if isinstance(metas, Mapping) else metas
        if not isinstance(holders, (list, tuple)):
            raise ValueError("metas must be a mapping/list/tuple or None")
        for idx, meta in enumerate(holders):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            if meta.get("detector_prediction_inverse_map_required") is not True:
                continue
            if "selected_axis_to_true_time_dense_index" not in meta:
                raise RuntimeError(
                    "frame_selector forward_test requires prediction inverse-map metadata, "
                    "but selected_axis_to_true_time_dense_index is missing"
                )
            if "detector_output_coordinate_space" not in meta:
                raise RuntimeError(
                    "frame_selector forward_test requires detector_output_coordinate_space metadata for remap"
                )

    @staticmethod
    def _remap_selector_segments_for_post_processing(segments, meta):
        if not isinstance(meta, Mapping):
            raise ValueError("meta must be a mapping")
        if meta.get("detector_prediction_inverse_map_required") is not True:
            return segments, meta
        coordinate_space = meta.get("detector_output_coordinate_space")
        if coordinate_space == TRUE_TIME_AXIS:
            return segments, meta
        if coordinate_space != SELECTED_AXIS:
            raise RuntimeError(
                "frame_selector post-processing expected detector_output_coordinate_space "
                f"{SELECTED_AXIS!r} or {TRUE_TIME_AXIS!r}, got {coordinate_space!r}"
            )
        remapped = remap_selected_axis_segments_to_true_time(segments, meta)
        remapped_meta = dict(meta)
        remapped_meta["detector_output_coordinate_space"] = TRUE_TIME_AXIS
        remapped_meta["irregular_native_axis"] = True
        return remapped, remapped_meta

    @staticmethod
    def _callable_accepts_metas(fn):
        signature = inspect.signature(fn)
        for param in signature.parameters.values():
            if param.name == "metas":
                return True
        return False
