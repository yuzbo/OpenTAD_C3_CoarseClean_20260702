import torch

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class TriDet(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        frame_selector=None,
    ):
        super(TriDet, self).__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            frame_selector=frame_selector,
        )

        self.max_seq_len = projection.max_seq_len
        assert len(projection.sgp_win_size) == len(rpn_head.prior_generator.strides)

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, projection.sgp_win_size):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len <= self.max_seq_len:
            max_len = self.max_seq_len
        else:
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        selector_loss_keys = set()
        gt_boundary_validity = kwargs.pop("gt_boundary_validity", None)
        original_gt_segments = gt_segments
        original_gt_labels = gt_labels
        if self.with_frame_selector:
            selector_kwargs = {
                key: kwargs.pop(key)
                for key in list(kwargs)
                if str(key).startswith("rime_")
            }
            selector_outputs = self.frame_selector.forward_train(
                inputs=inputs,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                gt_boundary_validity=gt_boundary_validity,
                **selector_kwargs,
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            gt_segments = selector_outputs["gt_segments"]
            gt_labels = selector_outputs["gt_labels"]
            self._validate_rime_selector_contract(
                inputs,
                masks,
                metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                original_gt_segments=original_gt_segments,
                original_gt_labels=original_gt_labels,
            )
            self._merge_selector_losses(
                losses,
                selector_outputs.get("losses", {}),
            )
            selector_loss_keys = set(losses)
        if self.with_backbone:
            if (
                getattr(
                    getattr(self, "frame_selector", None),
                    "selector_variant",
                    None,
                )
                == "duca_rime_physical"
            ):
                x = self.backbone(inputs, masks=masks)
            else:
                x = self.backbone(inputs)
        else:
            x = inputs

        # pad the features and unsqueeze the mask
        if not self.training:
            x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        loc_losses = self.rpn_head.forward_train(
            x,
            masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        self._merge_detector_losses(
            losses,
            loc_losses,
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
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            self._validate_rime_selector_contract(inputs, masks, metas)
            self._require_selector_remap_metadata(metas)
        if self.with_backbone:
            if (
                getattr(
                    getattr(self, "frame_selector", None),
                    "selector_variant",
                    None,
                )
                == "duca_rime_physical"
            ):
                x = self.backbone(inputs, masks=masks)
            else:
                x = self.backbone(inputs)
        else:
            x = inputs

        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        points, rpn_reg, rpn_scores = self.rpn_head.forward_test(
            x,
            masks,
            metas=metas,
            **kwargs,
        )
        self._last_forward_test_metas = metas
        predictions = points, rpn_reg, rpn_scores
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        points, rpn_reg, rpn_scores = predictions  # [N, 4], [B, num_classes, N, 2], [B, N, num_classes]

        pre_nms_thresh = 0.001
        pre_nms_topk = 2000
        num_classes = rpn_scores.shape[-1]
        points = points.cpu()

        results = {}
        for i in range(len(metas)):  # processing each video
            scores = rpn_scores[i].detach().cpu()  # [N]
            reg = rpn_reg[i].detach().cpu()  # [num_classes, N, 2]
            sample_points = points[i] if points.dim() == 3 else points

            if num_classes == 1:
                segments = self.rpn_head.get_proposals(
                    sample_points,
                    reg.squeeze(0),
                ).detach().cpu()  # [N, 2]
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
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

                segments = self.rpn_head.get_proposals(
                    sample_points[pt_idxs],
                    reg[cls_idxs, pt_idxs],
                ).detach().cpu()  # [N, 2]
                scores = pred_prob
                labels = cls_idxs

            # Non-uniform selected-axis predictions must be mapped back to the
            # physical detector axis before NMS.  Protected RIME/TriDet already
            # emits dense-physical proposals, so this is a checked identity.
            segments, meta = self._remap_selector_segments_for_post_processing(
                segments,
                metas[i],
            )

            # if not sliding window, do nms
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

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

    def get_optim_groups(self, cfg):
        # Reuse the audited DUCA/RIME-aware grouping contract.  It covers the
        # TriDet tail plus Conv2d/3d selector weights, frozen parameters, and
        # selector-specific learning rates.
        from .actionformer import ActionFormer

        return ActionFormer.get_optim_groups(self, cfg)

    def _validate_rime_selector_contract(
        self,
        inputs,
        masks,
        metas,
        *,
        gt_segments=None,
        gt_labels=None,
        original_gt_segments=None,
        original_gt_labels=None,
    ):
        selector = getattr(self, "frame_selector", None)
        if getattr(selector, "selector_variant", None) != "duca_rime_physical":
            return
        if masks.ndim != 2 or not bool(masks.to(dtype=torch.bool).all().item()):
            raise RuntimeError("TriDet RIME requires an exact-K, padding-free detector mask")
        temporal_dim = 3 if inputs.ndim == 6 else 2
        if inputs.ndim not in {3, 5, 6} or int(inputs.shape[temporal_dim]) != int(
            masks.shape[1]
        ):
            raise RuntimeError("TriDet RIME input and mask temporal axes disagree")
        if not isinstance(metas, (list, tuple)) or len(metas) != int(inputs.shape[0]):
            raise RuntimeError("TriDet RIME requires one metadata record per sample")
        for meta, mask in zip(metas, masks):
            selected_count = int(mask.to(dtype=torch.bool).sum().item())
            required = {
                "duca_contract": "duca_rime_physical_dynamic_k_v1",
                "physical_grid_contract": "duca_rime_physical_dynamic_k_v1",
                "irregular_native_axis": True,
                "detector_output_coordinate_space": "dense_physical",
                "proposal_axis": "dense_physical",
                "detector_prediction_inverse_map_required": False,
                "duca_dynamic_compute_realized": True,
                "duca_backbone_tail_padding_mode": "none_exact_k_bucket",
                "duca_execution_quantum": 16,
            }
            if any(meta.get(key) != value for key, value in required.items()):
                raise RuntimeError("TriDet RIME physical metadata contract is inconsistent")
            requested = int(meta.get("duca_requested_k", -1))
            effective = int(meta.get("duca_effective_k", -1))
            unique = int(meta.get("duca_unique_k", -1))
            backbone = int(meta.get("duca_backbone_input_k", -1))
            padded = int(meta.get("duca_padded_k", -1))
            if not (
                requested >= effective
                and effective == unique == backbone == padded == selected_count
                and effective % 16 == 0
            ):
                raise RuntimeError("TriDet RIME heavy-frame cost ledger is inconsistent")
        if original_gt_segments is not None and gt_segments is not original_gt_segments:
            raise RuntimeError("TriDet RIME must preserve dense-axis GT segment objects")
        if original_gt_labels is not None and gt_labels is not original_gt_labels:
            raise RuntimeError("TriDet RIME must preserve dense-axis GT label objects")
