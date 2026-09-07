from collections.abc import Mapping

import torch.nn as nn
from .single_stage import SingleStageDetector
from ..builder import DETECTORS
from ..bricks import Scale, AffineDropPath
from ..utils.truetime_geometry import SELECTED_AXIS


@DETECTORS.register_module()
class TemporalMaxer(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        frame_selector=None,
        selector_train_only=False,
        selector_train_only_skip_detector=False,
    ):
        if selector_train_only or selector_train_only_skip_detector:
            raise ValueError(
                "TemporalMaxer R5 requires joint live selector/detector training"
            )
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            frame_selector=frame_selector,
        )

    def forward_train(
        self,
        inputs,
        masks,
        metas,
        gt_segments,
        gt_labels,
        **kwargs,
    ):
        losses = {}
        selector_loss_keys = set()
        selector_kwargs = dict(kwargs)
        detector_kwargs = dict(kwargs)
        detector_kwargs.pop("gt_boundary_validity", None)
        if self.with_frame_selector:
            selector_outputs = self.frame_selector.forward_train(
                inputs=inputs,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **selector_kwargs,
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            gt_segments = selector_outputs["gt_segments"]
            gt_labels = selector_outputs["gt_labels"]
            self._merge_selector_losses(losses, selector_outputs.get("losses", {}))
            selector_loss_keys = set(losses)
            self._last_selected_axis_training_summary = (
                self._validate_selected_axis_training_metadata(
                    metas, masks, gt_segments
                )
            )

        self._validate_temporal_axis(inputs, masks)
        x = self._forward_backbone(inputs, masks)
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
                **detector_kwargs,
            )
            self._merge_detector_losses(
                losses,
                rpn_losses,
                source_name="rpn_head",
                protected_keys=selector_loss_keys,
            )
        losses["cost"] = sum(losses.values())
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

        self._validate_temporal_axis(inputs, masks)
        x = self._forward_backbone(inputs, masks)
        if self.with_projection:
            x, masks = self.projection(x, masks)
        if self.with_neck:
            x, masks, metas = self._call_neck_forward(x, masks, metas=metas)
        if self.with_rpn_head:
            proposals, scores = self._call_rpn_head_forward_test(
                x, masks, metas=metas
            )
        else:
            proposals = scores = None
        self._last_forward_test_metas = metas
        return proposals, scores

    @staticmethod
    def _validate_temporal_axis(inputs, masks):
        if masks.ndim != 2:
            raise ValueError("TemporalMaxer masks must be [batch, time]")
        if inputs.ndim == 6:
            temporal = int(inputs.shape[3])
        elif inputs.ndim == 3:
            temporal = int(inputs.shape[2])
        else:
            raise ValueError(
                "TemporalMaxer expects selected RGB [B,N,C,T,H,W] or features [B,C,T]"
            )
        if temporal != int(masks.shape[1]):
            raise ValueError(
                "TemporalMaxer selected input and mask temporal axes disagree"
            )

    def _forward_backbone(self, inputs, masks):
        if not self.with_backbone:
            return inputs
        if inputs.ndim == 6:
            # AdaTAD's raw-RGB VideoMAE wrapper accepts only the clip tensor.
            return self.backbone(inputs)
        # Preserve TemporalMaxer's existing feature-backbone calling contract.
        return self.backbone(inputs, masks)

    @staticmethod
    def _validate_selected_axis_training_metadata(metas, masks, gt_segments):
        if not isinstance(metas, (list, tuple)) or len(metas) != masks.shape[0]:
            raise ValueError("TemporalMaxer live selector must return one meta per sample")
        if gt_segments is not None and len(gt_segments) != len(metas):
            raise ValueError("TemporalMaxer selected-axis GT batch is misaligned")
        selected_counts = []
        for index, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"TemporalMaxer metas[{index}] must be a mapping")
            positions = meta.get("selected_axis_to_true_time_dense_index")
            if not isinstance(positions, (list, tuple)) or not positions:
                raise ValueError("TemporalMaxer live selector omitted its inverse map")
            selected_count = meta.get("duca_online_selected_count", len(positions))
            if int(selected_count) != len(positions) or len(positions) > masks.shape[1]:
                raise ValueError("TemporalMaxer selected mask and inverse map disagree")
            if gt_segments is not None and (
                meta.get("gt_remapped_to_selected_axis") is not True
                or meta.get("gt_coordinate_space") != SELECTED_AXIS
            ):
                raise ValueError("TemporalMaxer detector GT was not remapped to selected axis")
            selected_counts.append(len(positions))
        return {
            "selected_counts": selected_counts,
            "gt_coordinate_space": SELECTED_AXIS,
            "inverse_map_present": True,
        }

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        blacklist_weight_modules = (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Embedding,
        )

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)
                elif pn.endswith(("query_embed", "slot_queries")):
                    no_decay.add(fpn)

        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if not pn.startswith("backbone") and p.requires_grad
        }
        # DUCA selectors contain trainable tensors beyond the original
        # TemporalMaxer Conv1d/LN inventory. Classify every remaining tensor
        # by shape so live selector training cannot silently drop a parameter.
        for name, parameter in param_dict.items():
            if name in decay or name in no_decay:
                continue
            if parameter.ndim >= 2:
                decay.add(name)
            else:
                no_decay.add(name)

        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        selector = getattr(self, "frame_selector", None)
        transition_only = getattr(selector, "selector_variant", None) == "transition_only"
        protected_e2e = (
            getattr(selector, "selector_variant", None)
            == "protected_e2e_physical"
        )

        def parameter_lr(name):
            if selector is None:
                return float(cfg["lr"])
            if transition_only and name.startswith(
                (
                    "frame_selector.adapter.transition_scorer.",
                    "frame_selector.adapter.density_mixture_head.",
                )
            ):
                return float(getattr(selector, "transition_scorer_lr", cfg["lr"]))
            if protected_e2e and name.startswith(
                (
                    "frame_selector.transition_scorer.selector_adapter.",
                    "frame_selector.transition_scorer.selector_score_head.",
                )
            ):
                return float(getattr(selector, "selector_lr", cfg["lr"]))
            if name.startswith("frame_selector.raw_actionness_source.probe_module."):
                marker = ".official_temporal."
                tail = name.split(marker, 1)[1] if marker in name else ""
                parts = tail.split(".")
                action_head = tail.startswith("encoder.conv_out.") or (
                    len(parts) >= 4
                    and parts[0] == "decoders"
                    and parts[2] == "conv_out"
                )
                key = "action_head_lr" if action_head else "coarse_trunk_lr"
                return float(getattr(selector, key, cfg["lr"]))
            return float(cfg["lr"])

        if protected_e2e:
            allowed = (
                "frame_selector.transition_scorer.selector_adapter.",
                "frame_selector.transition_scorer.selector_score_head.",
                "frame_selector.raw_actionness_source.probe_module.",
            )
            unexpected = sorted(
                name
                for name in param_dict
                if name.startswith("frame_selector.")
                and not name.startswith(allowed)
            )
            if unexpected:
                raise AssertionError(
                    "protected DUCA exposes trainable selector parameters outside "
                    f"the optimizer contract: {unexpected}"
                )

        grouped = {}
        for names, weight_decay in (
            (decay, float(cfg["weight_decay"])),
            (no_decay, 0.0),
        ):
            for name in sorted(names):
                key = (parameter_lr(name), weight_decay)
                grouped.setdefault(key, []).append(param_dict[name])
        optim_groups = [
            {"params": params, "weight_decay": weight_decay, "lr": lr}
            for (lr, weight_decay), params in sorted(grouped.items())
            if params
        ]
        return optim_groups
