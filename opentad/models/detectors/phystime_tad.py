import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import Scale
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class PhysTimeTAD(SingleStageDetector):
    """Offline TAD detector defined on explicit irregular supports in seconds."""

    def __init__(
        self,
        projection,
        rpn_head,
        backbone=None,
        discretization_loss_weight=0.1,
    ):
        super().__init__(backbone=backbone, projection=projection, rpn_head=rpn_head)
        self.discretization_loss_weight = float(discretization_loss_weight)

    @staticmethod
    def _has_forbidden_metadata(meta):
        forbidden_fragments = (
            "teacher",
            "oracle",
            "ledger",
            "prediction_cache",
            "raw_prediction",
            "actionness",
            "budget",
            "selector_utility",
        )
        for key, value in meta.items():
            if value is None or value is False or (isinstance(value, str) and value == ""):
                continue
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in forbidden_fragments):
                return key
        return None

    def _validate_metas(self, metas, *, training):
        if not isinstance(metas, (list, tuple)) or not metas:
            raise ValueError("PhysTimeTAD requires one metadata dictionary per sample")
        for meta in metas:
            if not isinstance(meta, dict):
                raise ValueError("PhysTimeTAD metadata entries must be dictionaries")
            forbidden_key = self._has_forbidden_metadata(meta)
            if forbidden_key is not None:
                raise ValueError(f"PhysTimeTAD received forbidden metadata: {forbidden_key}")
            if meta.get("irregular_native_axis") is not True:
                raise ValueError("PhysTimeTAD requires the native physical-time axis")
            if any(
                bool(meta.get(key, False))
                for key in (
                    "remap_gt_to_selected_axis",
                    "gt_remapped_to_selected_axis",
                    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis",
                )
            ):
                raise ValueError("PhysTimeTAD forbids selected-axis GT remapping")
            if meta.get("prediction_time_unit") != "seconds":
                raise ValueError("PhysTimeTAD predictions must be declared in absolute seconds")
            if training and meta.get("gt_time_unit") != "seconds":
                raise ValueError("PhysTimeTAD training requires ground truth in absolute seconds")

    def _extract_observations(self, inputs, masks):
        if self.with_backbone:
            try:
                features = self.backbone(inputs, masks)
            except TypeError:
                features = self.backbone(inputs)
        else:
            features = inputs
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("PhysTimeTAD backbone must return a [B, C, K] tensor")
        return features

    def _forward_train_view(self, inputs, masks, metas, gt_segments, gt_labels):
        self._validate_metas(metas, training=True)
        observations = self._extract_observations(inputs, masks)
        feat_list, mask_list, geometry_list = self.projection(observations, masks, metas)
        losses, raw = self.rpn_head.forward_train(
            feat_list,
            mask_list,
            geometry_list,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            return_outputs=True,
        )
        return losses, raw

    @staticmethod
    def _common_coverage_consistency(first, second):
        if first["mask"].shape != second["mask"].shape:
            raise ValueError("paired PhysTime views must produce the same physical query grid")
        first_points = torch.cat(first["points"], dim=1)
        second_points = torch.cat(second["points"], dim=1)
        if not torch.allclose(first_points[..., 0], second_points[..., 0], atol=1.0e-6, rtol=0):
            raise ValueError("paired PhysTime views must use identical physical query centers")
        common_mask = first["mask"] & second["mask"]
        if not common_mask.any():
            return first["proposals_sec"].sum() * 0.0

        first_cls = torch.cat([item.transpose(1, 2) for item in first["cls_logits"]], dim=1).sigmoid()
        second_cls = torch.cat([item.transpose(1, 2) for item in second["cls_logits"]], dim=1).sigmoid()
        first_endpoint = torch.cat(first["endpoint_probabilities"], dim=1)
        second_endpoint = torch.cat(second["endpoint_probabilities"], dim=1)
        widths = first["cell_widths_sec"].clamp_min(1.0e-6)
        common_coverage = torch.minimum(first["coverage_sec"], second["coverage_sec"])
        coverage_weight = (common_coverage / widths).clamp(min=0.0, max=1.0)
        coverage_weight = coverage_weight * common_mask.to(coverage_weight.dtype)
        weight_sum = coverage_weight.sum().clamp_min(1.0e-6)

        cls_error = (first_cls - second_cls).square().mean(dim=-1)
        endpoint_error = (first_endpoint - second_endpoint).square().mean(dim=-1)
        cls_consistency = (cls_error * coverage_weight).sum() / weight_sum
        endpoint_consistency = (endpoint_error * coverage_weight).sum() / weight_sum
        first_segments = first["proposals_sec"] / widths[..., None]
        second_segments = second["proposals_sec"] / widths[..., None]
        segment_error = F.smooth_l1_loss(
            first_segments[common_mask],
            second_segments[common_mask],
            reduction="none",
        ).mean(dim=-1)
        segment_consistency = (
            segment_error * coverage_weight[common_mask]
        ).sum() / weight_sum
        return cls_consistency + endpoint_consistency + segment_consistency

    def forward_train(
        self,
        inputs,
        masks,
        metas,
        gt_segments,
        gt_labels,
        paired_inputs=None,
        paired_masks=None,
        paired_metas=None,
        **kwargs,
    ):
        losses, raw = self._forward_train_view(inputs, masks, metas, gt_segments, gt_labels)
        paired_values = (paired_inputs, paired_masks, paired_metas)
        if any(value is not None for value in paired_values):
            if not all(value is not None for value in paired_values):
                raise ValueError("paired PhysTime training requires paired_inputs, paired_masks, and paired_metas")
            paired_losses, paired_raw = self._forward_train_view(
                paired_inputs,
                paired_masks,
                paired_metas,
                gt_segments,
                gt_labels,
            )
            if losses.keys() != paired_losses.keys():
                raise RuntimeError("paired PhysTime views produced different detection loss keys")
            losses = {key: 0.5 * (value + paired_losses[key]) for key, value in losses.items()}
            losses["discretization_loss"] = (
                self._common_coverage_consistency(raw, paired_raw) * self.discretization_loss_weight
            )
        losses["cost"] = sum(losses.values())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        self._validate_metas(metas, training=False)
        observations = self._extract_observations(inputs, masks)
        feat_list, mask_list, geometry_list = self.projection(observations, masks, metas)
        predictions = self.rpn_head.forward_test(feat_list, mask_list, geometry_list)
        self._last_forward_test_metas = metas
        return predictions

    def get_optim_groups(self, cfg):
        decay = set()
        no_decay = set()
        decay_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        no_decay_modules = (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                if full_name.startswith("backbone."):
                    continue
                if parameter_name.endswith("bias"):
                    no_decay.add(full_name)
                elif parameter_name.endswith("weight") and isinstance(module, decay_modules):
                    decay.add(full_name)
                elif parameter_name.endswith("weight") and isinstance(module, no_decay_modules):
                    no_decay.add(full_name)
                elif parameter_name.endswith("scale") and isinstance(module, Scale):
                    no_decay.add(full_name)

        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("backbone.")
        }
        for name, parameter in parameters.items():
            if name in decay or name in no_decay:
                continue
            if parameter.ndim >= 2:
                decay.add(name)
            else:
                no_decay.add(name)
        if decay & no_decay or (decay | no_decay) != set(parameters):
            raise RuntimeError("PhysTimeTAD optimizer parameter classification is incomplete")
        return [
            {
                "params": [parameters[name] for name in sorted(decay)],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
            },
            {
                "params": [parameters[name] for name in sorted(no_decay)],
                "lr": cfg["lr"],
                "weight_decay": 0.0,
            },
        ]
