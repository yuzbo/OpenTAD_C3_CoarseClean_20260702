import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, Scale
from ..builder import HEADS, build_loss


@HEADS.register_module()
class PhysTimeHead(nn.Module):
    """Anchor-free TAD head whose points, ranges, and outputs are physical seconds."""

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        regression_ranges_sec,
        num_convs=2,
        loss=None,
        loss_normalizer=100.0,
        loss_normalizer_momentum=0.9,
        center_sample_radius=1.5,
        cls_prior_prob=0.01,
        endpoint_loss_weight=0.25,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.num_convs = int(num_convs)
        self.regression_ranges_sec = tuple(tuple(float(value) for value in item) for item in regression_ranges_sec)
        self.center_sample_radius = float(center_sample_radius)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.label_smoothing = float(label_smoothing)
        self.loss_normalizer_momentum = float(loss_normalizer_momentum)
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        self.cls_tower = self._make_tower()
        self.reg_tower = self._make_tower()
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.endpoint_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in self.regression_ranges_sec])
        if cls_prior_prob > 0:
            prior_bias = -math.log((1.0 - float(cls_prior_prob)) / float(cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, prior_bias)

        self.cls_loss = build_loss(loss["cls_loss"])
        self.reg_loss = build_loss(loss["reg_loss"])

    def _make_tower(self):
        layers = nn.ModuleList()
        for index in range(self.num_convs):
            layers.append(
                ConvModule(
                    self.in_channels if index == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
        return layers

    @staticmethod
    def integrated_event_probability(endpoint_logits, cell_widths_sec):
        intensity = F.softplus(endpoint_logits)
        widths = cell_widths_sec
        while widths.ndim < intensity.ndim:
            widths = widths.unsqueeze(-1)
        return -torch.expm1(-intensity * widths).clamp(max=1.0)

    @staticmethod
    def integrated_event_logit(endpoint_logits, cell_widths_sec):
        probability = PhysTimeHead.integrated_event_probability(
            endpoint_logits, cell_widths_sec
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        return torch.logit(probability)

    def build_physical_points(self, level_geometry):
        if len(level_geometry) != len(self.regression_ranges_sec):
            raise ValueError("PhysTimeHead regression range count must match physical query levels")
        points = []
        for geometry, regression_range in zip(level_geometry, self.regression_ranges_sec):
            centers = geometry["centers_sec"]
            widths = geometry["widths_sec"]
            lower = torch.full_like(centers, regression_range[0])
            upper = torch.full_like(centers, regression_range[1])
            points.append(torch.stack((centers, lower, upper, widths), dim=-1))
        return tuple(points)

    @staticmethod
    def decode_segments(points, regression_distances):
        all_points = torch.cat(points, dim=1)
        distances = torch.cat(regression_distances, dim=1)
        center = all_points[..., 0]
        width = all_points[..., 3]
        start = center - distances[..., 0] * width
        end = center + distances[..., 1] * width
        return torch.stack((torch.minimum(start, end), torch.maximum(start, end)), dim=-1)

    def _tower_forward(self, tower, feature, mask):
        for layer in tower:
            feature, mask = layer(feature, mask)
        return feature

    def _predict(self, feat_list, mask_list, level_geometry):
        if not (len(feat_list) == len(mask_list) == len(level_geometry) == len(self.scales)):
            raise ValueError("PhysTimeHead feature, mask, geometry, and level counts must match")
        cls_logits = []
        regression_distances = []
        endpoint_logits = []
        endpoint_probabilities = []
        for level, (feature, mask, geometry) in enumerate(zip(feat_list, mask_list, level_geometry)):
            if feature.shape[-1] != mask.shape[-1] or not torch.equal(mask, geometry["valid_mask"]):
                raise ValueError("PhysTimeHead masks must match physical query geometry")
            cls_feature = self._tower_forward(self.cls_tower, feature, mask)
            reg_feature = self._tower_forward(self.reg_tower, feature, mask)
            cls_logits.append(self.cls_head(cls_feature))
            regression = F.softplus(
                self.scales[level](self.reg_head(reg_feature)).float()
            ).transpose(1, 2)
            endpoint = self.endpoint_head(reg_feature).transpose(1, 2)
            regression_distances.append(regression)
            endpoint_logits.append(endpoint)
            endpoint_probabilities.append(
                self.integrated_event_probability(endpoint, geometry["widths_sec"])
            )
        points = self.build_physical_points(level_geometry)
        proposals = self.decode_segments(points, regression_distances)
        return {
            "cls_logits": tuple(cls_logits),
            "regression_distances": tuple(regression_distances),
            "endpoint_logits": tuple(endpoint_logits),
            "endpoint_probabilities": tuple(endpoint_probabilities),
            "points": points,
            "proposals_sec": proposals,
            "mask": torch.cat(mask_list, dim=1),
            "cell_widths_sec": torch.cat([item["widths_sec"] for item in level_geometry], dim=1),
            "coverage_sec": torch.cat([item["coverage_sec"] for item in level_geometry], dim=1),
        }

    @torch.no_grad()
    def _prepare_targets(self, points, level_geometry, gt_segments, gt_labels):
        all_points = torch.cat(points, dim=1)
        all_intervals = torch.cat([item["intervals_sec"] for item in level_geometry], dim=1)
        cls_targets = []
        reg_targets = []
        endpoint_targets = []
        for batch_idx, (segments, labels) in enumerate(zip(gt_segments, gt_labels)):
            point = all_points[batch_idx]
            intervals = all_intervals[batch_idx]
            num_points = point.shape[0]
            if segments.numel() == 0:
                cls_targets.append(point.new_zeros((num_points, self.num_classes)))
                reg_targets.append(point.new_zeros((num_points, 2)))
                endpoint_targets.append(point.new_zeros((num_points, 2)))
                continue

            segments = segments.to(device=point.device, dtype=point.dtype)
            labels = labels.to(device=point.device)
            left = point[:, 0, None] - segments[None, :, 0]
            right = segments[None, :, 1] - point[:, 0, None]
            distances = torch.stack((left, right), dim=-1)
            segment_lengths = (segments[:, 1] - segments[:, 0])[None, :].expand(num_points, -1).clone()

            centers = 0.5 * (segments[:, 0] + segments[:, 1])
            radius = point[:, 3, None] * self.center_sample_radius
            center_left = point[:, 0, None] - torch.maximum(centers[None, :] - radius, segments[None, :, 0])
            center_right = torch.minimum(centers[None, :] + radius, segments[None, :, 1]) - point[:, 0, None]
            inside = torch.minimum(center_left, center_right) > 0
            max_distance = distances.max(dim=-1).values
            in_range = (max_distance >= point[:, 1, None]) & (max_distance <= point[:, 2, None])
            eligible = inside & in_range
            segment_lengths.masked_fill_(~eligible, float("inf"))
            min_length, min_index = segment_lengths.min(dim=1)
            positive = torch.isfinite(min_length)

            cls_target = point.new_zeros((num_points, self.num_classes))
            if positive.any():
                cls_target[positive, labels[min_index[positive]].long()] = 1.0
            chosen_distances = distances[torch.arange(num_points, device=point.device), min_index]
            chosen_distances = chosen_distances.clamp_min(0) / point[:, 3, None].clamp_min(1.0e-8)
            chosen_distances[~positive] = 0

            endpoint_target = point.new_zeros((num_points, 2))
            max_right = intervals[:, 1].max()
            for endpoint_index, endpoint_values in enumerate((segments[:, 0], segments[:, 1])):
                inside_cell = (endpoint_values[None, :] >= intervals[:, 0, None]) & (
                    endpoint_values[None, :] < intervals[:, 1, None]
                )
                at_final_edge = (endpoint_values[None, :] == intervals[:, 1, None]) & (
                    intervals[:, 1, None] == max_right
                )
                endpoint_target[:, endpoint_index] = (inside_cell | at_final_edge).any(dim=1).to(point.dtype)

            cls_targets.append(cls_target)
            reg_targets.append(chosen_distances)
            endpoint_targets.append(endpoint_target)
        return torch.stack(cls_targets), torch.stack(reg_targets), torch.stack(endpoint_targets)

    def _losses(self, raw, mask_list, level_geometry, gt_segments, gt_labels):
        cls_target, reg_target, endpoint_target = self._prepare_targets(
            raw["points"], level_geometry, gt_segments, gt_labels
        )
        valid_mask = torch.cat(mask_list, dim=1)
        positive_mask = (cls_target.sum(dim=-1) > 0) & valid_mask
        num_positive = int(positive_mask.sum().item())
        if self.training:
            updated = self.loss_normalizer_momentum * self.loss_normalizer + (
                1.0 - self.loss_normalizer_momentum
            ) * max(num_positive, 1)
            self.loss_normalizer.copy_(updated.detach())
        normalizer = self.loss_normalizer.clamp_min(1.0)

        cls_logits = torch.cat([item.transpose(1, 2) for item in raw["cls_logits"]], dim=1)
        smoothed_target = cls_target * (1.0 - self.label_smoothing)
        smoothed_target = smoothed_target + self.label_smoothing / (self.num_classes + 1)
        cls_loss = self.cls_loss(cls_logits[valid_mask], smoothed_target[valid_mask], reduction="sum") / normalizer

        if positive_mask.any():
            with torch.cuda.amp.autocast(enabled=False):
                predicted_segments = raw["proposals_sec"][positive_mask].float()
                target_points = torch.cat(raw["points"], dim=1).float()
                target_segments = self.decode_segments(
                    (target_points,),
                    (reg_target.float(),),
                )[positive_mask]
                reg_loss = self.reg_loss(
                    predicted_segments, target_segments, reduction="sum"
                ) / normalizer.float()
        else:
            reg_loss = raw["proposals_sec"].sum() * 0.0

        with torch.cuda.amp.autocast(enabled=False):
            endpoint_event_logits = self.integrated_event_logit(
                torch.cat(raw["endpoint_logits"], dim=1).float(),
                raw["cell_widths_sec"].float(),
            )
            endpoint_loss = F.binary_cross_entropy_with_logits(
                endpoint_event_logits[valid_mask],
                endpoint_target[valid_mask].float(),
                reduction="sum",
            ) / valid_mask.sum().clamp_min(1)
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "endpoint_loss": endpoint_loss * self.endpoint_loss_weight,
        }

    def forward_train(
        self,
        feat_list,
        mask_list,
        level_geometry,
        gt_segments,
        gt_labels,
        return_outputs=False,
        **kwargs,
    ):
        raw = self._predict(feat_list, mask_list, level_geometry)
        losses = self._losses(raw, mask_list, level_geometry, gt_segments, gt_labels)
        return (losses, raw) if return_outputs else losses

    def forward_test(self, feat_list, mask_list, level_geometry, **kwargs):
        raw = self._predict(feat_list, mask_list, level_geometry)
        scores = torch.cat([item.transpose(1, 2) for item in raw["cls_logits"]], dim=1).sigmoid()
        proposals = []
        valid_scores = []
        for batch_idx, valid_mask in enumerate(raw["mask"]):
            proposals.append(raw["proposals_sec"][batch_idx, valid_mask])
            valid_scores.append(scores[batch_idx, valid_mask])
        return proposals, valid_scores
