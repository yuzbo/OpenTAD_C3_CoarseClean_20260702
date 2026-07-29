import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, Scale
from ..builder import HEADS, build_loss


@HEADS.register_module()
class SupportDecoupledPhysicalQueryHead(nn.Module):
    """Physical-query TAD head whose anchors are not selected observations.

    The query grid is produced by the PhysTime projection. Sparse observations
    only provide support features. Regression predicts signed center and log
    width offsets from each physical query cell, so a GT segment remains
    representable even when no observation timestamp lies inside it.
    """

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
        center_sample_radius=2.0,
        cls_prior_prob=0.01,
        endpoint_loss_weight=0.25,
        offset_loss_weight=0.25,
        label_smoothing=0.0,
        max_abs_delta_center=8.0,
        min_log_width=-6.0,
        max_log_width=6.0,
        width_reference_multiplier=2.0,
        diagnostics=True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.num_convs = int(num_convs)
        self.regression_ranges_sec = tuple(tuple(float(value) for value in item) for item in regression_ranges_sec)
        self.center_sample_radius = float(center_sample_radius)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.offset_loss_weight = float(offset_loss_weight)
        self.label_smoothing = float(label_smoothing)
        self.max_abs_delta_center = float(max_abs_delta_center)
        self.min_log_width = float(min_log_width)
        self.max_log_width = float(max_log_width)
        self.width_reference_multiplier = float(width_reference_multiplier)
        self.diagnostics_enabled = bool(diagnostics)
        self.loss_normalizer_momentum = float(loss_normalizer_momentum)
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))
        self._debug = {}

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
        probability = SupportDecoupledPhysicalQueryHead.integrated_event_probability(
            endpoint_logits, cell_widths_sec
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        return torch.logit(probability)

    def build_query_points(self, level_geometry):
        if len(level_geometry) != len(self.regression_ranges_sec):
            raise ValueError("SDPQ range count must match physical query levels")
        points = []
        for geometry, regression_range in zip(level_geometry, self.regression_ranges_sec):
            centers = geometry["centers_sec"]
            center_scale = geometry["widths_sec"].clamp_min(1.0e-8)
            width_reference = (center_scale * self.width_reference_multiplier).clamp_min(1.0e-8)
            lower = torch.full_like(centers, regression_range[0])
            upper = torch.full_like(centers, regression_range[1])
            points.append(torch.stack((centers, lower, upper, center_scale, width_reference), dim=-1))
        return tuple(points)

    def decode_segments(self, points, center_width_offsets):
        all_points = torch.cat(points, dim=1)
        offsets = torch.cat(center_width_offsets, dim=1)
        query_center = all_points[..., 0]
        center_scale = all_points[..., 3].clamp_min(1.0e-8)
        width_reference = all_points[..., 4].clamp_min(1.0e-8)
        delta_center = offsets[..., 0].clamp(
            min=-self.max_abs_delta_center, max=self.max_abs_delta_center
        )
        log_width = offsets[..., 1].clamp(min=self.min_log_width, max=self.max_log_width)
        center = query_center + center_scale * delta_center
        width = width_reference * torch.exp(log_width)
        start = center - 0.5 * width
        end = center + 0.5 * width
        return torch.stack((torch.minimum(start, end), torch.maximum(start, end)), dim=-1)

    def _tower_forward(self, tower, feature, mask):
        for layer in tower:
            feature, mask = layer(feature, mask)
        return feature

    def _predict(self, feat_list, mask_list, level_geometry):
        if not (len(feat_list) == len(mask_list) == len(level_geometry) == len(self.scales)):
            raise ValueError("SDPQ feature, mask, geometry, and level counts must match")
        cls_logits = []
        center_width_offsets = []
        endpoint_logits = []
        endpoint_probabilities = []
        for level, (feature, mask, geometry) in enumerate(zip(feat_list, mask_list, level_geometry)):
            if feature.shape[-1] != mask.shape[-1] or not torch.equal(mask, geometry["valid_mask"]):
                raise ValueError("SDPQ masks must match physical query geometry")
            cls_feature = self._tower_forward(self.cls_tower, feature, mask)
            reg_feature = self._tower_forward(self.reg_tower, feature, mask)
            cls_logits.append(self.cls_head(cls_feature))
            offsets = self.scales[level](self.reg_head(reg_feature)).float().transpose(1, 2)
            endpoint = self.endpoint_head(reg_feature).transpose(1, 2)
            center_width_offsets.append(offsets)
            endpoint_logits.append(endpoint)
            endpoint_probabilities.append(
                self.integrated_event_probability(endpoint, geometry["widths_sec"])
            )
        points = self.build_query_points(level_geometry)
        proposals = self.decode_segments(points, center_width_offsets)
        return {
            "cls_logits": tuple(cls_logits),
            "center_width_offsets": tuple(center_width_offsets),
            "endpoint_logits": tuple(endpoint_logits),
            "endpoint_probabilities": tuple(endpoint_probabilities),
            "points": points,
            "proposals_sec": proposals,
            "mask": torch.cat(mask_list, dim=1),
            "cell_widths_sec": torch.cat([item["widths_sec"] for item in level_geometry], dim=1),
            "coverage_sec": torch.cat([item["coverage_sec"] for item in level_geometry], dim=1),
        }

    @staticmethod
    def _duration_in_range(widths, lower, upper):
        return (widths >= lower) & (widths <= upper)

    @torch.no_grad()
    def _prepare_targets(self, points, level_geometry, gt_segments, gt_labels):
        all_points = torch.cat(points, dim=1)
        all_intervals = torch.cat([item["intervals_sec"] for item in level_geometry], dim=1)
        valid_mask = torch.cat([item["valid_mask"] for item in level_geometry], dim=1)
        assignment_mask = torch.cat(
            [item.get("assignment_mask", item["valid_mask"]) for item in level_geometry], dim=1
        )
        coverage_sec = torch.cat(
            [item.get("coverage_sec", torch.zeros_like(item["valid_mask"], dtype=all_points.dtype)) for item in level_geometry],
            dim=1,
        )
        cell_widths = torch.cat([item["widths_sec"] for item in level_geometry], dim=1).clamp_min(1.0e-8)
        level_ids = []
        for level, point in enumerate(points):
            level_ids.append(torch.full(point.shape[:2], level, device=point.device, dtype=torch.long))
        level_ids = torch.cat(level_ids, dim=1)

        cls_targets = []
        offset_targets = []
        segment_targets = []
        endpoint_targets = []
        debug_rows = []
        for batch_idx, (segments, labels) in enumerate(zip(gt_segments, gt_labels)):
            point = all_points[batch_idx]
            valid = valid_mask[batch_idx]
            assignable = assignment_mask[batch_idx]
            levels = level_ids[batch_idx]
            intervals = all_intervals[batch_idx]
            coverage_ratio = (coverage_sec[batch_idx] / cell_widths[batch_idx]).clamp(0, 1)
            num_points = point.shape[0]
            cls_target = point.new_zeros((num_points, self.num_classes))
            offset_target = point.new_zeros((num_points, 2))
            segment_target = point.new_zeros((num_points, 2))
            endpoint_target = point.new_zeros((num_points, 2))
            assigned_gt = torch.full((num_points,), -1, dtype=torch.long, device=point.device)

            if segments.numel() == 0:
                cls_targets.append(cls_target)
                offset_targets.append(offset_target)
                segment_targets.append(segment_target)
                endpoint_targets.append(endpoint_target)
                debug_rows.append({"gt_count": 0, "gt_without_assigned_query": 0, "short_gt_without_assigned_query": 0})
                continue

            segments = segments.to(device=point.device, dtype=point.dtype)
            labels = labels.to(device=point.device)
            gt_center = 0.5 * (segments[:, 0] + segments[:, 1])
            gt_width = (segments[:, 1] - segments[:, 0]).clamp_min(1.0e-8)
            query_center = point[:, 0]
            center_scale = point[:, 3].clamp_min(1.0e-8)
            width_reference = point[:, 4].clamp_min(1.0e-8)
            delta_center = (gt_center[None, :] - query_center[:, None]) / center_scale[:, None]
            delta_log_width = torch.log(gt_width[None, :] / width_reference[:, None])
            normalized_cost = delta_center.abs() + delta_log_width.abs()

            level_range_ok = torch.zeros((num_points, segments.shape[0]), dtype=torch.bool, device=point.device)
            for level, (lower, upper) in enumerate(self.regression_ranges_sec):
                level_mask = levels == level
                width_ok = self._duration_in_range(gt_width, lower, upper)
                level_range_ok |= level_mask[:, None] & width_ok[None, :]
            any_level = level_range_ok.any(dim=0)
            if not bool(any_level.all().item()):
                # If the configured pyramid cannot cover an unusual duration,
                # fall back to the closest normalized scale instead of losing
                # representability.
                level_range_ok[:, ~any_level] = True

            valid_candidates = assignable[:, None] & level_range_ok
            local_candidates = valid_candidates & (delta_center.abs() <= self.center_sample_radius)
            positive_candidates = local_candidates.clone()
            reserved_count = 0
            reservation_collision_count = 0
            gt_assigned = torch.zeros((segments.shape[0],), dtype=torch.bool, device=point.device)
            reserved_owner = torch.full((num_points,), -1, dtype=torch.long, device=point.device)
            candidate_counts = valid_candidates.sum(dim=0)
            # Reservation is sequential, so tied candidate counts must have an
            # explicit, cross-device tie-break.  Preserve annotation order just
            # like the independent NumPy audit instead of inheriting an
            # implementation-defined torch.argsort order.
            gt_order = torch.argsort(candidate_counts, stable=True)
            for gt_idx in gt_order.tolist():
                candidates = torch.nonzero(
                    valid_candidates[:, gt_idx] & (reserved_owner < 0),
                    as_tuple=False,
                ).flatten()
                if candidates.numel() == 0:
                    if bool(valid_candidates[:, gt_idx].any().item()):
                        reservation_collision_count += 1
                    continue
                chosen = candidates[normalized_cost[candidates, gt_idx].argmin()]
                positive_candidates[chosen, gt_idx] = True
                reserved_owner[chosen] = int(gt_idx)
                gt_assigned[gt_idx] = True
                reserved_count += 1

            candidate_cost = normalized_cost.masked_fill(~positive_candidates, float("inf"))
            min_cost, min_gt = candidate_cost.min(dim=1)
            positive = torch.isfinite(min_cost)
            reserved_positive = reserved_owner >= 0
            positive = positive | reserved_positive
            if positive.any():
                min_gt = torch.where(reserved_positive, reserved_owner.clamp_min(0), min_gt)
                assigned_gt[positive] = min_gt[positive]
                cls_target[positive, labels[min_gt[positive]].long()] = 1.0
                offset_target[positive, 0] = delta_center[positive, min_gt[positive]].clamp(
                    min=-self.max_abs_delta_center, max=self.max_abs_delta_center
                )
                offset_target[positive, 1] = delta_log_width[positive, min_gt[positive]].clamp(
                    min=self.min_log_width, max=self.max_log_width
                )
                segment_target[positive] = segments[min_gt[positive]]

            max_right = intervals[:, 1].max()
            for endpoint_index, endpoint_values in enumerate((segments[:, 0], segments[:, 1])):
                inside_cell = (endpoint_values[None, :] >= intervals[:, 0, None]) & (
                    endpoint_values[None, :] < intervals[:, 1, None]
                )
                at_final_edge = (endpoint_values[None, :] == intervals[:, 1, None]) & (
                    intervals[:, 1, None] == max_right
                )
                endpoint_target[:, endpoint_index] = (inside_cell | at_final_edge).any(dim=1).to(point.dtype)

            assigned_by_gt = torch.zeros_like(gt_assigned)
            if positive.any():
                assigned_by_gt[assigned_gt[positive].clamp_min(0)] = True
            short_gt = gt_width < 1.0
            debug_rows.append(
                {
                    "gt_count": int(segments.shape[0]),
                    "assigned_query_count": int(positive.sum().item()),
                    "reserved_match_count": int(reserved_count),
                    "reservation_collision_count": int(reservation_collision_count),
                    "gt_without_assigned_query": int((~assigned_by_gt).sum().item()),
                    "short_gt_count": int(short_gt.sum().item()),
                    "short_gt_without_assigned_query": int((short_gt & (~assigned_by_gt)).sum().item()),
                    "positive_uncovered_count": int((positive & (coverage_ratio <= 0)).sum().item()),
                    "positive_low_coverage_count": int((positive & (coverage_ratio < 1.0e-6)).sum().item()),
                }
            )
            cls_targets.append(cls_target)
            offset_targets.append(offset_target)
            segment_targets.append(segment_target)
            endpoint_targets.append(endpoint_target)
        if self.diagnostics_enabled:
            self._debug["target_assignment"] = debug_rows
        return (
            torch.stack(cls_targets),
            torch.stack(offset_targets),
            torch.stack(segment_targets),
            torch.stack(endpoint_targets),
        )

    def _losses(self, raw, mask_list, level_geometry, gt_segments, gt_labels):
        cls_target, offset_target, segment_target, endpoint_target = self._prepare_targets(
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
                target_segments = segment_target[positive_mask].float()
                reg_loss = self.reg_loss(
                    predicted_segments, target_segments, reduction="sum"
                ) / normalizer.float()
        else:
            reg_loss = raw["proposals_sec"].sum() * 0.0
        predicted_offsets = torch.cat(raw["center_width_offsets"], dim=1)
        if positive_mask.any():
            offset_loss = F.smooth_l1_loss(
                predicted_offsets[positive_mask].float(),
                offset_target[positive_mask].float(),
                reduction="sum",
            ) / normalizer.float()
        else:
            offset_loss = predicted_offsets.sum() * 0.0

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
        if self.diagnostics_enabled:
            coverage = raw["coverage_sec"]
            width = raw["cell_widths_sec"].clamp_min(1.0e-8)
            self._debug.update(
                valid_query_count=int(valid_mask.sum().item()),
                positive_query_count=num_positive,
                covered_query_count=int(((coverage > 0) & valid_mask).sum().item()),
                positive_uncovered_count=int(((coverage <= 0) & positive_mask).sum().item()),
                positive_low_coverage_count=int((((coverage / width).clamp(0, 1) < 1.0e-6) & positive_mask).sum().item()),
                mean_support_observability=float(((coverage / width).clamp(0, 1)[valid_mask]).mean().item())
                if valid_mask.any()
                else 0.0,
            )
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "offset_loss": offset_loss * self.offset_loss_weight,
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
        self._debug = {}
        raw = self._predict(feat_list, mask_list, level_geometry)
        losses = self._losses(raw, mask_list, level_geometry, gt_segments, gt_labels)
        return (losses, raw) if return_outputs else losses

    def forward_test(self, feat_list, mask_list, level_geometry, **kwargs):
        self._debug = {}
        raw = self._predict(feat_list, mask_list, level_geometry)
        scores = torch.cat([item.transpose(1, 2) for item in raw["cls_logits"]], dim=1).sigmoid()
        proposals = []
        valid_scores = []
        for batch_idx, valid_mask in enumerate(raw["mask"]):
            proposals.append(raw["proposals_sec"][batch_idx, valid_mask])
            valid_scores.append(scores[batch_idx, valid_mask])
        return proposals, valid_scores

    def collect_debug_state(self):
        return dict(self._debug)
