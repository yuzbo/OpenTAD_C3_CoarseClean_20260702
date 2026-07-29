import inspect
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from collections.abc import Mapping

from ..builder import DETECTORS, build_selector, build_token_compressor
from .single_stage import SingleStageDetector
from ..bricks import Scale, AffineDropPath
from ..duca.true_time_residual import TrueTimeFeatureResidual


_PC_OT_MRAS_READER_OUTPUTS_META_KEY = "pc_ot_mras_reader_outputs"
_PC_OT_MRAS_VALUE_TARGET_KEYS = (
    "pc_ot_mras_value_targets",
    "value_transport_targets",
    "teacher_value_targets",
    "pc_ot_mras_value_manifest",
)


@DETECTORS.register_module()
class ActionFormer(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        frame_selector=None,
        token_compressor=None,
        true_time_residual=None,
        pc_ot_mras_reader=None,
        pc_ot_mras_reader_feature_level=0,
        pc_ot_mras_reader_aux_loss=None,
        pc_ot_mras_reader_soft_hard_loss=None,
        pc_ot_mras_reader_value_loss=None,
        pc_ot_mras_reader_eval_override=None,
        selector_train_only=False,
        selector_train_only_skip_detector=False,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )
        self.frame_selector = build_selector(frame_selector) if frame_selector is not None else None
        self.token_compressor = build_token_compressor(token_compressor) if token_compressor is not None else None
        self.true_time_residual = (
            TrueTimeFeatureResidual(**dict(true_time_residual))
            if true_time_residual is not None
            else None
        )
        self.pc_ot_mras_reader = build_selector(pc_ot_mras_reader) if pc_ot_mras_reader is not None else None
        self.pc_ot_mras_reader_feature_level = int(pc_ot_mras_reader_feature_level)
        self.pc_ot_mras_reader_aux_loss = self._normalize_pc_ot_mras_reader_aux_loss(pc_ot_mras_reader_aux_loss)
        self.pc_ot_mras_reader_soft_hard_loss = self._normalize_pc_ot_mras_reader_soft_hard_loss(
            pc_ot_mras_reader_soft_hard_loss
        )
        self.pc_ot_mras_reader_value_loss = self._normalize_pc_ot_mras_reader_value_loss(pc_ot_mras_reader_value_loss)
        self.pc_ot_mras_reader_eval_override = self._normalize_pc_ot_mras_reader_eval_override(
            pc_ot_mras_reader_eval_override
        )
        if self.pc_ot_mras_reader_aux_loss is not None and self.pc_ot_mras_reader is None:
            raise ValueError("pc_ot_mras_reader_aux_loss requires pc_ot_mras_reader")
        if self.pc_ot_mras_reader_soft_hard_loss is not None and self.pc_ot_mras_reader is None:
            raise ValueError("pc_ot_mras_reader_soft_hard_loss requires pc_ot_mras_reader")
        if self.pc_ot_mras_reader_value_loss is not None and self.pc_ot_mras_reader is None:
            raise ValueError("pc_ot_mras_reader_value_loss requires pc_ot_mras_reader")
        self.selector_train_only = bool(selector_train_only)
        self.selector_train_only_skip_detector = bool(
            selector_train_only_skip_detector
        )
        if self.selector_train_only_skip_detector and not self.selector_train_only:
            raise ValueError(
                "selector_train_only_skip_detector=True requires selector_train_only=True"
            )
        if self.selector_train_only:
            if self.frame_selector is None:
                raise ValueError("selector_train_only=True requires frame_selector")
            self._freeze_non_selector_trainable_parameters()

        n_mha_win_size = self.projection.n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + self.projection.arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + self.projection.arch[-1])
            self.mha_win_size = n_mha_win_size
        self.max_seq_len = self.projection.max_seq_len
        if self.token_compressor is not None and hasattr(self.token_compressor, "target_len"):
            compressor_target_len = int(self.token_compressor.target_len)
            if compressor_target_len != int(self.max_seq_len):
                raise ValueError(
                    "token_compressor.target_len must equal projection.max_seq_len; "
                    f"got {compressor_target_len} and {self.max_seq_len}. "
                    "Set both to the compressed bucket length to avoid padding tokens back."
                )

        max_div_factor = 1
        for s, w in zip(self.rpn_head.prior_generator.strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert (
                self.max_seq_len % stride == 0
            ), f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks
        elif feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:  # feat_len > self.max_seq_len
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def train(self, mode=True):
        super().train(mode)
        if self.selector_train_only:
            self.frame_selector.train(mode)
            for module in (
                getattr(self, "backbone", None),
                getattr(self, "projection", None),
                getattr(self, "neck", None),
                getattr(self, "rpn_head", None),
            ):
                if module is not None:
                    module.eval()
        return self

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        skip_frame_selector = bool(kwargs.pop("_duca_skip_frame_selector", False))
        counterfactual_eval = bool(kwargs.pop("_duca_counterfactual_eval", False))
        gt_boundary_validity = kwargs.pop("gt_boundary_validity", None)
        losses = dict()
        selector_loss_keys = set()
        raw_selector_context = None
        detector_rng_state = self._capture_protected_detector_rng(inputs)
        if self.frame_selector is not None and not skip_frame_selector:
            raw_selector_context = (inputs, masks, metas, gt_segments, gt_labels)
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
            self._validate_protected_selector_contract(
                inputs,
                masks,
                metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                original_gt_segments=raw_selector_context[3],
                original_gt_labels=raw_selector_context[4],
            )
            selector_losses = selector_outputs.get("losses", {})
            self._validate_selector_losses(selector_losses)
            for key, value in selector_losses.items():
                if key in losses:
                    raise ValueError(f"frame_selector loss key collision: {key}")
                losses[key] = value
            selector_loss_keys = set(losses)
            if self.selector_train_only:
                inputs = inputs.detach()

        request = (
            selector_outputs.get("counterfactual_request")
            if self.frame_selector is not None and not skip_frame_selector
            else None
        )
        if (
            self.frame_selector is not None
            and not skip_frame_selector
            and self.frame_selector.require_counterfactual_utility_teacher
            and request is None
        ):
            raise RuntimeError("required integrated counterfactual teacher request is missing")
        if self.selector_train_only_skip_detector:
            if request is not None:
                raise RuntimeError(
                    "selector-only detector skipping is incompatible with a counterfactual detector teacher"
                )
            if isinstance(self.frame_selector.last_forward_summary, dict):
                self.frame_selector.last_forward_summary[
                    "frontend_only_detector_skipped"
                ] = True
            return self._selector_only_losses(losses, selector_loss_keys)
        counterfactual_loss = None
        if request is not None:
            if raw_selector_context is None:
                raise RuntimeError("counterfactual teacher requires the pre-selector training context")
            # Evaluate and restore every train-only hard alternative before the
            # main detector graph is built. Restoring mutable training buffers
            # after that graph exists would invalidate autograd version checks.
            counterfactual_loss = self._duca_counterfactual_teacher_loss(
                raw_selector_context, selector_outputs["selector_outputs"], request, **kwargs
            )

        self._restore_protected_detector_rng(detector_rng_state)
        if self.with_backbone:
            x = self._forward_backbone_with_temporal_mask(inputs, masks)
        else:
            x = inputs

        self._assert_feature_mask_temporal_match(x, masks, "before token_compressor")
        if self.token_compressor is not None:
            compressor_outputs = self.token_compressor.forward_train(
                features=x,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
            )
            x = compressor_outputs["features"]
            masks = compressor_outputs["masks"]
            metas = compressor_outputs.get("metas", metas)
            gt_segments = compressor_outputs["gt_segments"]
            gt_labels = compressor_outputs["gt_labels"]
            losses.update(compressor_outputs.get("losses", {}))
            self._assert_feature_mask_temporal_match(x, masks, "after token_compressor")
            if x.shape[-1] != self.max_seq_len:
                raise RuntimeError(
                    "token_compressor output length must match projection.max_seq_len before pad_data; "
                    f"got {x.shape[-1]} and {self.max_seq_len}"
                )

        if self.true_time_residual is not None:
            self._assert_feature_mask_temporal_match(x, masks, "before true_time_residual")
            x = self.true_time_residual(x, masks, metas)
            self._assert_feature_mask_temporal_match(x, masks, "after true_time_residual")

        # pad the features and unsqueeze the mask for actionformer
        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        metas = self._inject_pc_ot_mras_reader_outputs(x, masks, metas)
        reader_extra_losses = {}
        reader_extra_losses.update(self._pc_ot_mras_reader_auxiliary_losses(metas, gt_segments))
        reader_extra_losses.update(self._pc_ot_mras_reader_soft_hard_losses(metas))
        reader_extra_losses.update(self._pc_ot_mras_reader_value_losses(metas))
        metas = self._strip_pc_ot_mras_value_targets_from_metas(metas)

        if self.with_neck:
            x, masks, metas = self._call_neck_forward(x, masks, metas=metas)

        loc_losses = self._call_rpn_head_forward_train(
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
        if (
            self.frame_selector is not None
            and not skip_frame_selector
            and not counterfactual_eval
            and hasattr(self.frame_selector, "detector_contribution_distillation_losses")
        ):
            contribution_losses = self.frame_selector.detector_contribution_distillation_losses(
                selector_outputs=selector_outputs["selector_outputs"],
                detector_losses=loc_losses,
                selected_inputs=selector_outputs["selector_outputs"].get(
                    "detector_contribution_teacher_inputs",
                    inputs,
                ),
            )
            for key, value in contribution_losses.items():
                if key in losses:
                    raise ValueError(f"detector contribution loss key collision: {key}")
                losses[key] = value
        self._merge_pc_ot_mras_extra_losses(
            losses,
            reader_extra_losses,
            source_name="pc_ot_mras_reader_losses",
        )

        if counterfactual_loss is not None:
            if "counterfactual_utility_distillation_loss" in losses:
                raise ValueError("counterfactual distillation loss key collision")
            losses["counterfactual_utility_distillation_loss"] = counterfactual_loss

        if counterfactual_eval:
            return losses

        # only key has loss will be record
        if self.selector_train_only:
            losses = self._selector_only_losses(losses, selector_loss_keys)
        else:
            losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    @staticmethod
    def _selector_only_losses(losses, selector_loss_keys):
        selector_loss_keys = tuple(sorted(str(key) for key in selector_loss_keys))
        if not selector_loss_keys:
            raise RuntimeError("selector_train_only=True requires at least one selector loss")
        selector_losses = {key: losses[key] for key in selector_loss_keys}
        selector_losses["cost"] = sum(selector_losses.values())
        return selector_losses

    @staticmethod
    def _duca_gather_raw(inputs, positions):
        if inputs.ndim not in (3, 5, 6):
            raise ValueError("DUCA counterfactual gather requires [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")
        time_dim = 2 if inputs.ndim in (3, 5) else 3
        view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
        view[time_dim] = positions.shape[1]
        expand = list(inputs.shape)
        expand[time_dim] = positions.shape[1]
        slot_mask = positions >= 0
        index = positions.clamp_min(0).view(view).expand(expand)
        gathered = torch.gather(inputs, time_dim, index)
        mask_view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
        mask_view[time_dim] = positions.shape[1]
        return gathered * slot_mask.view(mask_view).to(dtype=gathered.dtype)

    @staticmethod
    def _duca_detector_objective(losses):
        terms = [
            value for key, value in losses.items()
            if key != "cost" and key.endswith("loss") and ("cls" in key or "reg" in key)
        ]
        if not terms:
            raise RuntimeError("official ActionFormer counterfactual pass produced no cls/reg loss")
        objective = sum(terms)
        if objective.ndim != 0 or not torch.isfinite(objective):
            raise RuntimeError("counterfactual cls+reg objective must be a finite scalar")
        return objective

    @staticmethod
    def _duca_counterfactual_teacher_autocast(inputs):
        if inputs.is_cuda and torch.is_autocast_enabled():
            # The teacher is the first consumer of the detector parameters in
            # this forward. Caching its no-grad FP16 casts would make the main
            # detector reuse detached weights under the outer autocast context.
            return torch.autocast(
                device_type="cuda",
                dtype=torch.get_autocast_gpu_dtype(),
                enabled=True,
                cache_enabled=False,
            )
        return nullcontext()

    def _duca_counterfactual_teacher_loss(self, raw_context, selector_state, request, **kwargs):
        raw_inputs, raw_masks, raw_metas, raw_segments, raw_labels = raw_context
        selections = request["candidate_selections"]
        candidate_valid = request["candidate_valid"]
        baseline_positions = selector_state["grid"].selected_positions
        detector_grid_positions = request.get("detector_grid_positions", baseline_positions)
        if detector_grid_positions.shape != baseline_positions.shape:
            raise RuntimeError("counterfactual detector grid must align with acquisition positions")
        utilities = selector_state["center_scores"].new_zeros(
            candidate_valid.shape,
            dtype=torch.float32,
        )
        baseline_detector_loss = selector_state["center_scores"].new_full(
            (candidate_valid.shape[0],),
            float("nan"),
            dtype=torch.float32,
        )
        candidate_detector_loss = selector_state["center_scores"].new_full(
            candidate_valid.shape,
            float("nan"),
            dtype=torch.float32,
        )

        if raw_metas is None or raw_segments is None or raw_labels is None:
            raise RuntimeError("counterfactual teacher requires train-only metas, segments and labels")

        def evaluate_one(batch_index, positions, detector_positions):
            positions = positions.reshape(1, -1).to(device=raw_inputs.device)
            detector_positions = detector_positions.reshape(1, -1).to(device=raw_inputs.device)
            selected_inputs = self._duca_gather_raw(raw_inputs[batch_index : batch_index + 1], positions)
            selected_masks = positions >= 0
            meta = dict(raw_metas[batch_index])
            active = selected_masks[0]
            detector_pos_list = [
                int(x)
                for x in detector_positions[0, active].detach().cpu().tolist()
            ]
            meta["selected_axis_to_true_time_dense_index"] = detector_pos_list
            meta["duca_acquisition_positions"] = [
                int(x) for x in positions[0, active].detach().cpu().tolist()
            ]
            meta["duca_detector_grid_positions"] = detector_pos_list
            meta["truetime_dense_len"] = int(raw_masks.shape[-1])
            meta["truetime_dense_valid_len"] = int(raw_masks[batch_index].sum().item())
            remapped_segments, remapped_labels, remapped_metas = self.frame_selector._remap_train_targets_to_selected_axis(
                [raw_segments[batch_index]], [raw_labels[batch_index]], [meta]
            )
            candidate_losses = self.forward_train(
                selected_inputs, selected_masks, remapped_metas, remapped_segments, remapped_labels,
                _duca_skip_frame_selector=True, _duca_counterfactual_eval=True, **kwargs
            )
            return self._duca_detector_objective(candidate_losses)

        modules = tuple(self.modules())
        module_training = {module: module.training for module in modules}
        buffer_state = {
            name: value.detach().clone()
            for name, value in self.named_buffers()
            if not name.startswith("frame_selector.")
        }
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        normalizer = self.rpn_head.loss_normalizer.detach().clone()

        def restore_teacher_state():
            current_buffers = dict(self.named_buffers())
            for name, value in buffer_state.items():
                current_buffers[name].copy_(value)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

        try:
            if not self.training or not self.rpn_head.training:
                raise RuntimeError("counterfactual teacher must use the official training objective")
            self.rpn_head.duca_set_frozen_loss_normalizer(normalizer)
            with torch.no_grad(), self._duca_counterfactual_teacher_autocast(raw_inputs):
                for b in range(baseline_positions.shape[0]):
                    if not bool(candidate_valid[b].any().item()):
                        continue
                    restore_teacher_state()
                    baseline_loss = evaluate_one(b, baseline_positions[b], detector_grid_positions[b])
                    baseline_detector_loss[b] = baseline_loss
                    for m in range(selections.shape[1]):
                        if candidate_valid[b, m]:
                            restore_teacher_state()
                            candidate_loss = evaluate_one(
                                b,
                                selections[b, m],
                                detector_grid_positions[b],
                            )
                            candidate_detector_loss[b, m] = candidate_loss
                            utilities[b, m] = baseline_loss - candidate_loss
        finally:
            self.rpn_head.duca_set_frozen_loss_normalizer(None)
            restore_teacher_state()
            for module, training in module_training.items():
                module.training = training
        if not torch.isfinite(utilities[candidate_valid]).all():
            raise RuntimeError("counterfactual utility teacher produced non-finite gain")
        active_batches = candidate_valid.any(dim=1)
        if not torch.isfinite(baseline_detector_loss[active_batches]).all():
            raise RuntimeError("counterfactual utility teacher produced non-finite baseline loss")
        if not torch.isfinite(candidate_detector_loss[candidate_valid]).all():
            raise RuntimeError("counterfactual utility teacher produced non-finite candidate loss")
        return self.frame_selector.counterfactual_distillation_loss(
            selector_state, request["candidate_positions"], request["replaced_slots"],
            utilities.detach(), candidate_valid,
            baseline_detector_loss=baseline_detector_loss.detach(),
            candidate_detector_loss=candidate_detector_loss.detach(),
        )

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        self._reject_pc_ot_mras_value_targets_in_forward_test(metas)
        detector_rng_state = self._capture_protected_detector_rng(inputs)
        if self.frame_selector is not None:
            selector_outputs = self.frame_selector.forward_test(
                inputs=inputs,
                masks=masks,
                metas=metas,
            )
            inputs = selector_outputs["inputs"]
            masks = selector_outputs["masks"]
            metas = selector_outputs.get("metas", metas)
            self._validate_protected_selector_contract(
                inputs,
                masks,
                metas,
            )
            self._reject_pc_ot_mras_value_targets_in_forward_test(metas)
            self._require_selector_remap_metadata(metas)

        self._restore_protected_detector_rng(detector_rng_state)
        if self.with_backbone:
            x = self._forward_backbone_with_temporal_mask(inputs, masks)
        else:
            x = inputs

        self._assert_feature_mask_temporal_match(x, masks, "before token_compressor")
        if self.token_compressor is not None:
            compressor_outputs = self.token_compressor.forward_test(
                features=x,
                masks=masks,
                metas=metas,
            )
            x = compressor_outputs["features"]
            masks = compressor_outputs["masks"]
            metas = compressor_outputs.get("metas", metas)
            self._reject_pc_ot_mras_value_targets_in_forward_test(metas)
            self._assert_feature_mask_temporal_match(x, masks, "after token_compressor")
            if x.shape[-1] != self.max_seq_len:
                raise RuntimeError(
                    "token_compressor output length must match projection.max_seq_len before pad_data; "
                    f"got {x.shape[-1]} and {self.max_seq_len}"
                )

        if self.true_time_residual is not None:
            self._assert_feature_mask_temporal_match(x, masks, "before true_time_residual")
            x = self.true_time_residual(x, masks, metas)
            self._assert_feature_mask_temporal_match(x, masks, "after true_time_residual")

        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        metas = self._inject_pc_ot_mras_reader_outputs(x, masks, metas)

        if self.with_neck:
            x, masks, metas = self._call_neck_forward(x, masks, metas=metas)

        rpn_proposals, rpn_scores = self._call_rpn_head_forward_test(x, masks, metas=metas, **kwargs)
        self._last_forward_test_metas = metas
        predictions = rpn_proposals, rpn_scores
        return predictions

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
                elif pn.endswith("query_embed"):
                    # learned slot/query embeddings should not receive weight decay
                    no_decay.add(fpn)
                elif pn.endswith("slot_queries"):
                    # pre-backbone selector reader slot queries are learned embeddings
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone") and p.requires_grad}
        for fpn, param in param_dict.items():
            if fpn in decay or fpn in no_decay or not fpn.startswith("frame_selector."):
                continue
            if fpn.endswith(".weight") and param.ndim >= 2:
                decay.add(fpn)
            else:
                no_decay.add(fpn)

        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        selector = getattr(self, "frame_selector", None)
        transition_only = getattr(selector, "selector_variant", None) == "transition_only"
        protected_e2e = (
            getattr(selector, "selector_variant", None)
            in {
                "protected_e2e_physical",
                "protected_e2e_selected_axis",
            }
        )
        rime = getattr(selector, "selector_variant", None) in {
            "duca_rime_physical",
            "duca_rime_selected_axis",
        }

        def parameter_lr(name):
            scorer_prefixes = (
                "frame_selector.adapter.transition_scorer.",
                "frame_selector.adapter.density_mixture_head.",
                "frame_selector.adapter.sampling_rate_utility_fusion.",
            )
            protected_adapter_prefix = (
                "frame_selector.transition_scorer.selector_adapter."
            )
            protected_head_prefix = (
                "frame_selector.transition_scorer.selector_score_head."
            )
            coarse_prefix = "frame_selector.raw_actionness_source.probe_module."
            if transition_only and name.startswith(scorer_prefixes):
                return float(selector.transition_scorer_lr)
            if (protected_e2e or rime) and name.startswith(
                (protected_adapter_prefix, protected_head_prefix)
            ):
                return float(selector.selector_lr)
            if rime and name.startswith("frame_selector.budget_controller."):
                return float(selector.selector_lr)
            if name.startswith(coarse_prefix):
                temporal_marker = ".official_temporal."
                tail = name.split(temporal_marker, 1)[1] if temporal_marker in name else ""
                parts = tail.split(".")
                is_encoder_action_head = tail.startswith("encoder.conv_out.")
                is_decoder_action_head = (
                    len(parts) >= 4 and parts[0] == "decoders" and parts[2] == "conv_out"
                )
                if is_encoder_action_head or is_decoder_action_head:
                    return float(selector.action_head_lr)
                return float(selector.coarse_trunk_lr)
            return float(cfg["lr"])

        if protected_e2e or rime:
            protected_prefixes = (
                "frame_selector.transition_scorer.selector_adapter.",
                "frame_selector.transition_scorer.selector_score_head.",
                "frame_selector.raw_actionness_source.probe_module.",
            )
            if rime:
                protected_prefixes += ("frame_selector.budget_controller.",)
            unexpected = sorted(
                name
                for name in param_dict
                if name.startswith("frame_selector.")
                and not name.startswith(protected_prefixes)
            )
            if unexpected:
                raise AssertionError(
                    "protected/RIME DUCA exposes trainable selector parameters outside "
                    f"the frozen optimizer contract: {unexpected}"
                )

        grouped = {}
        for names, weight_decay in ((decay, float(cfg["weight_decay"])), (no_decay, 0.0)):
            for name in sorted(names):
                lr = parameter_lr(name)
                grouped.setdefault((lr, weight_decay), []).append(param_dict[name])
        optim_groups = [
            {"params": params, "weight_decay": weight_decay, "lr": lr}
            for (lr, weight_decay), params in sorted(grouped.items())
            if params
        ]
        return optim_groups

    def _capture_protected_detector_rng(self, inputs):
        selector = getattr(self, "frame_selector", None)
        if selector is None or not bool(
            getattr(selector, "separate_detector_rng", False)
        ):
            return None
        return {
            "cpu": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.is_tensor(inputs) and inputs.is_cuda
                else None
            ),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

    @staticmethod
    def _restore_protected_detector_rng(state):
        if state is None:
            return
        torch.random.set_rng_state(state["cpu"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])

    def _validate_protected_selector_contract(
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
        selector_variant = getattr(selector, "selector_variant", None)
        if selector_variant in {
            "protected_e2e_selected_axis",
            "duca_rime_selected_axis",
        }:
            self._validate_rime_selected_axis_contract(
                inputs,
                masks,
                metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                original_gt_segments=original_gt_segments,
                original_gt_labels=original_gt_labels,
            )
            return
        if selector_variant not in {"protected_e2e_physical", "duca_rime_physical"}:
            return
        if masks.ndim != 2 or int(masks.shape[1]) > int(selector.budget):
            raise RuntimeError("protected DUCA detector mask violates exact-K budget")
        temporal_dim = 3 if inputs.ndim == 6 else 2
        if inputs.ndim not in {3, 5, 6} or int(inputs.shape[temporal_dim]) != int(
            masks.shape[1]
        ):
            raise RuntimeError(
                "protected DUCA detector input and mask temporal axes disagree"
            )
        if not isinstance(metas, (list, tuple)) or len(metas) != int(
            inputs.shape[0]
        ):
            raise RuntimeError("protected DUCA requires one detector meta per sample")
        for meta, mask in zip(metas, masks):
            expected_contract = (
                "duca_rime_physical_dynamic_k_v1"
                if selector_variant == "duca_rime_physical"
                else "duca_protected_e2e_physical_v1"
            )
            if meta.get("duca_contract") != expected_contract:
                raise RuntimeError("protected/RIME DUCA metadata contract is missing")
            if meta.get("irregular_native_axis") is not True:
                raise RuntimeError("protected DUCA requires native dense detector axis")
            if any(
                bool(meta.get(key, False))
                for key in (
                    "remap_gt_to_selected_axis",
                    "gt_remapped_to_selected_axis",
                    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis",
                    "selected_axis_remap_required",
                    "detector_prediction_inverse_map_required",
                )
            ):
                raise RuntimeError("protected DUCA forbids selected-axis remapping")
            if meta.get("detector_output_coordinate_space") != "dense_physical":
                raise RuntimeError(
                    "protected DUCA detector output must remain in dense physical coordinates"
                )
            selected_count = int(meta.get("irregular_selected_count", -1))
            if selected_count != int(mask.to(dtype=torch.bool).sum().item()):
                raise RuntimeError(
                    "protected DUCA metadata selected count does not match detector mask"
                )
            if selector_variant in {
                "protected_e2e_physical",
                "duca_rime_physical",
            }:
                requested = int(meta.get("duca_requested_k", -1))
                effective = int(meta.get("duca_effective_k", -1))
                unique = int(meta.get("duca_unique_k", -1))
                backbone = int(meta.get("duca_backbone_input_k", -1))
                padded = int(meta.get("duca_padded_k", -1))
                if not (
                    requested >= effective
                    and effective == unique == backbone == padded == selected_count
                ):
                    raise RuntimeError(
                        "protected/RIME requested/effective/heavy-frame ledger "
                        "is inconsistent"
                    )
                if meta.get("duca_dynamic_compute_realized") is not True:
                    raise RuntimeError(
                        "protected/RIME DUCA must realize dynamic heavy-backbone compute"
                    )
                if meta.get("duca_backbone_tail_padding_mode") != "none_exact_k_bucket":
                    raise RuntimeError(
                        "protected/RIME DUCA forbids padded heavy execution"
                    )
                execution_quantum = int(
                    getattr(selector, "execution_quantum", -1)
                )
                if (
                    execution_quantum <= 0
                    or int(meta.get("duca_execution_quantum", -1))
                    != execution_quantum
                    or effective % execution_quantum != 0
                ):
                    raise RuntimeError(
                        "protected/RIME heavy execution violates its exact quantum"
                    )
        if original_gt_segments is not None and gt_segments is not original_gt_segments:
            raise RuntimeError("protected DUCA must preserve dense GT segment objects")
        if original_gt_labels is not None and gt_labels is not original_gt_labels:
            raise RuntimeError("protected DUCA must preserve dense GT label objects")

    def _validate_rime_selected_axis_contract(
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
        if bool(getattr(self.rpn_head, "physical_grid_enabled", False)) or bool(
            getattr(self.rpn_head, "physical_grid_required", False)
        ):
            raise RuntimeError(
                "pure selected-axis RIME requires the standard detector head"
            )
        if masks.ndim != 2 or not bool(
            masks.to(dtype=torch.bool).all().item()
        ):
            raise RuntimeError(
                "selected-axis RIME detector mask must be exact-K and padding-free"
            )
        temporal_dim = 3 if inputs.ndim == 6 else 2
        if inputs.ndim not in {3, 5, 6} or int(inputs.shape[temporal_dim]) != int(
            masks.shape[1]
        ):
            raise RuntimeError(
                "selected-axis RIME detector input and mask temporal axes disagree"
            )
        if not isinstance(metas, (list, tuple)) or len(metas) != int(
            inputs.shape[0]
        ):
            raise RuntimeError(
                "selected-axis RIME requires one detector meta per sample"
            )
        for meta, mask in zip(metas, masks):
            expected_contract = str(
                getattr(
                    getattr(self, "frame_selector", None),
                    "detector_coordinate_contract",
                    "",
                )
            )
            if meta.get("duca_contract") != expected_contract:
                raise RuntimeError(
                    "selected-axis RIME metadata contract is missing"
                )
            if meta.get("detector_coordinate_mode") != "selected_axis_plugin":
                raise RuntimeError(
                    "selected-axis RIME coordinate mode is not explicit"
                )
            if meta.get("physical_grid_contract") is not None or bool(
                meta.get("physical_grid_actionformer", False)
            ):
                raise RuntimeError(
                    "pure selected-axis RIME metadata contains physical-head state"
                )
            required_flags = (
                "remap_gt_to_selected_axis",
                "selected_axis_remap_required",
                "detector_prediction_inverse_map_required",
            )
            if not all(bool(meta.get(key, False)) for key in required_flags):
                raise RuntimeError(
                    "selected-axis RIME mapping contract is incomplete"
                )
            if (
                meta.get("detector_output_coordinate_space")
                != "selected_axis_index"
                or meta.get("proposal_axis") != "selected_axis_index"
            ):
                raise RuntimeError(
                    "pure RIME detector outputs must stay on the selected axis"
                )
            positions = meta.get("selected_axis_to_true_time_dense_index")
            if not isinstance(positions, (list, tuple)) or not positions:
                raise RuntimeError(
                    "selected-axis RIME inverse-map positions are missing"
                )
            positions = [int(value) for value in positions]
            if any(
                right <= left for left, right in zip(positions, positions[1:])
            ):
                raise RuntimeError(
                    "selected-axis RIME inverse-map positions are not ordered"
                )
            dense_valid_len = int(meta.get("irregular_dense_valid_len", -1))
            true_time_dense_valid_len = int(
                meta.get("truetime_dense_valid_len", -1)
            )
            if (
                dense_valid_len <= 0
                or true_time_dense_valid_len != dense_valid_len
                or any(
                    position < 0 or position >= dense_valid_len
                    for position in positions
                )
            ):
                raise RuntimeError(
                    "selected-axis RIME inverse-map positions exceed the "
                    "declared dense valid axis"
                )
            selected_count = int(meta.get("irregular_selected_count", -1))
            active_count = int(mask.to(dtype=torch.bool).sum().item())
            if selected_count != len(positions) or selected_count != active_count:
                raise RuntimeError(
                    "selected-axis RIME selected-count contract is inconsistent"
                )
            requested = int(meta.get("duca_requested_k", -1))
            effective = int(meta.get("duca_effective_k", -1))
            unique = int(meta.get("duca_unique_k", -1))
            backbone = int(meta.get("duca_backbone_input_k", -1))
            padded = int(meta.get("duca_padded_k", -1))
            if not (
                requested >= effective
                and effective
                == unique
                == backbone
                == padded
                == selected_count
            ):
                raise RuntimeError(
                    "selected-axis RIME exact-K execution ledger is inconsistent"
                )
            if meta.get("duca_dynamic_compute_realized") is not True:
                raise RuntimeError(
                    "selected-axis RIME must realize dynamic heavy compute"
                )
            if (
                meta.get("duca_backbone_tail_padding_mode")
                != "none_exact_k_bucket"
            ):
                raise RuntimeError(
                    "selected-axis RIME forbids padded heavy execution"
                )
        if original_gt_segments is not None:
            if gt_segments is original_gt_segments:
                raise RuntimeError(
                    "selected-axis RIME must remap dense GT before the head"
                )
            if not all(
                bool(meta.get("gt_remapped_to_selected_axis", False))
                for meta in metas
            ):
                raise RuntimeError(
                    "selected-axis RIME GT remap receipt is missing"
                )
        if original_gt_labels is not None and gt_labels is not original_gt_labels:
            raise RuntimeError(
                "selected-axis RIME must preserve GT label objects"
            )

    def _freeze_non_selector_trainable_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("frame_selector."):
                param.requires_grad = False
        if self.with_backbone and hasattr(self.backbone, "freeze_backbone"):
            self.backbone.freeze_backbone = True

    def _call_rpn_head_forward_train(self, feat_list, mask_list, metas, gt_segments, gt_labels, **kwargs):
        call_kwargs = dict(kwargs)
        if self._head_accepts_metas(self.rpn_head.forward_train):
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
        if self._head_accepts_metas(self.rpn_head.forward_test):
            call_kwargs["metas"] = metas
        return self.rpn_head.forward_test(feat_list, mask_list, **call_kwargs)

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

    @staticmethod
    def _head_accepts_metas(fn):
        return ActionFormer._callable_accepts_metas(fn)

    @staticmethod
    def _callable_accepts_metas(fn):
        signature = inspect.signature(fn)
        for param in signature.parameters.values():
            if param.name == "metas":
                return True
        return False

    @staticmethod
    def _assert_feature_mask_temporal_match(features, masks, stage):
        if features.shape[-1] != masks.shape[-1]:
            raise RuntimeError(
                f"feature/mask temporal length mismatch {stage}: "
                f"features={features.shape[-1]}, masks={masks.shape[-1]}"
            )

    def _inject_pc_ot_mras_reader_outputs(self, feat_list, mask_list, metas):
        if self.pc_ot_mras_reader is None and self.pc_ot_mras_reader_eval_override is None:
            return metas
        features, masks = self._pc_ot_mras_reader_feature_and_mask(feat_list, mask_list)
        if features.shape[0] != masks.shape[0] or features.shape[-1] != masks.shape[-1]:
            raise RuntimeError(
                "PC-OT-MRAS reader feature/mask shape mismatch after projection: "
                f"features={tuple(features.shape)}, masks={tuple(masks.shape)}"
            )
        output_metas = self._clone_pc_ot_mras_metas_for_writer(metas, batch_size=int(features.shape[0]))
        if self.pc_ot_mras_reader_eval_override is not None and not self.training:
            reader_outputs = self._pc_ot_mras_eval_override_outputs(features, masks)
        elif self.pc_ot_mras_reader is None:
            raise ValueError("pc_ot_mras_reader is required outside eval override mode")
        else:
            reader_outputs = self.pc_ot_mras_reader(features.transpose(1, 2).contiguous(), masks)
        for meta in output_metas:
            meta[_PC_OT_MRAS_READER_OUTPUTS_META_KEY] = reader_outputs
        return output_metas

    def _pc_ot_mras_eval_override_outputs(self, features, masks):
        config = self.pc_ot_mras_reader_eval_override
        mode = config["mode"]
        if mode != "exact_uniform":
            raise ValueError(f"unsupported pc_ot_mras_reader_eval_override mode: {mode}")
        num_slots = int(config["num_slots"])
        batch, _, time = features.shape
        valid = masks.bool()
        valid_len = valid.long().sum(dim=1)
        if bool((valid_len <= 0).any().item()):
            raise ValueError("pc_ot_mras_reader_eval_override requires at least one valid token per sample")
        allocation = features.new_zeros((batch, num_slots, time))
        selected_mask = torch.zeros((batch, num_slots), dtype=torch.bool, device=features.device)
        selected_times = features.new_zeros((batch, num_slots))
        centers = features.new_zeros((batch, num_slots))
        widths = features.new_zeros((batch, num_slots))
        gates = features.new_zeros((batch, num_slots))
        time_coords = self._pc_ot_mras_dense_time_coords(valid, dtype=features.dtype)

        for batch_idx in range(batch):
            count = min(num_slots, int(valid_len[batch_idx].item()))
            positions = self._pc_ot_mras_exact_uniform_positions(valid_len[batch_idx], count)
            slots = torch.arange(count, device=features.device)
            allocation[batch_idx, slots, positions] = 1.0
            selected_mask[batch_idx, :count] = True
            selected_times[batch_idx, :count] = time_coords[batch_idx, positions]
            centers[batch_idx, :count] = selected_times[batch_idx, :count]
            widths[batch_idx, :count] = (1.0 / valid_len[batch_idx].to(dtype=features.dtype)).clamp_min(1.0e-6)
            gates[batch_idx, :count] = 1.0

        return {
            "schema_version": "pc_ot_mras_eval_override_reader_outputs_v0",
            "override_mode": mode,
            "allocation": allocation,
            "acquisition_matrix": allocation,
            "valid_mask": valid,
            "selected_mask": selected_mask,
            "selected_times": selected_times,
            "centers": centers,
            "widths": widths,
            "gates": gates,
            "time_coords": time_coords,
        }

    @staticmethod
    def _pc_ot_mras_dense_time_coords(valid_mask, *, dtype):
        batch, time = valid_mask.shape
        valid_len = valid_mask.long().sum(dim=1).clamp(min=1)
        pos = torch.arange(time, device=valid_mask.device, dtype=dtype)[None, :].expand(batch, -1)
        denom = (valid_len - 1).clamp(min=1).to(dtype=dtype)[:, None]
        coords = pos / denom
        return coords.masked_fill(~valid_mask, 0.0)

    @staticmethod
    def _pc_ot_mras_exact_uniform_positions(valid_len, count):
        if count <= 0:
            raise ValueError("count must be positive")
        if count == 1:
            return torch.zeros((1,), dtype=torch.long, device=valid_len.device)
        stop = valid_len.to(dtype=torch.float32) - 1.0
        return torch.linspace(0.0, float(stop.item()), steps=count, device=valid_len.device).round().long()

    def _pc_ot_mras_reader_feature_and_mask(self, feat_list, mask_list):
        if not isinstance(feat_list, (tuple, list)) or not isinstance(mask_list, (tuple, list)):
            raise ValueError("PC-OT-MRAS reader hook expects projection outputs as feature/mask tuples")
        level = self.pc_ot_mras_reader_feature_level
        if level < 0 or level >= len(feat_list) or level >= len(mask_list):
            raise ValueError("PC-OT-MRAS reader feature level is out of range")
        features = feat_list[level]
        masks = mask_list[level]
        if not torch.is_tensor(features):
            raise ValueError("PC-OT-MRAS reader feature level must be a tensor")
        if not torch.is_tensor(masks):
            raise ValueError("PC-OT-MRAS reader mask level must be a tensor")
        if features.ndim != 3:
            raise ValueError(f"PC-OT-MRAS reader feature must be [B,C,T], got {tuple(features.shape)}")
        if masks.ndim != 2:
            raise ValueError(f"PC-OT-MRAS reader mask must be [B,T], got {tuple(masks.shape)}")
        if features.device != masks.device:
            raise ValueError("PC-OT-MRAS reader feature and mask must be on the same device")
        if torch.is_complex(features):
            raise ValueError("PC-OT-MRAS reader feature must be real-valued")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("PC-OT-MRAS reader feature must be finite")
        return features, masks

    @staticmethod
    def _clone_pc_ot_mras_metas_for_writer(metas, *, batch_size):
        if metas is None:
            return [{} for _ in range(batch_size)]
        if not isinstance(metas, (list, tuple)):
            raise ValueError("PC-OT-MRAS reader hook expects metas as a list/tuple")
        if len(metas) != int(batch_size):
            raise ValueError("PC-OT-MRAS reader hook metas length must match batch size")
        out = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"PC-OT-MRAS reader hook metas[{idx}] must be a mapping")
            if _PC_OT_MRAS_READER_OUTPUTS_META_KEY in meta:
                raise ValueError(
                    f"metas[{idx}] already contains '{_PC_OT_MRAS_READER_OUTPUTS_META_KEY}'; "
                    "refusing external reader-output injection"
                )
            out.append(dict(meta))
        return out

    @staticmethod
    def _normalize_pc_ot_mras_reader_aux_loss(config):
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("pc_ot_mras_reader_aux_loss must be a mapping when provided")
        config = dict(config)
        enabled = bool(config.pop("enabled", False))
        if not enabled:
            return None
        allowed = {"weights", "boundary_sigma", "short_action_len", "adjacent_gap"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"unknown pc_ot_mras_reader_aux_loss keys: {unknown}")
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, Mapping):
                raise ValueError("pc_ot_mras_reader_aux_loss.weights must be a mapping")
            config["weights"] = dict(weights)
        return config

    @staticmethod
    def _normalize_pc_ot_mras_reader_value_loss(config):
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("pc_ot_mras_reader_value_loss must be a mapping when provided")
        config = dict(config)
        enabled = bool(config.pop("enabled", False))
        if not enabled:
            return None
        allowed = {
            "weights",
            "require_targets",
            "target_source",
            "allow_train_gt",
            "allow_teacher_targets",
            "pair_temperature",
        }
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"unknown pc_ot_mras_reader_value_loss keys: {unknown}")
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, Mapping):
                raise ValueError("pc_ot_mras_reader_value_loss.weights must be a mapping")
            config["weights"] = dict(weights)
        return config

    def _normalize_pc_ot_mras_reader_eval_override(self, config):
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("pc_ot_mras_reader_eval_override must be a mapping when provided")
        config = dict(config)
        enabled = bool(config.pop("enabled", False))
        if not enabled:
            return None
        allowed = {"mode", "num_slots"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"unknown pc_ot_mras_reader_eval_override keys: {unknown}")
        mode = str(config.get("mode", "exact_uniform"))
        if mode != "exact_uniform":
            raise ValueError("pc_ot_mras_reader_eval_override.mode must be 'exact_uniform'")
        num_slots = config.get("num_slots")
        if num_slots is None:
            if self.pc_ot_mras_reader is None or not hasattr(self.pc_ot_mras_reader, "cfg"):
                raise ValueError("pc_ot_mras_reader_eval_override.num_slots is required when reader cfg is unavailable")
            num_slots = self.pc_ot_mras_reader.cfg.num_slots
        num_slots = int(num_slots)
        if num_slots <= 0:
            raise ValueError("pc_ot_mras_reader_eval_override.num_slots must be positive")
        return {"mode": mode, "num_slots": num_slots}

    @staticmethod
    def _normalize_pc_ot_mras_reader_soft_hard_loss(config):
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("pc_ot_mras_reader_soft_hard_loss must be a mapping when provided")
        config = dict(config)
        enabled = bool(config.pop("enabled", False))
        if not enabled:
            return None
        allowed = {"weights", "eps"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"unknown pc_ot_mras_reader_soft_hard_loss keys: {unknown}")
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, Mapping):
                raise ValueError("pc_ot_mras_reader_soft_hard_loss.weights must be a mapping")
            config["weights"] = dict(weights)
        return config

    def _pc_ot_mras_reader_auxiliary_losses(self, metas, gt_segments):
        if self.pc_ot_mras_reader_aux_loss is None:
            return {}
        reader_outputs = self._pc_ot_mras_reader_outputs_from_metas(metas)
        from ..losses.pc_ot_mras_auxiliary_losses import pc_ot_mras_auxiliary_losses

        return pc_ot_mras_auxiliary_losses(
            reader_outputs,
            gt_segments,
            **self.pc_ot_mras_reader_aux_loss,
        )

    def _pc_ot_mras_reader_value_losses(self, metas):
        if self.pc_ot_mras_reader_value_loss is None:
            return {}
        reader_outputs = self._pc_ot_mras_reader_outputs_from_metas(metas)
        from ..losses.pc_ot_mras_value_distillation_losses import pc_ot_mras_value_distillation_losses

        return pc_ot_mras_value_distillation_losses(
            reader_outputs,
            metas,
            **self.pc_ot_mras_reader_value_loss,
        )

    def _pc_ot_mras_reader_soft_hard_losses(self, metas):
        if self.pc_ot_mras_reader_soft_hard_loss is None:
            return {}
        reader_outputs = self._pc_ot_mras_reader_outputs_from_metas(metas)
        from ..losses.pc_ot_mras_soft_hard_consistency_losses import pc_ot_mras_soft_hard_consistency_losses

        return pc_ot_mras_soft_hard_consistency_losses(
            reader_outputs,
            **self.pc_ot_mras_reader_soft_hard_loss,
        )

    @staticmethod
    def _merge_pc_ot_mras_aux_losses(losses, aux_losses):
        ActionFormer._merge_pc_ot_mras_extra_losses(
            losses,
            aux_losses,
            source_name="pc_ot_mras_reader_aux_loss",
        )

    @staticmethod
    def _merge_pc_ot_mras_extra_losses(losses, extra_losses, *, source_name):
        for key, value in extra_losses.items():
            if key == "cost":
                raise ValueError(f"{source_name} must not return a cost key")
            if key in losses:
                raise ValueError(f"{source_name} key collision: {key}")
            if not torch.is_tensor(value):
                raise ValueError(f"{source_name} value for {key} must be a tensor")
            if value.ndim != 0:
                raise ValueError(f"{source_name} value for {key} must be scalar")
            if torch.is_complex(value) or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{source_name} value for {key} must be finite and real-valued")
            losses[key] = value

    @staticmethod
    def _strip_pc_ot_mras_value_targets_from_metas(metas):
        if metas is None:
            return None
        if isinstance(metas, Mapping):
            return {key: value for key, value in metas.items() if key not in _PC_OT_MRAS_VALUE_TARGET_KEYS}
        if isinstance(metas, list):
            out = []
            for idx, meta in enumerate(metas):
                if not isinstance(meta, Mapping):
                    raise ValueError(f"metas[{idx}] must be a mapping")
                out.append({key: value for key, value in meta.items() if key not in _PC_OT_MRAS_VALUE_TARGET_KEYS})
            return out
        if isinstance(metas, tuple):
            return tuple(ActionFormer._strip_pc_ot_mras_value_targets_from_metas(list(metas)))
        raise ValueError("metas must be a mapping/list/tuple or None")

    @staticmethod
    def _reject_pc_ot_mras_value_targets_in_forward_test(metas):
        if metas is None:
            return
        holders = [metas] if isinstance(metas, Mapping) else metas
        if not isinstance(holders, (list, tuple)):
            raise ValueError("metas must be a mapping/list/tuple or None")
        for idx, meta in enumerate(holders):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            present = [key for key in _PC_OT_MRAS_VALUE_TARGET_KEYS if key in meta]
            if present:
                raise ValueError(f"forward_test forbids train-only PC-OT-MRAS value targets: {present}")

    @staticmethod
    def _pc_ot_mras_reader_outputs_from_metas(metas):
        if not isinstance(metas, (list, tuple)) or len(metas) == 0:
            raise ValueError("pc_ot_mras_reader_aux_loss expects non-empty metas after reader injection")
        first = metas[0]
        if not isinstance(first, Mapping) or _PC_OT_MRAS_READER_OUTPUTS_META_KEY not in first:
            raise ValueError("pc_ot_mras_reader_aux_loss requires injected PC-OT-MRAS reader outputs")
        reader_outputs = first[_PC_OT_MRAS_READER_OUTPUTS_META_KEY]
        if not isinstance(reader_outputs, Mapping):
            raise ValueError("injected PC-OT-MRAS reader outputs must be a mapping")
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping for PC-OT-MRAS aux loss")
            if meta.get(_PC_OT_MRAS_READER_OUTPUTS_META_KEY) is not reader_outputs:
                raise ValueError("PC-OT-MRAS aux loss expects one shared batch reader_outputs mapping")
        return reader_outputs
