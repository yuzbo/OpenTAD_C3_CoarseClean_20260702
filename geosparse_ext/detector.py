"""OpenTAD detection losses with current-model PG/acquisition training."""
from dataclasses import replace
import torch
from mmengine.config import ConfigDict
from opentad.models.builder import DETECTORS
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from .model import GeoSparseModel
from .data import video_batch
from .routing import policy_losses, acquisition_loss


def restore_buffers(module, values):
    # Replace tensors: in-place restore would invalidate saved autograd versions
    # from the primary loss (including the head's loss normalizer).
    for name, value in values.items():
        parent, _, leaf = name.rpartition(".")
        target = module.get_submodule(parent) if parent else module
        target._buffers[leaf] = value.clone()


def load_recognition_weights(source, path):
    checkpoint = torch.load(path, map_location="cpu")
    weights = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state = {name.removeprefix("backbone."): value for name, value in weights.items()
             if not name.startswith("cls_head.")}
    outcome = source.load_state_dict(state, strict=False)
    # Upstream recognition checkpoints omit the deterministic sinusoidal buffer.
    # VisionTransformerAdapter constructs this exact native PE in __init__.
    required = [key for key in outcome.missing_keys if ".adapter." not in key and key not in {"pos_embed", "_amod_successful_update"}]
    if required or outcome.unexpected_keys:
        raise ValueError(f"recognition checkpoint mismatch: missing={required}, unexpected={outcome.unexpected_keys}")


@DETECTORS.register_module()
class GeoSparseDetector(ActionFormer):
    def __init__(self, projection, rpn_head, neck, geosparse, recognition_checkpoint=None, source_config=None):
        super().__init__(projection=ConfigDict(projection), rpn_head=ConfigDict(rpn_head), neck=neck)
        config = dict(geosparse)
        large = config["backbone"] == "videomae_l"
        source_args = dict(img_size=224, patch_size=16, embed_dims=1024 if large else 768,
                           depth=24 if large else 12, num_heads=16 if large else 12,
                           num_frames=16, tubelet_size=2, total_frames=768,
                           adapter_index=[] if config["route"] == "B" else list(range(24 if large else 12)),
                           return_feat_map=True, with_cp=True, drop_path_rate=0.1)
        if source_config is not None:
            source_args.update(source_config)  # only explicit correctness-test configuration
        if config["route"] == "DEPTH":
            depth = config["active_depth"]
            if not 2 <= depth <= source_args["depth"]:
                raise ValueError("active depth exceeds the source backbone")
            source_args["adapter_index"] = [round(i * (source_args["depth"] - 1) / (depth - 1)) for i in range(depth)]
        source = None if config["route"] == "COARSE" else VisionTransformerAdapter(**source_args)
        if source is not None:
            if recognition_checkpoint is not None:
                load_recognition_weights(source, recognition_checkpoint)
            source._freeze_layers()
            if source.fc_norm is not None:
                source.fc_norm.requires_grad_(False)  # feature-map forward does not use it
        self.geosparse = GeoSparseModel(source, config)
        self.epoch = 0
        self.minibatch = 0
        self.latest_route_plan = None
        self.latest_acquisition = None
        self.pending_cost = []
        self.inference_seed = 0
        self.inference_video_index = {}

    def _task_losses(self, state, masks, metas, gt_segments, gt_labels, raw_length):
        scale = state.features.shape[-1] / raw_length
        segments = [x.to(state.features.device) * scale for x in gt_segments]
        labels = [x.to(state.features.device) for x in gt_labels]
        return super().forward_train(state.features.float(), state.valid, metas, segments, labels)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        batch = video_batch(inputs, masks, metas, next(self.parameters()).device)
        config = self.geosparse.config
        buffers_before = {n: v.detach().clone() for n, v in self.named_buffers()}
        normalizer_before = self.rpn_head.loss_normalizer
        state, plan, baseline, cost, actionness = self.geosparse(batch, self.epoch)
        execution_rng = self.geosparse.execution_rng
        primary_trace = list(self.geosparse.encoder.trace) if self.geosparse.encoder is not None else []
        losses = self._task_losses(state, masks, metas, gt_segments, gt_labels, batch.frames_hi.shape[2])
        task = losses.pop("cost")
        self.latest_route_plan = plan
        if bool(plan.learned_sample.any()):
            losses.update(policy_losses(task, baseline, plan, cost, config["budget"], self.geosparse.dual))
        if actionness is not None and (config["actionness_aux"] or config["selector"] in {"actionness", "uncertainty", "cdf_native"}):
            spatial = plan.selected_native.shape[-1] // 8
            centers = ((plan.atom_to_native.float() // spatial) * 2 + 0.5).mean(-1)
            target = torch.zeros_like(actionness)
            for b, segments in enumerate(gt_segments):
                for left, right in segments.to(actionness.device):
                    target[b] = torch.maximum(target[b], ((centers >= left) & (centers <= right)).to(target.dtype))
            losses["actionness_loss"] = torch.nn.functional.binary_cross_entropy_with_logits(actionness, target)
        self.latest_acquisition = None
        probe = config["estimator"] == "pg_acquisition" and self.epoch >= config["warmup_epochs"] and config["probe_every"] > 0 and self.minibatch % config["probe_every"] == 0
        if probe:
            flat_valid = batch.valid_frames.reshape(batch.frames_hi.shape[0], -1, 2).any(-1)
            spatial = plan.selected_native.shape[-1] // 8
            native_valid = flat_valid.repeat_interleave(spatial, 1)
            valid_atoms = native_valid[:, plan.atom_to_native].any(-1)
            available = torch.nonzero(valid_atoms & ~plan.selected_atoms, as_tuple=False)
            if len(available):
                row, atom = available[torch.randint(len(available), ())].tolist()
                selected_atoms = plan.selected_atoms.clone()
                selected_native = plan.selected_native.clone()
                selected_atoms[row, atom] = True
                members = plan.atom_to_native[atom]
                selected_native[row].flatten()[members] = native_valid[row, members]
                added = replace(plan, selected_atoms=selected_atoms, selected_native=selected_native,
                                execution_order=[torch.where(mask.flatten())[0] for mask in selected_native])
                buffers_after = {n: v.detach().clone() for n, v in self.named_buffers()}
                normalizer_after = self.rpn_head.loss_normalizer
                rng_after_cpu = torch.get_rng_state()
                rng_after_cuda = torch.cuda.get_rng_state_all() if batch.frames_hi.is_cuda else None
                try:
                    with torch.no_grad():
                        restore_buffers(self, buffers_before)
                        self.rpn_head.loss_normalizer = normalizer_before
                        acquired, _, _, extra_cost, _ = self.geosparse(batch, self.epoch, forced_plan=added, execution_rng=execution_rng)
                        alt = self._task_losses(acquired, masks, metas, gt_segments, gt_labels, batch.frames_hi.shape[2])["cost"]
                finally:
                    with torch.no_grad():
                        restore_buffers(self, buffers_after)
                        self.rpn_head.loss_normalizer = normalizer_after
                    torch.set_rng_state(rng_after_cpu)
                    if rng_after_cuda is not None:
                        torch.cuda.set_rng_state_all(rng_after_cuda)
                loss, target = acquisition_loss(plan.predicted_gain[row, atom], task, alt)
                losses["acquisition_loss"] = loss
                self.latest_acquisition = dict(kind="acquisition", row=row, atom=atom, signed_target=float(target), extra_heavy_cost=extra_cost.detach().cpu().tolist())
                self.geosparse.encoder.trace = primary_trace
        losses["cost"] = task + sum(value for name, value in losses.items() if name in {"actor_loss", "critic_loss", "actionness_loss", "acquisition_loss"})
        self.minibatch += 1
        if config["budget_mode"] == "dynamic" and self.training:
            self.pending_cost.append(cost.detach())
        return losses

    def after_optimizer_step(self):
        if self.pending_cost:
            with torch.no_grad():
                self.geosparse.cost_ema.lerp_(torch.cat(self.pending_cost).mean(), 0.1)
                self.geosparse.dual.add_(0.001 * (self.geosparse.cost_ema - self.geosparse.config["budget"])).clamp_(0, 10)
            self.pending_cost.clear()
        return None

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        batch = video_batch(inputs, masks, metas, next(self.parameters()).device)
        return self.forward_video_batch(batch, metas, infer_cfg, **kwargs)

    def forward_video_batch(self, batch, metas, infer_cfg=None, **kwargs):
        """Normalized device-resident inputs, also the device benchmark boundary."""
        generators = [torch.Generator(device=batch.frames_hi.device).manual_seed(
            (self.inference_seed * 1000000007 + self.inference_video_index.get(name, 0) * 1000003
             + int(batch.source_frame_id[i, 0])) % (2**63 - 1)) for i, name in enumerate(batch.video_id)]
        state, plan, _, _, _ = self.geosparse(batch, self.epoch, generators=generators)
        self.latest_route_plan = plan
        proposals, scores = super().forward_test(state.features.float(), state.valid, metas, infer_cfg, **kwargs)
        factor = batch.frames_hi.shape[2] / state.features.shape[-1]
        return [p * factor for p in proposals], scores
