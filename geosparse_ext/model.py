"""End-to-end native encoder/query routes, sharing one cheap observation path."""
import torch
from dataclasses import replace
from torch import nn
from torch.nn import functional as F
from .geometry import native_layout, canonical_atoms, detector_validity
from .routing import Scout, atom_features, make_plan
from .sparse import NativeEncoder, heavy_macs
from .receivers import Receiver
from .evidence import spatial_slots
from .contracts import DetectionState


class GeoSparseModel(nn.Module):
    def __init__(self, source, config, implementation="bucket"):
        super().__init__()
        self.config = dict(config)
        self.route = config["route"]
        if self.route not in {"DENSE", "A", "B", "C", "COARSE", "DEPTH", "BCR"}:
            raise NotImplementedError(f"route {self.route} needs its registered baseline implementation")
        self.parent_frames = 16
        self.query_length = config["query_length"]
        layers = None
        if self.route == "DEPTH":
            depth = config["active_depth"]
            layers = [round(i * (len(source.blocks) - 1) / (depth - 1)) for i in range(depth)]
        self.encoder = None if self.route == "COARSE" else NativeEncoder(
            source, "DENSE" if self.route in {"DENSE", "DEPTH"} else "A" if self.route == "BCR" else self.route,
            implementation, config["coarse_variant"], layers, config.get("prefix_depth", 0))
        self.need_scout = self.route in {"B", "COARSE"} or config["selector"] not in {"none", "uniform", "random", "static"}
        if self.need_scout:
            self.scout = Scout(config["scout_width"], config["scout_resolution"], config["scout_temporal_stride"])
        if self.route in {"B", "COARSE"}:
            from opentad.models.backbones.vit_adapter import Adapter
            self.query_projection = nn.Linear(config["scout_width"], 256)
            if self.route == "B":
                if any(block.use_adapter for block in source.blocks):
                    raise ValueError("B heavy encoder must disable TIA; regular TIA follows its receiver")
                self.receiver = Receiver(source.embed_dims, 256, config["receiver"], config["receiver_layers"],
                                         fusion=config["fusion_variant"])
            self.regular_tia = Adapter(256, temporal_size=384)
        self.register_buffer("dual", torch.tensor(0.1))
        self.register_buffer("cost_ema", torch.tensor(float(config["budget"])))

    def forward(self, batch, epoch=0, forced_plan=None, generators=None, execution_rng=None):
        frames = batch.frames_hi
        b, c, t, h, w = frames.shape
        config = self.config
        heavy_frames = frames
        if config["source_resolution"] == 112:
            if self.route != "B" or config["axis"] != "T":
                raise ValueError("112 full-frame fidelity is registered for B temporal selection")
            images = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            images = F.interpolate(images, (112, 112), mode="bilinear", align_corners=False)
            heavy_frames = images.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        hh, hw = heavy_frames.shape[-2:]
        layout = native_layout(replace(batch, frames_hi=heavy_frames))
        hn, wn = hh // 16, hw // 16
        atoms = canonical_atoms(t // 2, hn, wn, temporal_atom=config["temporal_atom_tubelets"],
                                spatial_group=config["spatial_group"], axis=config["axis"], device=frames.device)
        scout_features = None
        if self.need_scout:
            scout_features, budget_logits, baseline = self.scout(frames, batch.valid_frames)
            pooled = atom_features(scout_features, atoms, (hn, wn))
            gain, actionness = self.scout.gain(pooled).squeeze(-1), self.scout.actionness(pooled).squeeze(-1)
        else:
            gain = frames.new_zeros(b, len(atoms))
            baseline, budget_logits, actionness = frames.new_zeros(b), None, None
        motion = None
        if config["selector"] == "motion":
            small = F.interpolate(frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                                  (config["scout_resolution"],) * 2, mode="bilinear", align_corners=False)
            gray = small.mean(1).reshape(b, t, *small.shape[-2:])
            differences = torch.cat((torch.zeros_like(gray[:, :1]), (gray[:, 1:] - gray[:, :-1]).abs()), 1)
            differences = differences.reshape(b, t // 2, 2, *gray.shape[-2:]).mean(2)
            motion = atom_features(differences[..., None], atoms, (hn, wn)).squeeze(-1)
        routing_config = dict(config, selector="none", budget=1.) if self.route == "DEPTH" else config
        plan = forced_plan if forced_plan is not None else make_plan(
            gain, atoms, layout.valid, layout.parent_tubelets, (hn, wn), routing_config, budget_logits,
            actionness, motion, epoch, self.training, generators)
        if execution_rng is not None:
            torch.set_rng_state(execution_rng[0])
            if execution_rng[1] is not None:
                torch.cuda.set_rng_state_all(execution_rng[1])
        self.execution_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if frames.is_cuda else None)
        if self.route == "COARSE":
            native_features = None
            cost = frames.new_zeros(b)
        else:
            parents = heavy_frames.reshape(b, c, t // 16, 16, hh, hw).permute(0, 2, 1, 3, 4, 5).reshape(-1, c, 16, hh, hw)
            valid_parents = batch.valid_frames.reshape(-1, 16)
            if self.route == "C":
                selection = plan.selected_atoms.reshape(-1, 8, hn // 2, wn // 2)
            else:
                selection = plan.selected_native.flatten(0, 1)
            if self.route == "DENSE" or (config["selector"] == "none" and config["budget"] == 1.):
                # The upstream full limit processes padded inputs as well and
                # applies its output mask after encoding; preserve that limit.
                valid_parents = None
                selection = torch.ones_like(selection)
            native = self.encoder(parents, selection, valid_parents, position=config["geometry"] != "no_position")
            native_features = native.reshape(b, t // 16, -1, 8, hn, wn).permute(0, 2, 1, 3, 4, 5).reshape(b, -1, t // 2, hn, wn)
            width = self.encoder.source.embed_dims
            depth = len(self.encoder.source.blocks)
            reference_tokens = 8 * hn * wn
            dense_macs = depth * (t // 16) * ((4 + 8) * reference_tokens * width ** 2 + 2 * reference_tokens ** 2 * width)
            cost = frames.new_tensor([heavy_macs([s for s in self.encoder.trace if s["parent"] // (t // 16) == i], width)
                                      / dense_macs for i in range(b)])
        if self.route in {"B", "COARSE"}:
            query = self.query_projection(scout_features.mean((2, 3)))
            if query.shape[1] != t // 2:
                raise ValueError("cheap query must retain the full native tubelet grid")
            if self.route == "B":
                evidence = spatial_slots(native_features, plan.selected_native.reshape(b, t // 2, hn, wn), layout, config["evidence_slots"], batch.spatial_transform)
                radius = (layout.nominal_time_s[:, 1:] - layout.nominal_time_s[:, :-1]).median(-1).values * 8
                query = self.receiver(query, layout.nominal_time_s, evidence, radius)
            # One native full-window TIA per example, before detector resampling.
            # Heavy self-attention remains independently partitioned into parents.
            self.regular_tia.temporal_size = t // 2
            query = self.regular_tia(query, 1, 1)
            features = F.interpolate(query.transpose(1, 2), self.query_length, mode="linear", align_corners=False)
        else:
            features = F.interpolate(native_features.mean((-1, -2)), self.query_length, mode="linear", align_corners=False)
        times = F.interpolate(layout.nominal_time_s[:, None], self.query_length, mode="linear", align_corners=False)[:, 0]
        valid = detector_validity(batch.valid_frames, self.query_length)
        features = features * valid[:, None]
        stride = (times[:, -1] - times[:, 0]) / max(1, self.query_length - 1)
        return DetectionState(features, times, valid, stride), plan, baseline, cost, actionness
