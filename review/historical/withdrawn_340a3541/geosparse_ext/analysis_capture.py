"""Read-only observers: no model imports, so a frozen training snapshot can be used."""
import base64
from collections import Counter
import math

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def component(name):
    if "scout" in name:
        return "scout_router"
    if "patch_embed" in name:
        return "patch_embedding"
    if ".adapter" in name or "regular_tia" in name:
        return "tia"
    if "encoder.coarse" in name:
        return "coarse_projection"
    if "encoder" in name:
        return "heavy_encoder"
    if "receiver" in name or "query_projection" in name:
        return "query_receiver"
    if name.startswith(("projection", "neck", "rpn_head")):
        return "detection_head"
    return "other"


class OperationMacCounter(TorchDispatchMode):
    """Executed Conv/Linear/Matmul MACs; activation/normalization/interpolation excluded.

    Profiling overhead makes this forward unsuitable for latency measurement.
    Fused attention without an exposed matmul is explicitly reported unsupported.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __enter__(self):
        self.stack, self.handles = [], []
        self.macs, self.ops, self.unsupported = Counter(), Counter(), Counter()
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_pre_hook(lambda module, args, name=name: self.stack.append(name)))
            self.handles.append(module.register_forward_hook(self._leave))
        return super().__enter__()

    def _leave(self, module, args, output):
        self.stack.pop()
        # Returning a value from a PyTorch forward hook replaces the output.
        return None

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        return super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = str(func).split(".")[1]
        self.ops[op] += 1
        count = 0
        if op in {"mm", "bmm", "matmul", "mv", "dot"}:
            count = out.numel() * args[0].shape[-1]
        elif op in {"addmm", "baddbmm"}:
            count = out.numel() * args[1].shape[-1]
        elif op == "linear":
            count = out.numel() * args[0].shape[-1]
        elif op in {"convolution", "_convolution"}:
            if args[6]:
                self.unsupported[op + ":transposed"] += 1
            else:
                count = out.numel() * math.prod(args[1].shape[1:])
        elif "attention" in op or "einsum" in op or "convolution" in op:
            self.unsupported[op] += 1
        if count:
            self.macs[component(self.stack[-1] if self.stack else "")] += int(count)
        return out

    def report(self):
        return dict(macs=sum(self.macs.values()), components=dict(self.macs), unsupported=dict(self.unsupported),
                    operator_calls=dict(self.ops), complete_for_conv_linear_matmul=not self.unsupported,
                    scope="executed Conv/Linear/Matmul MACs; excludes normalization, softmax, activations, interpolation and preprocessing; 1 MAC = 2 arithmetic FLOPs")


def cpu(value):
    return value.detach().cpu().numpy()


class SelectionObserver:
    def __init__(self, model):
        self.model = model
        self.payload, self.evidence = None, None
        self.handles = [model.geosparse.register_forward_hook(self.capture)]
        if hasattr(model.geosparse, "receiver"):
            self.handles.append(model.geosparse.receiver.register_forward_pre_hook(self.capture_evidence))

    def capture(self, module, args, output):
        batch = args[0]
        state, plan, baseline, cost, actionness = output
        if len(batch.video_id) != 1:
            raise ValueError("selection export uses one window per forward for exact cost attribution")
        self.payload = dict(
            selected_native=cpu(plan.selected_native[0]), selected_atoms=cpu(plan.selected_atoms[0]),
            atom_to_native=cpu(plan.atom_to_native), predicted_gain=cpu(plan.predicted_gain[0]),
            actionness=None if actionness is None else cpu(actionness[0]),
            sampling_order=cpu(plan.sampling_order[0]), execution_order=cpu(plan.execution_order[0]),
            requested_budget=float(plan.requested_budget[0]), cost=float(cost[0]), baseline=float(baseline[0]),
            source_frame_id=cpu(batch.source_frame_id[0]), intervals_s=cpu(batch.source_intervals_s[0]),
            regular_s=cpu(batch.target_time_s[0]), valid_frames=cpu(batch.valid_frames[0]),
            spatial_transform=cpu(batch.spatial_transform[0]), regular_query_s=cpu(state.regular_time_s[0]),
            video_id=batch.video_id[0], window_id=batch.window_id[0])

    def capture_evidence(self, module, args):
        evidence = args[2]
        self.evidence = {name: cpu(getattr(evidence, name)[0]) for name in
                         ("source_support", "support_valid", "physical_time_s", "roi_xyxy", "parent_clip_id", "valid", "fidelity")}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def record(self, job, counter, index):
        p = self.payload
        temporal = len(p["source_frame_id"]) // 2
        side = math.isqrt(p["selected_native"].size // temporal)
        selected = p["selected_native"].reshape(temporal, side, side)
        valid = p["valid_frames"].reshape(temporal, 2).any(-1)
        selected = selected & valid[:, None, None]
        encoder = self.model.geosparse.encoder
        trace = [] if encoder is None else [dict(row) for row in encoder.trace]
        width = (1024 if job["model"]["backbone"] == "videomae_l" else 768) if encoder is None else encoder.source.embed_dims
        depth = (24 if width == 1024 else 12) if encoder is None else len(encoder.source.blocks)
        heavy = sum(12 * row["qkv_tokens"] * width**2 + 2 * row["qkv_tokens"]**2 * width for row in trace)
        # The registered denominator remains full 224-resolution native Heavy,
        # including padded computation, even for the B-112 fidelity variant.
        native_n = 8 * 14 * 14
        reference = depth * (temporal // 8) * (12 * native_n * width**2 + 2 * native_n**2 * width)
        arrays = {key: value for key, value in p.items() if isinstance(value, np.ndarray)}
        if self.evidence is not None:
            arrays.update({"evidence_" + key: value for key, value in self.evidence.items()})
        counts = selected.sum((1, 2))
        summary = dict(window_index=index, video_id=p["video_id"], window_id=p["window_id"], route=job["route"],
            selected_semantics="fine refinement; coarse Heavy remains" if job["route"] == "C" else "Heavy evidence" if job["route"] == "B" else "Heavy updates",
            requested_budget=p["requested_budget"], selected_native_members=int(counts.sum()), valid_native_members=int(valid.sum() * side * side),
            selected_native_ratio=float(counts.sum() / max(1, valid.sum() * side * side)),
            selected_temporal_ratio=float(((counts > 0) & valid).sum() / max(1, valid.sum())),
            native_shape=[temporal, side, side], temporal_selected_fraction=(counts / (side * side)).tolist(),
            heavy_macs=heavy, heavy_reference_macs=reference, heavy_mac_ratio=heavy / reference,
            model_macs_counted=counter["macs"], mac_components=counter["components"], mac_scope=counter["scope"],
            mac_count_complete=counter["complete_for_conv_linear_matmul"], unsupported_ops=counter["unsupported"],
            parent_heavy_tokens=[sum(row["qkv_tokens"] for row in trace if row["parent"] == parent) for parent in range(temporal // 8)],
            zero_heavy_parent_fraction=sum(not any(row["qkv_tokens"] for row in trace if row["parent"] == parent) for parent in range(temporal // 8)) / (temporal // 8),
            nominal_time_s=p["regular_s"].reshape(temporal, 2).mean(-1).tolist(), valid_tubelets=valid.tolist(),
            support_valid=p["valid_frames"].reshape(temporal, 2).tolist(),
            source_frame_id=p["source_frame_id"].reshape(temporal, 2).tolist(), support_intervals_s=p["intervals_s"].reshape(temporal, 2, 2).tolist(),
            spatial_transform=p["spatial_transform"].tolist(), selected_bits=base64.b64encode(np.packbits(selected, bitorder="little").tobytes()).decode(),
            layers=trace)
        return summary, arrays
