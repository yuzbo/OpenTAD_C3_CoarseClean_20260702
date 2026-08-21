import copy
from collections.abc import Mapping
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from mmengine.runner import load_checkpoint

BACKBONES = MM_BACKBONES


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print("freeze_backbone: {}, norm_eval: {}".format(self.freeze_backbone, self.norm_eval))

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_dim"
            ), "temporal_checkpointing_chunk_dim should be provided when using temporal checkpointing"
            self.temporal_checkpointing_chunk_num = custom_cfg.temporal_checkpointing_chunk_num
            self.temporal_checkpointing_chunk_dim = custom_cfg.temporal_checkpointing_chunk_dim

    def forward(self, frames, masks=None, metas=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        physical_inputs = self._physical_backbone_inputs(
            frames,
            masks=masks,
            metas=metas,
        )

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    if physical_inputs is not None:
                        raise ValueError("physical-time backbone forbids temporal checkpoint chunking")
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self._forward_backbone(frames, physical_inputs)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                if physical_inputs is not None:
                    raise ValueError("physical-time backbone forbids temporal checkpoint chunking")
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self._forward_backbone(frames, physical_inputs)

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def _forward_backbone(self, frames, physical_inputs):
        if physical_inputs is None:
            return self.model.backbone(frames)
        return self.model.backbone(
            frames,
            source_positions=physical_inputs["source_positions"],
            valid_mask=physical_inputs["valid_mask"],
        )

    def _physical_backbone_inputs(self, frames, *, masks, metas):
        backbone = self.model.backbone
        if not bool(getattr(backbone, "physical_time", False)):
            return None
        if not isinstance(metas, (list, tuple)) or len(metas) == 0:
            raise ValueError("physical-time backbone requires one metadata mapping per original sample")
        if masks is None or masks.ndim != 2 or int(masks.shape[0]) != len(metas):
            raise ValueError("physical-time backbone requires selected observation masks [B,K]")

        position_rows = []
        requested = []
        effective = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"physical-time metas[{idx}] must be a mapping")
            positions = meta.get("irregular_selected_positions")
            if positions is None:
                raise ValueError("physical-time backbone requires irregular_selected_positions")
            positions = torch.as_tensor(positions, device=frames.device, dtype=torch.float32).reshape(-1)
            active = masks[idx].to(device=frames.device, dtype=torch.bool)
            active_count = int(active.sum().item())
            if active_count <= 0:
                raise ValueError("physical-time backbone requires at least one valid selected observation")
            expected_prefix = torch.arange(
                int(active.numel()), device=frames.device
            ) < active_count
            if not torch.equal(active, expected_prefix):
                raise ValueError("physical-time selected masks must be valid-prefix masks")
            if int(positions.numel()) != active_count:
                raise ValueError(
                    "selected positions must exactly match the number of valid selected observations"
                )
            valid_positions = positions
            if valid_positions.numel() > 1 and not bool(
                torch.all(valid_positions[1:] > valid_positions[:-1]).item()
            ):
                raise ValueError("physical-time source positions must be strictly increasing before padding")
            padded_positions = torch.cat(
                (
                    valid_positions,
                    valid_positions[-1:].expand(int(active.numel()) - active_count),
                ),
                dim=0,
            )
            position_rows.append(padded_positions)
            requested.append(int(active.numel()))
            effective.append(active_count)

        positions = torch.stack(position_rows, dim=0)
        flattened_video_count = int(frames.shape[0])
        clip_length = int(frames.shape[2])
        if positions.numel() != flattened_video_count * clip_length:
            raise ValueError(
                "physical-time K/clip reshape mismatch: "
                f"positions={positions.numel()}, clips={flattened_video_count}, clip_length={clip_length}"
            )
        source_positions = positions.reshape(flattened_video_count, clip_length)
        valid_mask = masks.to(device=frames.device, dtype=torch.bool).reshape(flattened_video_count, clip_length)

        clips_per_sample = flattened_video_count // len(metas)
        executed_k = clips_per_sample * clip_length
        for idx, meta in enumerate(metas):
            meta["duca_heavy_requested_k"] = requested[idx]
            meta["duca_heavy_effective_k"] = effective[idx]
            meta["duca_heavy_executed_k"] = executed_k
            meta["duca_heavy_call_boundary"] = "VisionTransformerAdapter.pre_patch_embed"
            meta["duca_heavy_source_positions_consumed"] = True
        return {
            "source_positions": source_positions,
            "valid_mask": valid_mask,
        }

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames):
            return self.model.backbone(frames)

        video_feat = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=chunk_dim):  # B*N is chunked
            # we can use torch.cp.checkpoint to implement an efficient temporal checkpointing mechanism
            mini_feat = cp.checkpoint(
                _inner_forward,
                mini_frames,
                use_reentrant=False,
            )
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [torch.cat([f[idx] for f in video_feat], dim=chunk_dim) for idx in range(len(video_feat[0]))]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
