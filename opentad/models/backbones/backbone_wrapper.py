import copy
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

        dynamic_cfg = getattr(custom_cfg, "dynamic_sparse_temporal", None)
        self.dynamic_sparse_temporal = dict(dynamic_cfg) if dynamic_cfg is not None else None
        if self.dynamic_sparse_temporal is not None and not bool(self.dynamic_sparse_temporal.get("enabled", False)):
            self.dynamic_sparse_temporal = None

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

        if self.dynamic_sparse_temporal is not None and self._dynamic_sparse_requested(metas):
            return self._forward_dynamic_sparse_temporal(frames, masks=masks, metas=metas)

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self.model.backbone(frames)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames)

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

    @staticmethod
    def _dynamic_sparse_requested(metas):
        return bool(
            isinstance(metas, (list, tuple))
            and metas
            and all(bool(meta.get("duca_sparse_variable_compute", False)) for meta in metas)
        )

    def _run_backbone(self, frames):
        if self.freeze_backbone:
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    return self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                return self.model.backbone(frames)
        if self.use_temporal_checkpointing:
            return self.temporal_checkpointing(
                frames,
                self.temporal_checkpointing_chunk_num,
                self.temporal_checkpointing_chunk_dim,
            )
        return self.model.backbone(frames)

    def _forward_dynamic_sparse_temporal(self, frames, *, masks, metas):
        cfg = self.dynamic_sparse_temporal
        clip_len = int(cfg.get("clip_len", 16))
        tubelet_size = int(cfg.get("tubelet_size", 2))
        output_len = int(cfg.get("output_len", 768))
        if frames.ndim != 6:
            raise ValueError("dynamic sparse temporal backbone expects [B,N,C,K,H,W]")
        batch, num_views, channels, selected_len, height, width = frames.shape
        if batch != 1:
            raise ValueError("dynamic sparse temporal backbone currently requires batch_size=1")
        if selected_len <= 0 or selected_len % clip_len != 0:
            raise ValueError("selected frame count must be positive and divisible by clip_len")
        if clip_len % tubelet_size != 0:
            raise ValueError("clip_len must be divisible by tubelet_size")
        if masks is None or tuple(masks.shape) != (batch, output_len):
            raise ValueError("dynamic sparse temporal backbone requires the original dense mask")

        chunks = selected_len // clip_len
        packed = frames.reshape(batch, num_views, channels, chunks, clip_len, height, width)
        packed = packed.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            batch * chunks * num_views,
            channels,
            clip_len,
            height,
            width,
        )
        features = self._run_backbone(packed)
        if isinstance(features, (tuple, list)):
            features = torch.cat(
                [self._pool_dynamic_sparse_feature(item, batch, chunks, num_views) for item in features],
                dim=1,
            )
        else:
            features = self._pool_dynamic_sparse_feature(features, batch, chunks, num_views)

        positions = torch.as_tensor(
            metas[0].get("duca_sparse_physical_positions", []),
            device=features.device,
            dtype=features.dtype,
        )
        if positions.numel() != selected_len:
            raise ValueError("selected physical-position metadata must match selected frame count")
        if bool((positions[1:] <= positions[:-1]).any().item()):
            raise ValueError("selected physical positions must be strictly increasing")
        token_positions = positions.reshape(-1, tubelet_size).mean(dim=1)
        if int(token_positions.numel()) != int(features.shape[-1]):
            raise ValueError("backbone temporal features do not match selected tubelet positions")
        features = self._interpolate_irregular_time(features, token_positions, output_len)
        features = features * masks.unsqueeze(1).detach().float()
        metas[0]["duca_dynamic_b_backbone_input_frames"] = int(selected_len)
        metas[0]["duca_dynamic_b_backbone_dense_equivalent_frames"] = int(output_len)
        metas[0]["duca_dynamic_b_true_compute_reduction"] = bool(selected_len < output_len)
        metas[0]["duca_dynamic_b_feature_reconstruction"] = "linear_physical_time_to_dense_axis"
        return features.to(torch.float32)

    @staticmethod
    def _pool_dynamic_sparse_feature(features, batch, chunks, num_views):
        if features.ndim != 5:
            raise ValueError("dynamic sparse VideoMAE features must be [B,C,T,H,W]")
        channels, tubelets, feat_h, feat_w = features.shape[1:]
        features = features.reshape(batch, chunks, num_views, channels, tubelets, feat_h, feat_w)
        features = features.mean(dim=(2, 5, 6))
        return features.permute(0, 2, 1, 3).reshape(batch, channels, chunks * tubelets)

    @staticmethod
    def _interpolate_irregular_time(features, positions, output_len):
        if features.ndim != 3 or positions.ndim != 1:
            raise ValueError("irregular interpolation expects [B,C,T] features and [T] positions")
        if int(features.shape[-1]) != int(positions.numel()):
            raise ValueError("feature and position lengths must match")
        target = torch.arange(output_len, device=features.device, dtype=features.dtype)
        right = torch.searchsorted(positions.contiguous(), target.contiguous())
        right = right.clamp(max=int(positions.numel()) - 1)
        left = (right - 1).clamp(min=0)
        left_pos = positions[left]
        right_pos = positions[right]
        denom = (right_pos - left_pos).clamp_min(torch.finfo(features.dtype).eps)
        weight = torch.where(right == left, torch.zeros_like(target), (target - left_pos) / denom)
        weight = weight.clamp(0.0, 1.0).view(1, 1, -1)
        left_feat = features.index_select(-1, left)
        right_feat = features.index_select(-1, right)
        return left_feat + (right_feat - left_feat) * weight

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
