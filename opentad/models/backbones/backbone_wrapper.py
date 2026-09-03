import copy
import inspect
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

    def forward(self, frames, masks=None, irregular_selected_positions=None,
                irregular_dense_valid_len=None, **kwargs):
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
        actual_positions = canonical_positions = None
        if irregular_selected_positions is not None:
            from opentad.models.utils.temporal_grid import global_rank_clip_coordinates
            coords = global_rank_clip_coordinates(irregular_selected_positions, irregular_dense_valid_len, k=irregular_selected_positions.shape[1], clip_len=16, tubelet_size=2)
            actual_positions = coords["actual"].flatten(0, 1)
            canonical_positions = coords["canonical"].flatten(0, 1)
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        # prepare signature-aware backbone kwargs
        bb_kwargs = dict(kwargs)
        sig = inspect.signature(self.model.backbone.forward)
        params = sig.parameters
        if "actual_positions" in params:
            bb_kwargs["actual_positions"] = actual_positions
        if "canonical_positions" in params:
            bb_kwargs["canonical_positions"] = canonical_positions

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                        actual_positions, canonical_positions,
                        **kwargs,
                    )
                else:
                    features = self.model.backbone(frames, **bb_kwargs)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                    actual_positions, canonical_positions,
                    **kwargs,
                )
            else:
                features = self.model.backbone(frames, **bb_kwargs)


        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        B_feat = features.shape[0]
        self.latest_support_metadata = getattr(self.model.backbone, "latest_support_metadata", None)
        if self.latest_support_metadata is not None:
            unflattened_meta = {}
            for k, v in self.latest_support_metadata.items():
                if v is not None and v.ndim >= 2:
                    if v.ndim == 2:
                        unflattened_meta[k] = v.view(B_feat, -1)
                    elif v.ndim == 3:
                        unflattened_meta[k] = v.view(B_feat, -1, v.shape[-1])
            self.latest_support_metadata = unflattened_meta

        # apply mask
        if masks is not None and features.dim() == 3:
            # If features temporal dimension was reduced by Token Merging, adjust masks to match features length
            if masks.shape[1] != features.shape[-1] or masks.shape[0] != B_feat:
                if masks.shape[0] == B_feat:
                    ratio = float(features.shape[-1]) / float(masks.shape[1])
                    valid_lens = (masks.long().sum(dim=-1).float() * ratio).round().long()
                    new_masks = torch.zeros((B_feat, features.shape[-1]), dtype=masks.dtype, device=masks.device)
                    for b_idx in range(B_feat):
                        new_masks[b_idx, :min(features.shape[-1], valid_lens[b_idx].item())] = True
                    masks = new_masks
                else:
                    masks = torch.ones((B_feat, features.shape[-1]), dtype=masks.dtype, device=masks.device)
            features = features * masks.unsqueeze(1).detach().float()


        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

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

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim, actual_positions=None, canonical_positions=None, **kwargs):
        """Temporal Checkpointing for Video Backbone."""
        sig = inspect.signature(self.model.backbone.forward)
        params = sig.parameters

        def _inner_forward(frames, actual_positions=None, canonical_positions=None):
            inner_kwargs = dict(kwargs)
            if "actual_positions" in params:
                inner_kwargs["actual_positions"] = actual_positions
            if "canonical_positions" in params:
                inner_kwargs["canonical_positions"] = canonical_positions
            return self.model.backbone(frames, **inner_kwargs)

        video_feat = []
        for chunk_index, mini_frames in enumerate(torch.chunk(frames, chunk_num, dim=chunk_dim)):
            mini_actual = mini_canonical = None
            if actual_positions is not None and chunk_dim == 0:
                start = sum(x.shape[0] for x in torch.chunk(frames, chunk_num, dim=chunk_dim)[:chunk_index])
                mini_actual = actual_positions[start:start + mini_frames.shape[0]]
                mini_canonical = canonical_positions[start:start + mini_frames.shape[0]]
            mini_feat = cp.checkpoint(
                _inner_forward,
                mini_frames, mini_actual, mini_canonical,
                use_reentrant=False,
            )
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [torch.cat([f[idx] for f in video_feat], dim=chunk_dim) for idx in range(len(video_feat[0]))]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
