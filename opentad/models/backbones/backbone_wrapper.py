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
        self.strict_temporal_padding_mask = bool(
            getattr(custom_cfg, "strict_temporal_padding_mask", False)
        )
        self.latest_temporal_padding_mask_summary = None

        print("freeze_backbone: {}, norm_eval: {}".format(self.freeze_backbone, self.norm_eval))

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"

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

    def forward(self, frames, masks=None, boundary_prior=None, delta_t=None, **kwargs):
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
        original_batches = int(frames.shape[0])
        raw_masks = None
        if self.strict_temporal_padding_mask:
            if masks is None or masks.ndim != 2 or int(masks.shape[0]) != original_batches:
                raise ValueError("strict temporal padding isolation requires raw masks with shape [B,K]")
            raw_masks = masks.to(device=frames.device, dtype=torch.bool)

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]
        backbone_temporal_mask = None
        if self.strict_temporal_padding_mask:
            if self.use_temporal_checkpointing:
                raise ValueError("strict temporal padding isolation does not support wrapper checkpoint chunking")
            temporal_length = int(frames.shape[2])
            if batches % original_batches:
                raise ValueError("pre-processing changed the batch axis incompatibly with temporal masking")
            temporal_chunks = batches // original_batches
            if int(raw_masks.shape[1]) != temporal_chunks * temporal_length:
                raise ValueError("raw mask length does not match pre-processed temporal chunks")
            backbone_temporal_mask = raw_masks.reshape(
                original_batches, temporal_chunks, temporal_length
            ).reshape(batches, temporal_length)
            backbone_temporal_mask = (
                backbone_temporal_mask[:, None, :]
                .expand(batches, num_segs, temporal_length)
                .reshape(batches * num_segs, temporal_length)
            )
            frames = frames * backbone_temporal_mask[:, None, :, None, None].to(dtype=frames.dtype)

        backbone_kwargs = {}
        if self.strict_temporal_padding_mask:
            backbone_kwargs["temporal_mask"] = backbone_temporal_mask
        if boundary_prior is not None:
            if boundary_prior.ndim == 2 and boundary_prior.shape[0] == original_batches:
                temporal_chunks = max(1, batches // original_batches)
                raw_prior_len = int(boundary_prior.shape[1])
                tubelet_size = int(getattr(getattr(self.model, "backbone", self.model), "tubelet_size", 2))
                frames_per_chunk = raw_prior_len // temporal_chunks
                tubelets_per_chunk = max(1, frames_per_chunk // tubelet_size)
                bp = boundary_prior.reshape(original_batches, temporal_chunks, tubelets_per_chunk, tubelet_size).amax(dim=-1)
                bp = bp.reshape(batches, tubelets_per_chunk)
                if num_segs > 1:
                    bp = bp[:, None, :].expand(batches, num_segs, tubelets_per_chunk).reshape(batches * num_segs, tubelets_per_chunk)
                backbone_kwargs["boundary_prior"] = bp
            else:
                backbone_kwargs["boundary_prior"] = boundary_prior
        if delta_t is not None:
            if delta_t.ndim == 2 and delta_t.shape[0] == original_batches:
                temporal_chunks = max(1, batches // original_batches)
                raw_dt_len = int(delta_t.shape[1])
                tubelet_size = int(getattr(getattr(self.model, "backbone", self.model), "tubelet_size", 2))
                # Check if delta_t is already tubelet-level (e.g. 192 for 24 chunks of 8 tubelets)
                # or frame-level (e.g. 384 for 24 chunks of 16 frames)
                if raw_dt_len % (temporal_chunks * tubelet_size) == 0 and raw_dt_len // temporal_chunks == (getattr(self.model, "backbone", self.model).num_frames if hasattr(getattr(self.model, "backbone", self.model), "num_frames") else 16):
                    frames_per_chunk = raw_dt_len // temporal_chunks
                    tubelets_per_chunk = max(1, frames_per_chunk // tubelet_size)
                    dt = delta_t.reshape(original_batches, temporal_chunks, tubelets_per_chunk, tubelet_size).mean(dim=-1)
                else:
                    tubelets_per_chunk = max(1, raw_dt_len // temporal_chunks)
                    dt = delta_t.reshape(original_batches, temporal_chunks, tubelets_per_chunk)
                dt = dt.reshape(batches, tubelets_per_chunk)
                if num_segs > 1:
                    dt = dt[:, None, :].expand(batches, num_segs, tubelets_per_chunk).reshape(batches * num_segs, tubelets_per_chunk)
                backbone_kwargs["delta_t"] = dt
            else:
                backbone_kwargs["delta_t"] = delta_t

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
                    features = self.model.backbone(frames, **backbone_kwargs)
        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames, **backbone_kwargs)

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)
        if masks is not None and features.dim() == 3:
            if self.strict_temporal_padding_mask:
                feature_length = int(features.shape[-1])
                raw_length = int(raw_masks.shape[-1])
                if raw_length % feature_length:
                    raise ValueError("strict temporal padding mask cannot be reduced to backbone feature length")
                output_masks = raw_masks.reshape(
                    raw_masks.shape[0], feature_length, raw_length // feature_length
                ).any(dim=-1)
                features = features * output_masks.unsqueeze(1).to(dtype=features.dtype)
                model_summary = getattr(
                    self.model.backbone, "latest_temporal_padding_mask_summary", None
                )
                if not isinstance(model_summary, dict) or model_summary.get(
                    "strict_isolation_verified"
                ) is not True:
                    raise RuntimeError("video backbone did not verify strict temporal padding isolation")
                self.latest_temporal_padding_mask_summary = {
                    **model_summary,
                    "wrapper_enabled": True,
                    "output_invalid_features_zeroed": True,
                    "output_valid_counts": [int(row.sum().item()) for row in output_masks],
                }
            else:
                features = features * masks.unsqueeze(1).detach().float()
                self.latest_temporal_padding_mask_summary = {
                    "enabled": False,
                    "strict_isolation_verified": False,
                }

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
