import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dynamic_temporal_bucket = bool(
            getattr(custom_cfg, "dynamic_temporal_bucket", False)
        )
        self.dynamic_temporal_clip_len = int(
            getattr(custom_cfg, "dynamic_temporal_clip_len", 16)
        )
        if self.dynamic_temporal_clip_len <= 0:
            raise ValueError("dynamic_temporal_clip_len must be positive")
        # Set on every real wrapper forward.  Paper-facing selectors consume
        # this receipt only after the heavy backbone has accepted the tensor;
        # selector-side planned K is not sufficient physical-cost evidence.
        self.last_forward_input_contract = None

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

    def forward(self, frames, masks=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        self.last_forward_input_contract = None
        original_shape = tuple(int(value) for value in frames.shape)
        original_batch = int(frames.shape[0])
        original_num_segs = int(frames.shape[1]) if frames.ndim == 6 else 1
        original_temporal_len = (
            int(frames.shape[3]) if frames.ndim == 6 else int(frames.shape[2])
        )

        # set all normalization layers
        self.set_norm_layer()

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.dynamic_temporal_bucket:
            frames, dynamic_shape = self._prepare_dynamic_temporal_bucket(
                frames,
                masks,
            )
        else:
            dynamic_shape = None
        if self.pre_processing_pipeline is not None and not self.dynamic_temporal_bucket:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]
        if self.dynamic_temporal_bucket:
            if frames.ndim != 5:
                raise RuntimeError(
                    "dynamic temporal backbone boundary must be [BN,C,T,H,W]"
                )
            inner_reconstructed_k = (
                int(frames.shape[0])
                * int(frames.shape[2])
                // (original_batch * original_num_segs)
            )
            mask_temporal_len = (
                int(masks.shape[1])
                if masks is not None and masks.ndim == 2
                else -1
            )
            all_mask_active = bool(
                masks is not None
                and masks.numel() > 0
                and masks.to(dtype=torch.bool).all().item()
            )
            if (
                original_num_segs != 1
                or original_temporal_len != mask_temporal_len
                or inner_reconstructed_k != original_temporal_len
                or not all_mask_active
            ):
                raise RuntimeError(
                    "dynamic heavy-backbone physical input accounting drift"
                )
            self.last_forward_input_contract = {
                "schema_version": "duca_dynamic_backbone_input_v1",
                "measurement_source": "actual_backbone_wrapper_and_videomae_input_tensors",
                "wrapper_input_shape": list(original_shape),
                "wrapper_temporal_k": original_temporal_len,
                "mask_temporal_k": mask_temporal_len,
                "all_mask_active": all_mask_active,
                "inner_backbone_input_shape": [int(value) for value in frames.shape],
                "inner_temporal_chunk_k": int(frames.shape[2]),
                "inner_reconstructed_k": inner_reconstructed_k,
                "num_segs": original_num_segs,
                "padding_or_repetition_observed": False,
            }

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
        if self.dynamic_temporal_bucket:
            if isinstance(features, (tuple, list)):
                features = torch.cat(
                    [
                        self.dynamic_unflatten_and_pool_features(f, dynamic_shape)
                        for f in features
                    ],
                    dim=1,
                )
            else:
                features = self.dynamic_unflatten_and_pool_features(
                    features,
                    dynamic_shape,
                )
        elif isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def _prepare_dynamic_temporal_bucket(self, frames, masks):
        if frames.ndim != 6:
            raise ValueError(
                "dynamic temporal bucket expects [B,N,C,K,H,W] pre-backbone RGB"
            )
        batch, num_segs, channels, temporal_len, height, width = frames.shape
        if temporal_len <= 0 or temporal_len % self.dynamic_temporal_clip_len != 0:
            raise ValueError(
                "dynamic RIME K must be positive and divisible by the heavy clip length"
            )
        if masks is None or tuple(masks.shape) != (batch, temporal_len):
            raise ValueError("dynamic RIME backbone requires an aligned [B,K] mask")
        active = masks.to(device=frames.device, dtype=torch.bool)
        if not bool(active.all().item()):
            raise ValueError(
                "dynamic RIME backbone forbids inactive tail padding inside a K bucket"
            )
        chunk_count = temporal_len // self.dynamic_temporal_clip_len
        frames = (
            frames.reshape(
                batch,
                num_segs,
                channels,
                chunk_count,
                self.dynamic_temporal_clip_len,
                height,
                width,
            )
            .permute(0, 3, 1, 2, 4, 5, 6)
            .reshape(
                batch * chunk_count,
                num_segs,
                channels,
                self.dynamic_temporal_clip_len,
                height,
                width,
            )
            .contiguous()
        )
        return frames, (batch, chunk_count, num_segs, temporal_len)

    @staticmethod
    def dynamic_unflatten_and_pool_features(features, dynamic_shape):
        batch, chunk_count, num_segs, temporal_len = dynamic_shape
        expected = int(batch * chunk_count * num_segs)
        if int(features.shape[0]) != expected:
            raise RuntimeError("dynamic RIME backbone feature batch does not match K chunks")
        if features.ndim == 5:
            _, channels, feature_time, _, _ = features.shape
            pooled = features.reshape(
                batch,
                chunk_count,
                num_segs,
                channels,
                feature_time,
                features.shape[-2],
                features.shape[-1],
            ).mean(dim=(2, 5, 6))
            pooled = pooled.permute(0, 2, 1, 3).reshape(
                batch,
                channels,
                chunk_count * feature_time,
            )
        elif features.ndim == 2:
            channels = int(features.shape[1])
            pooled = (
                features.reshape(batch, chunk_count, num_segs, channels)
                .mean(dim=2)
                .permute(0, 2, 1)
            )
        else:
            raise RuntimeError(
                "dynamic RIME backbone expects [BN,C,T,H,W] or [BN,C] features"
            )
        if int(pooled.shape[-1]) != int(temporal_len):
            pooled = F.interpolate(
                pooled,
                size=int(temporal_len),
                mode="linear",
                align_corners=False,
            )
        return pooled

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
