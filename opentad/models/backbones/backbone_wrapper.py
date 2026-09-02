import copy
import os
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
        pretrain_path = getattr(custom_cfg, "pretrain", None)
        self.pretrain_allowed_missing_prefixes = tuple(
            getattr(custom_cfg, "pretrain_allowed_missing_prefixes", ("jacobian_approx", "adapter"))
        )
        self.pretrain_report = None
        if pretrain_path is not None:
            if not isinstance(pretrain_path, str) or not os.path.exists(pretrain_path):
                if bool(getattr(custom_cfg, "pretrain_required", False)):
                    raise FileNotFoundError(
                        f"required pretrained checkpoint does not exist: {pretrain_path}"
                    )
                print(f"Warning: pretrained checkpoint not found, using random initialization: {pretrain_path}")
            else:
                checkpoint = load_checkpoint(self.model, pretrain_path, map_location="cpu")
                self._record_taylor_pretrain_report(checkpoint, pretrain_path)
        elif bool(getattr(custom_cfg, "pretrain_required", False)):
            raise FileNotFoundError("pretrain_required=True but custom.pretrain is missing")
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

    def forward(self, frames, masks=None):
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

    def _record_taylor_pretrain_report(self, checkpoint, pretrain_path):
        """Record exact load coverage for ET-TRC's split attention modules."""
        taylor_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "pretrained_qkv_remapped")
        ]
        if not taylor_modules:
            return

        source = checkpoint.get("state_dict", checkpoint)
        target = self.model.state_dict()
        mapped = {}
        consumed = set()
        embed_dims = {
            name: int(module.embed_dims)
            for name, module in self.model.named_modules()
            if hasattr(module, "pretrained_qkv_remapped")
        }
        for key, value in source.items():
            if key.endswith(".attn.qkv.weight"):
                prefix = key[: -len("qkv.weight")]
                dims = embed_dims.get(prefix[:-1])
                if dims is not None and getattr(value, "shape", None) is not None:
                    if int(value.shape[0]) == 3 * dims:
                        mapped[prefix + "q_proj.weight"] = value[:dims]
                        mapped[prefix + "k_proj.weight"] = value[dims : 2 * dims]
                        mapped[prefix + "v_proj.weight"] = value[2 * dims :]
                        consumed.add(key)
            elif key.endswith(".attn.q_bias") or key.endswith(".attn.v_bias"):
                suffix = "q_bias" if key.endswith("q_bias") else "v_bias"
                prefix = key[: -len(suffix)]
                dims = embed_dims.get(prefix[:-1])
                if dims is not None and tuple(getattr(value, "shape", ())) == (dims,):
                    if key.endswith("q_bias"):
                        mapped[prefix + "q_proj.bias"] = value
                        mapped[prefix + "k_proj.bias"] = torch.zeros_like(value)
                    else:
                        mapped[prefix + "v_proj.bias"] = value
                    consumed.add(key)
            else:
                mapped[key] = value
                consumed.add(key)

        loaded_keys = [
            key
            for key, value in target.items()
            if key in mapped and tuple(getattr(mapped[key], "shape", ())) == tuple(value.shape)
        ]
        missing_keys = [
            key
            for key, value in target.items()
            if key not in mapped or tuple(getattr(mapped[key], "shape", ())) != tuple(value.shape)
        ]
        unexpected_keys = [key for key in source if key not in consumed]
        undeclared_missing = [
            key
            for key in missing_keys
            if not any(token in key for token in self.pretrain_allowed_missing_prefixes)
        ]
        if undeclared_missing:
            raise RuntimeError(
                "ET-TRC pretrained load has undeclared missing backbone keys: "
                + ", ".join(undeclared_missing[:20])
            )
        frozen = sum(1 for parameter in self.model.parameters() if not parameter.requires_grad)
        trainable = sum(1 for parameter in self.model.parameters() if parameter.requires_grad)
        self.pretrain_report = {
            "path": os.path.abspath(os.path.expanduser(str(pretrain_path))),
            "loaded": int(len(loaded_keys)),
            "remapped": int(sum(bool(module.pretrained_qkv_remapped) for module in taylor_modules)),
            "loaded_keys": loaded_keys,
            "remapped_keys": sorted(set(mapped).intersection(target)),
            "missing": int(len(missing_keys)),
            "missing_keys": missing_keys,
            "unexpected": int(len(unexpected_keys)),
            "unexpected_keys": unexpected_keys,
            "new_random": int(len(missing_keys)),
            "randomly_initialized_keys": missing_keys,
            "undeclared_missing_keys": undeclared_missing,
            "frozen": int(frozen),
            "trainable": int(trainable),
        }
        print(
            "[ET-TRC PRETRAIN] "
            + " ".join(f"{key}={value}" for key, value in self.pretrain_report.items())
        )

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
