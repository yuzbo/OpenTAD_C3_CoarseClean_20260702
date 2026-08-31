import copy
from collections.abc import Mapping

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
                irregular_dense_valid_len=None, duca_multibudget_context=None):
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

        if duca_multibudget_context is not None:
            context = self._validate_duca_multibudget_context(
                duca_multibudget_context,
                batch_size=int(frames.shape[0]),
                device=frames.device,
            )
            if self._duca_multibudget_uses_legacy_path(frames, context):
                self.last_execution_profile = self._duca_execution_profile(
                    context, used_legacy_k384_path=True
                )
            else:
                return self._forward_duca_multibudget(
                    frames,
                    masks=masks,
                    context=context,
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

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                        actual_positions, canonical_positions,
                    )
                else:
                    features = self.model.backbone(frames, actual_positions=actual_positions, canonical_positions=canonical_positions)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                    actual_positions, canonical_positions,
                )
            else:
                features = self.model.backbone(frames, actual_positions=actual_positions, canonical_positions=canonical_positions)

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
    def _validate_duca_multibudget_context(context, *, batch_size, device):
        if not isinstance(context, Mapping):
            raise ValueError("duca_multibudget_context must be a mapping")
        required = {
            "requested_budget",
            "effective_budget",
            "actual_observations",
            "execution_slots",
            "collapsed_to_baseline",
            "detector_length",
            "packet_size",
        }
        missing = sorted(required - set(context))
        if missing:
            raise ValueError(f"duca_multibudget_context is missing {missing}")
        out = dict(context)
        for key in (
            "requested_budget",
            "effective_budget",
            "actual_observations",
            "execution_slots",
        ):
            value = torch.as_tensor(out[key], device=device, dtype=torch.long).reshape(-1)
            if value.shape != (batch_size,):
                raise ValueError(f"duca_multibudget_context.{key} must be [B]")
            out[key] = value
        collapsed = torch.as_tensor(
            out["collapsed_to_baseline"], device=device, dtype=torch.bool
        ).reshape(-1)
        if collapsed.shape != (batch_size,):
            raise ValueError("duca_multibudget_context.collapsed_to_baseline must be [B]")
        out["collapsed_to_baseline"] = collapsed
        out["detector_length"] = int(out["detector_length"])
        out["packet_size"] = int(out["packet_size"])
        if out["detector_length"] != 384 or out["packet_size"] != 16:
            raise ValueError("DUCA exposure keeps detector length 384 and packet size 16")
        if torch.any(out["actual_observations"] <= 0):
            raise ValueError("actual heavy observations must be positive")
        if torch.any(out["execution_slots"] < out["actual_observations"]):
            raise ValueError("execution slots cannot omit an actual observation")
        if torch.any(out["execution_slots"] % out["packet_size"] != 0):
            raise ValueError("heavy execution slots must be packet aligned")
        if torch.any(
            ~torch.isin(
                out["requested_budget"],
                torch.tensor((256, 384, 512), device=device, dtype=torch.long),
            )
        ):
            raise ValueError("requested exposure budget must be 256, 384, or 512")
        return out

    @staticmethod
    def _duca_multibudget_uses_legacy_path(frames, context):
        temporal_dim = 3 if frames.ndim == 6 else 2 if frames.ndim == 5 else None
        if temporal_dim is None:
            raise ValueError("DUCA heavy input must be [B,N,C,T,H,W] or [B,C,T,H,W]")
        return (
            int(frames.shape[temporal_dim]) == 384
            and bool((context["effective_budget"] == 384).all().item())
            and bool((context["execution_slots"] == 384).all().item())
        )

    @staticmethod
    def _duca_execution_profile(context, *, used_legacy_k384_path):
        return {
            "requested_budget": [
                int(value) for value in context["requested_budget"].detach().cpu().tolist()
            ],
            "effective_budget": [
                int(value) for value in context["effective_budget"].detach().cpu().tolist()
            ],
            "actual_observations": [
                int(value) for value in context["actual_observations"].detach().cpu().tolist()
            ],
            "execution_slots": [
                int(value) for value in context["execution_slots"].detach().cpu().tolist()
            ],
            "total_actual_observations": int(
                context["actual_observations"].sum().detach().cpu().item()
            ),
            "total_execution_slots": int(
                context["execution_slots"].sum().detach().cpu().item()
            ),
            "packet_size": int(context["packet_size"]),
            "detector_length": int(context["detector_length"]),
            "used_legacy_k384_path": bool(used_legacy_k384_path),
            "counts_real_videomae_inputs_not_detector_padding": True,
        }

    def _forward_duca_multibudget(self, frames, *, masks, context):
        if self.use_temporal_checkpointing:
            raise ValueError("DUCA variable-length exposure uses the frozen non-checkpointed H65 backbone")
        if frames.ndim == 5:
            frames = frames[:, None, ...]
        if frames.ndim != 6:
            raise ValueError("DUCA variable-length heavy input must be [B,N,C,T,H,W]")
        max_available = int(frames.shape[3])
        if torch.any(context["execution_slots"] > max_available):
            raise ValueError("selector output does not contain every requested execution slot")

        row_features = [None] * int(frames.shape[0])
        unique_execution = torch.unique(context["execution_slots"], sorted=True)
        for execution_tensor in unique_execution:
            execution_slots = int(execution_tensor.item())
            rows = torch.nonzero(
                context["execution_slots"] == execution_slots, as_tuple=False
            ).flatten()
            group = frames.index_select(0, rows)[:, :, :, :execution_slots]
            packet_size = int(context["packet_size"])
            packet_count = execution_slots // packet_size
            batch, num_segs, channels, _, height, width = group.shape
            group = (
                group.reshape(
                    batch,
                    num_segs,
                    channels,
                    packet_count,
                    packet_size,
                    height,
                    width,
                )
                .permute(0, 3, 1, 2, 4, 5, 6)
                .reshape(
                    batch * packet_count,
                    num_segs,
                    channels,
                    packet_size,
                    height,
                    width,
                )
            )
            backbone_input = group.flatten(0, 1).contiguous()
            if self.freeze_backbone:
                with torch.no_grad():
                    raw_features = self.model.backbone(backbone_input)
            else:
                raw_features = self.model.backbone(backbone_input)
            if isinstance(raw_features, (tuple, list)):
                group_features = torch.cat(
                    [
                        self._duca_pool_variable_features(
                            value,
                            batch=batch,
                            packet_count=packet_count,
                            num_segs=num_segs,
                            execution_slots=execution_slots,
                            represented_observations=torch.where(
                                context["effective_budget"].index_select(0, rows) == 384,
                                context["execution_slots"].index_select(0, rows),
                                context["actual_observations"].index_select(0, rows),
                            ),
                            detector_length=int(context["detector_length"]),
                        )
                        for value in raw_features
                    ],
                    dim=1,
                )
            else:
                group_features = self._duca_pool_variable_features(
                    raw_features,
                    batch=batch,
                    packet_count=packet_count,
                    num_segs=num_segs,
                    execution_slots=execution_slots,
                    represented_observations=torch.where(
                        context["effective_budget"].index_select(0, rows) == 384,
                        context["execution_slots"].index_select(0, rows),
                        context["actual_observations"].index_select(0, rows),
                    ),
                    detector_length=int(context["detector_length"]),
                )
            for local_index, row in enumerate(rows.tolist()):
                row_features[row] = group_features[local_index : local_index + 1]
        if any(value is None for value in row_features):
            raise RuntimeError("DUCA variable-length execution did not produce every batch row")
        features = torch.cat(row_features, dim=0)
        if masks is not None:
            if masks.shape != (int(features.shape[0]), int(context["detector_length"])):
                raise ValueError("DUCA detector mask must be [B,384]")
            features = features * masks[:, None, :].detach().to(dtype=features.dtype)
        self.last_execution_profile = self._duca_execution_profile(
            context, used_legacy_k384_path=False
        )
        return features.to(torch.float32)

    @staticmethod
    def _duca_pool_variable_features(
        features,
        *,
        batch,
        packet_count,
        num_segs,
        execution_slots,
        represented_observations,
        detector_length,
    ):
        features = features.unflatten(0, (batch * packet_count, num_segs))
        if features.ndim < 4:
            raise ValueError("VideoMAE features must retain a temporal axis")
        reduce_dims = (1,) + tuple(range(4, features.ndim))
        features = features.mean(dim=reduce_dims)
        features = (
            features.unflatten(0, (batch, packet_count))
            .permute(0, 2, 1, 3)
            .reshape(batch, features.shape[1], -1)
        )
        temporal_tokens = int(features.shape[-1])
        rows = []
        for row in range(int(batch)):
            represented = int(represented_observations[row].item())
            valid_tokens = (
                represented * temporal_tokens + int(execution_slots) - 1
            ) // int(execution_slots)
            if valid_tokens <= 0 or valid_tokens > temporal_tokens:
                raise ValueError("DUCA packet feature trim produced an invalid temporal length")
            rows.append(
                F.interpolate(
                    features[row : row + 1, :, :valid_tokens],
                    size=int(detector_length),
                    mode="linear",
                    align_corners=False,
                )
            )
        return torch.cat(rows, dim=0)

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

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim, actual_positions=None, canonical_positions=None):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames, actual_positions=None, canonical_positions=None):
            return self.model.backbone(frames, actual_positions=actual_positions, canonical_positions=canonical_positions)

        video_feat = []
        for chunk_index, mini_frames in enumerate(torch.chunk(frames, chunk_num, dim=chunk_dim)):  # B*N is chunked
            mini_actual = mini_canonical = None
            if actual_positions is not None and chunk_dim == 0:
                start = sum(x.shape[0] for x in torch.chunk(frames, chunk_num, dim=chunk_dim)[:chunk_index])
                mini_actual = actual_positions[start:start + mini_frames.shape[0]]
                mini_canonical = canonical_positions[start:start + mini_frames.shape[0]]
            # we can use torch.cp.checkpoint to implement an efficient temporal checkpointing mechanism
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
