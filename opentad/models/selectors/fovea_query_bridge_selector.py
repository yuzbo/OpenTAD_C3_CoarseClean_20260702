"""FoveaSampler / Query-Bridge frame selector.

Independent DUCA selector arm implementing the user-approved design:

* dense temporal observation is encoded by :class:`FoveaScout`;
* a small query bank cross-attends to the scout memory to produce the
  contribution heatmap ``A`` and the internal query memory ``Q1``
  (``Q1`` never enters the heavy detector);
* the three manual score branches (saliency / boundary / uncertainty) stay
  intact and are fused as ``s = saliency + boundary_edge + uncertainty_context``;
* a coarse proposal head provides the auxiliary ``L_coarse``;
* :class:`FoveatedSampler` turns the score map into an exact-``K`` foveated
  frame set; training uses the Gumbel-TopK straight-through surrogate,
  inference is deterministic greedy MMR;
* the heavy detector receives only the selected high-resolution frames plus
  physical-position metadata consumed by the dynamic sparse temporal backbone.

The optional post-heavy cycle loss is deferred: the detector calls
:meth:`set_cycle_feedback` after its head forward and then
:meth:`finalize_cycle_loss` to replace the placeholder cycle term.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS
from ..losses.fovea_losses import build_fovea_losses, focal_bce
from .fovea_heads import CoarseProposalHead, FoveaHeads
from .fovea_sampler import FoveatedSampler
from .fovea_scout import FoveaScout
from .query_bridge import QueryBridgeWithDecoder
from .pc_ot_mras_prebackbone_frame_selector import _as_bool_prefix_mask


def _lowres_observations(inputs: torch.Tensor, target_len: int = 32) -> torch.Tensor:
    """Build ``[B,3,T,32,32]`` scout observations from dense ``[B,3,T,H,W]``."""
    if inputs.ndim != 5:
        raise ValueError("descriptor input must be [B,3,T,H,W]")
    b, c, t, h, w = inputs.shape
    flat = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = F.interpolate(flat, size=(target_len, target_len), mode="bilinear", align_corners=False)
    flat = flat.reshape(b, t, c, target_len, target_len).permute(0, 2, 1, 3, 4)
    return flat.contiguous()


def _transport_inputs(
    inputs: torch.Tensor,
    transport: torch.Tensor,
    indices: torch.Tensor,
    selected_len: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Hard-gather selected frames while keeping the straight-through gradient.

    ``transport`` already contains the hard one-hot selection plus the
    differentiable ``soft - soft.detach()`` surrogate.  ``indices`` is detached
    and used for the explicit hard pass, so the returned tensor is numerically
    identical to a pure hard gather.
    """
    if inputs.ndim != 5:
        raise ValueError("selected transport expects [B,C,T,H,W]")
    b, c, t, h, w = inputs.shape
    flat = inputs.permute(0, 2, 1, 3, 4).reshape(b, t, -1)
    hard = flat.gather(1, indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
    soft = torch.einsum("bkt,btf->bkf", transport, flat)
    gathered = hard + (soft - soft.detach())
    gathered = gathered.reshape(b, selected_len, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return gathered, [indices]


def _fill_base_meta(
    meta: Dict[str, Any],
    positions: torch.Tensor,
    dense_valid_len: int,
) -> Dict[str, Any]:
    out = dict(meta)
    out.update(
        {
            "duca_sparse_variable_compute": True,
            "duca_sparse_physical_positions": positions.detach().cpu().tolist(),
            "duca_selector": "FoveaQueryBridgeFrameSelector",
            "duca_selector_original_valid_frames": int(dense_valid_len),
            "duca_selector_selected_frames": int(positions.numel()),
            "duca_selector_center_offset_frames": int(meta.get("center_offset_frames", 0)),
        }
    )
    return out


def _dense_cycle_mask_from_feedback(
    proposals: torch.Tensor,
    scores: torch.Tensor,
    length: int,
    valid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert detached post-heavy proposals into a dense 0/1 cycle target."""
    if proposals.numel() == 0 or scores.numel() == 0:
        return torch.zeros((1, length), dtype=torch.float32, device=device)
    props = torch.as_tensor(proposals, device=device, dtype=torch.float32).reshape(-1, 2)
    conf = torch.as_tensor(scores, device=device, dtype=torch.float32)
    if conf.ndim == 2:
        conf = conf.max(dim=-1).values
    conf = conf.reshape(-1).clamp(0.0, 1.0)
    keep = conf >= 0.5
    if not bool(keep.any().item()):
        return torch.zeros((1, length), dtype=torch.float32, device=device)
    props = props[keep]
    conf = conf[keep]
    top_n = min(int(props.shape[0]), 64)
    top = torch.topk(conf, top_n).indices
    props = props[top]
    idx = torch.arange(length, device=device, dtype=torch.float32)
    target = torch.zeros((1, length), dtype=torch.float32, device=device)
    for start, end in props:
        lo = float(start.clamp(min=0.0, max=float(length - 1)).item())
        hi = float(end.clamp(min=0.0, max=float(length - 1)).item())
        if hi < lo:
            lo, hi = hi, lo
        inside = (idx >= lo) & (idx <= hi)
        target[0, inside] = 1.0
    target = target.masked_fill(~valid.bool(), 0.0)
    return target


@SELECTORS.register_module()
class FoveaQueryBridgeFrameSelector(nn.Module):
    """FoveaSampler / Query-Bridge pre-backbone frame selector."""

    def __init__(
        self,
        # scout
        scout_in_dim: int = 3 * 32 * 32,
        scout_hidden_dim: int = 96,
        scout_temporal_layers: int = 4,
        scout_kernel_size: int = 5,
        scout_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        scout_dropout: float = 0.10,
        scout_target_len: int = 32,
        # query bridge
        query_hidden_dim: int = 96,
        num_queries: int = 4,
        query_decoder_layers: int = 2,
        query_num_heads: int = 4,
        query_dropout: float = 0.10,
        # foveated sampler
        target_k: int = 384,
        min_k: int = 256,
        max_k: int = 512,
        budget_step: int = 16,
        boundary_quota: int = 64,
        boundary_center_top_m: int = 8,
        boundary_radius: int = 2,
        boundary_pair_max_gap: int = 8,
        mmr_lambda: float = 0.10,
        gumbel_tau: float = 1.0,
        # score composition (three manual branches are always built, never deleted)
        score_mode: str = "fused_three_branch",
        # losses
        loss_mask_weight: float = 1.0,
        loss_coarse_weight: float = 1.0,
        loss_cycle_weight: float = 0.0,
        loss_budget_weight: float = 0.05,
        loss_diversity_weight: float = 0.05,
        cycle_warmup_iterations: int = 1500,
        cycle_enabled: bool = False,
    ) -> None:
        super().__init__()
        if target_k % 16 != 0:
            raise ValueError("target_k must be divisible by 16 for the VideoMAE clip path")
        if query_hidden_dim % query_num_heads != 0:
            raise ValueError("query_hidden_dim must be divisible by query_num_heads")
        if score_mode not in ("fused_three_branch", "query_contribution"):
            raise ValueError(f"unsupported fovea score_mode: {score_mode!r}")
        self.score_mode = str(score_mode)
        self.scout = FoveaScout(
            in_dim=scout_in_dim,
            hidden_dim=scout_hidden_dim,
            temporal_layers=scout_temporal_layers,
            kernel_size=scout_kernel_size,
            dilations=scout_dilations,
            dropout=scout_dropout,
        )
        self.query_bridge = QueryBridgeWithDecoder(
            hidden_dim=query_hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=query_decoder_layers,
            num_heads=query_num_heads,
            dropout=query_dropout,
        )
        self.heads = FoveaHeads(hidden_dim=query_hidden_dim, num_queries=num_queries)
        self.coarse_head = CoarseProposalHead(hidden_dim=query_hidden_dim)
        self.sampler = FoveatedSampler(
            target_k=target_k,
            min_k=min_k,
            max_k=max_k,
            budget_step=budget_step,
            boundary_quota=boundary_quota,
            boundary_center_top_m=boundary_center_top_m,
            boundary_radius=boundary_radius,
            boundary_pair_max_gap=boundary_pair_max_gap,
            mmr_lambda=mmr_lambda,
            dynamic_budget=True,
            gumbel_tau=gumbel_tau,
        )

        self.scout_target_len = int(scout_target_len)
        self.target_k = int(target_k)
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.loss_mask_weight = float(loss_mask_weight)
        self.loss_coarse_weight = float(loss_coarse_weight)
        self.loss_cycle_weight = float(loss_cycle_weight)
        self.loss_budget_weight = float(loss_budget_weight)
        self.loss_diversity_weight = float(loss_diversity_weight)
        self.cycle_warmup_iterations = int(cycle_warmup_iterations)
        self.cycle_enabled = bool(cycle_enabled)

        # Training-only deferred state (never used at inference).
        self._step = 0
        self._pending_cycle: Optional[Dict[str, torch.Tensor]] = None
        self._cycle_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # frontend and selection
    # ------------------------------------------------------------------
    def _run_frontend(self, inputs: torch.Tensor, masks: torch.Tensor) -> Dict[str, Any]:
        if inputs.ndim != 5:
            raise ValueError("FoveaQueryBridgeFrameSelector expects dense [B,C,T,H,W] inputs")
        if masks.ndim != 2:
            raise ValueError("selector masks must be [B,T]")
        if masks.shape[0] != inputs.shape[0] or masks.shape[1] != inputs.shape[2]:
            raise ValueError("inputs and masks have inconsistent temporal lengths")
        valid = _as_bool_prefix_mask(masks, expected_shape=(int(masks.shape[0]), int(masks.shape[1])))
        lowres = _lowres_observations(inputs, target_len=self.scout_target_len)
        z = self.scout(lowres, valid)
        bridge = self.query_bridge(z, valid)
        heads = self.heads(z, bridge.contribution, bridge.query_memory, valid)
        if self.score_mode == "query_contribution":
            contribution_score = bridge.contribution.max(dim=1).values
            heads["frame_score"] = contribution_score.masked_fill(~valid, -torch.inf)
            heads["uncertainty"] = torch.sigmoid(contribution_score)
        coarse = self.coarse_head(z, bridge.contribution, valid, query_memory=bridge.query_memory)
        return {
            "valid": valid,
            "scout_z": z,
            "contribution": bridge.contribution,
            "query_memory": bridge.query_memory,
            "heads": heads,
            "coarse": coarse,
        }

    def _select(self, frontend: Dict[str, Any], inputs: torch.Tensor, training: bool) -> Dict[str, Any]:
        valid = frontend["valid"]
        batch = int(inputs.shape[0])
        frame_score = frontend["heads"]["frame_score"]
        contribution = frontend["contribution"]
        uncertainty = frontend["heads"]["uncertainty"]

        indices_list: List[torch.Tensor] = []
        positions_list: List[torch.Tensor] = []
        transport_list: List[torch.Tensor] = []
        probs_list: List[torch.Tensor] = []
        for i in range(batch):
            result = self.sampler(
                frame_score[i].unsqueeze(0),
                valid[i].unsqueeze(0),
                contribution=contribution[i].unsqueeze(0),
                training=training,
                global_start_offset=0,
                uncertainty=uncertainty[i].unsqueeze(0),
            )
            indices_list.append(result["indices"].squeeze(0))
            positions_list.append(result["positions"].squeeze(0))
            transport_list.append(result["transport"].squeeze(0))
            probs_list.append(result["probs"].squeeze(0))

        indices = torch.stack(indices_list, dim=0)
        positions = torch.stack(positions_list, dim=0)
        transport = torch.stack(transport_list, dim=0)
        probs = torch.stack(probs_list, dim=0)
        selected_len = int(indices.shape[1])
        selected_inputs, _ = _transport_inputs(inputs, transport, indices, selected_len)
        return {
            "indices": indices,
            "positions": positions,
            "transport": transport,
            "probs": probs,
            "selected_inputs": selected_inputs,
            "selected_len": selected_len,
        }

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: List[Dict[str, Any]],
        gt_segments: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, Any]:
        training = bool(self.training)
        frontend = self._run_frontend(inputs, masks)
        selection = self._select(frontend, inputs, training=training)
        selected_len = int(selection["selected_len"])
        dense_valid_len = int(frontend["valid"].long().sum(dim=1)[0].item())

        selected_masks = masks
        selected_metas: List[Dict[str, Any]] = []
        for i, meta in enumerate(metas):
            selected_metas.append(_fill_base_meta(meta, selection["positions"][i], dense_valid_len))

        bundle = build_fovea_losses(
            contribution=selection["contribution"],
            frame_score=frontend["heads"]["frame_score"],
            coarse_logits=frontend["coarse"]["coarse_logits"],
            coarse_center=frontend["coarse"]["coarse_center"],
            coarse_width=frontend["coarse"]["coarse_width"],
            valid=frontend["valid"],
            gt_segments=gt_segments,
            boundary_radius=int(self.sampler.boundary_radius),
            cycle_mask=None,
            budget_target=int(self.target_k),
            selected_count=selection["probs"].sum(dim=1),
            weights={
                "mask": self.loss_mask_weight,
                "coarse": self.loss_coarse_weight,
                "cycle": 0.0,
                "budget": self.loss_budget_weight,
                "diversity": self.loss_diversity_weight,
            },
        )
        losses = self._loss_dict(bundle)

        if training:
            self._step += 1
            self._cycle_context = {
                "frame_score": frontend["heads"]["frame_score"],
                "valid": frontend["valid"],
            }
            self._pending_cycle = None
        else:
            self._cycle_context = None
            self._pending_cycle = None

        return {
            "inputs": selection["selected_inputs"],
            "masks": selected_masks,
            "metas": selected_metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
            "selected_len": selected_len,
        }

    def forward_test(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if metas is None:
            metas = [dict() for _ in range(inputs.shape[0])]
        frontend = self._run_frontend(inputs, masks)
        selection = self._select(frontend, inputs, training=False)
        dense_valid_len = int(frontend["valid"].long().sum(dim=1)[0].item())
        selected_metas = [
            _fill_base_meta(meta, selection["positions"][i], dense_valid_len) for i, meta in enumerate(metas)
        ]
        return {
            "inputs": selection["selected_inputs"],
            "masks": masks,
            "metas": selected_metas,
            "selected_len": int(selection["selected_len"]),
        }

    # ------------------------------------------------------------------
    # optional cycle feedback (training only, target detached)
    # ------------------------------------------------------------------
    def cycle_weight_now(self, step: int, training: bool) -> float:
        if not training or not self.cycle_enabled:
            return 0.0
        base = self.loss_cycle_weight if float(self.loss_cycle_weight) > 0.0 else 0.5
        if step < self.cycle_warmup_iterations:
            return 0.0
        return min(base, base * (step - self.cycle_warmup_iterations) / 500.0)

    def cycle_feedback_requested(self) -> bool:
        return bool(
            self.training
            and self.cycle_enabled
            and self._cycle_context is not None
            and self._step >= self.cycle_warmup_iterations
        )

    def set_cycle_feedback(self, proposals: torch.Tensor, scores: torch.Tensor) -> None:
        """Attach a detached post-heavy cycle target for the current batch."""
        if not self.cycle_enabled or self._cycle_context is None or self._step < self.cycle_warmup_iterations:
            self._pending_cycle = None
            return
        if isinstance(proposals, (list, tuple)):
            proposals = proposals[0] if len(proposals) else torch.empty(0, 2)
        if isinstance(scores, (list, tuple)):
            scores = scores[0] if len(scores) else torch.empty(0)
        self._pending_cycle = {
            "proposals": proposals.detach(),
            "scores": scores.detach(),
        }

    def finalize_cycle_loss(self) -> torch.Tensor:
        """Replace the placeholder cycle loss after the detector head forward."""
        if self._cycle_context is None:
            return torch.zeros((), dtype=torch.float32, device=torch.device("cpu"))
        context = self._cycle_context
        self._cycle_context = None
        frame_score = context["frame_score"]
        valid = context["valid"]
        zero = frame_score.new_zeros(())
        weight = self.cycle_weight_now(self._step, training=True)
        if self._pending_cycle is None or weight <= 0.0:
            self._pending_cycle = None
            return zero
        cycle_mask = _dense_cycle_mask_from_feedback(
            self._pending_cycle["proposals"],
            self._pending_cycle["scores"],
            int(valid.shape[1]),
            valid,
            frame_score.device,
        )
        self._pending_cycle = None
        return focal_bce(frame_score, cycle_mask, valid) * weight

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    @staticmethod
    def _loss_dict(bundle) -> Dict[str, torch.Tensor]:
        return {
            "selector_fovea_mask_loss": bundle.mask_loss,
            "selector_fovea_coarse_loss": bundle.coarse_loss,
            "selector_fovea_cycle_loss": bundle.cycle_loss,
            "selector_fovea_budget_loss": bundle.budget_loss,
            "selector_fovea_diversity_loss": bundle.diversity_loss,
        }

    @torch.no_grad()
    def validation_probe(self, inputs: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_test(inputs, masks, None)

    def get_state_dict_meta(self) -> Dict[str, Any]:
        return {
            "selector": self.__class__.__name__,
            "target_k": self.target_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "scout_in_dim": int(self.scout.in_dim),
            "scout_hidden_dim": int(self.scout.hidden_dim),
            "num_queries": int(self.query_bridge.num_queries),
        }
