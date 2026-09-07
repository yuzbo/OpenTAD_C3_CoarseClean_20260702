"""Native coordinates are independent of routing and packed storage order."""
from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor


@dataclass
class VideoBatch:
    frames_hi: Tensor                 # [B,C,T,H,W], before parent partitioning
    pts_s: Tensor                     # [B,T], actual decoded timestamps
    target_time_s: Tensor             # [B,T], regular output reference
    valid_frames: Tensor              # [B,T], bool
    source_frame_id: Tensor           # [B,T], original decoded indices
    source_intervals_s: Tensor        # [B,T,2], actual display intervals
    video_id: List[str]
    window_id: List[str]
    spatial_transform: Tensor         # [B,3,3], original normalized -> view normalized
    targets: Optional[List[Tensor]] = None  # [instances,class/start/end], source seconds


@dataclass
class NativeLayout:
    parent_clip_id: Tensor            # [Tn,H,W]
    token_id: Tensor                  # [Tn,H,W], native flattened indices
    patch_yx: Tensor                  # [Tn,H,W,2]
    source_support: Tensor            # [B,Tn,2,2], UNION of actual sample intervals
    support_valid: Tensor            # [B,Tn,2]
    nominal_time_s: Tensor            # [B,Tn]
    valid: Tensor                    # [B,Tn,H,W]
    parent_tubelets: int


@dataclass
class RoutePlan:
    selected_native: Tensor           # [B,P,N], bool; N=parent_tubelets*H*W
    selected_atoms: Tensor            # [B,A], bool
    atom_to_native: Tensor            # [A,G], each row is one legal disjoint atom
    sampling_order: List[Tensor]      # stochastic PL order; never encoder order
    execution_order: List[Tensor]     # native order
    log_prob: Tensor                  # [B], includes budget probability if sampled
    learned_sample: Tensor            # [B], actor eligibility
    requested_budget: Tensor          # [B], requested token ratio or cost target
    budget_id: Tensor                 # [B]
    predicted_gain: Tensor            # [B,A], signed

    @property
    def per_clip_counts(self):
        return self.selected_native.sum(-1)

    @property
    def realized_token_count(self):
        return self.per_clip_counts.sum(-1)


@dataclass
class EvidenceBatch:
    features: Tensor                 # [B,E,D]
    source_support: Tensor           # [B,E,U,2], anchor union, never a hull/receptive field
    support_valid: Tensor            # [B,E,U]
    physical_time_s: Tensor          # [B,E]
    roi_xyxy: Tensor                 # [B,E,4], original normalized coordinates
    parent_clip_id: Tensor           # [B,E]
    valid: Tensor                    # [B,E]
    fidelity: Tensor                 # [B,E]
    roi_polygon: Optional[Tensor] = None  # [B,E,4,2], unclipped inverse-mapped anchor corners
    # Features depend on all selected tokens in parent_clip_id, not only this
    # anchor. This structure is not an independent packet cache key.


@dataclass
class DetectionState:
    features: Tensor                 # [B,D,Q]
    regular_time_s: Tensor            # [B,Q]
    valid: Tensor                    # [B,Q]
    grid_stride_s: Tensor            # [B]
