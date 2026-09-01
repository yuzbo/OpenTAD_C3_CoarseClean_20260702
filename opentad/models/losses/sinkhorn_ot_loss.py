import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class SinkhornOptimalTransportLoss(nn.Module):
    """Entropy-regularized Sinkhorn Optimal Transport Loss for 1D Temporal Frame Selection.

    Aligns the continuous distribution of Scout frame selection probabilities with the target
    measure derived from ground-truth action segments and phase transitions.

    Args:
        epsilon (float): Entropy regularization parameter. Default: 0.05.
        num_iters (int): Number of Sinkhorn-Knopp iterations. Default: 20.
        temperature (float): Softmax temperature for converting logits to probabilities. Default: 1.0.
        boundary_bandwidth (float): Standard deviation of Gaussian kernel for boundary targets. Default: 2.0.
        loss_weight (float): Loss scaling weight. Default: 1.0.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        num_iters: int = 20,
        temperature: float = 1.0,
        boundary_bandwidth: float = 2.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.temperature = float(temperature)
        self.boundary_bandwidth = float(boundary_bandwidth)
        self.loss_weight = float(loss_weight)

    def _build_cost_matrix(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalized quadratic cost matrix on 1D grid [0, 1]."""
        coords = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)
        cost = (coords.unsqueeze(0) - coords.unsqueeze(1)) ** 2
        return cost  # [T, T]

    def _sinkhorn_1d(
        self,
        p: torch.Tensor,  # [B, T] source/target measure 1
        q: torch.Tensor,  # [B, T] predicted measure 2
        cost: torch.Tensor,  # [T, T]
    ) -> torch.Tensor:  # [B]
        """Computes regularized Wasserstein distance using Sinkhorn-Knopp algorithm."""
        B, T = p.shape
        K = torch.exp(-cost / self.epsilon)  # [T, T] Gibbs kernel

        u = torch.ones((B, T), device=p.device, dtype=p.dtype)
        for _ in range(self.num_iters):
            # v = q / (K^T u + eps)
            Kv = torch.matmul(u, K)  # [B, T]
            v = q / (Kv.clamp_min(1e-8))
            # u = p / (K v + eps)
            Ku = torch.matmul(v, K.t())  # [B, T]
            u = p / (Ku.clamp_min(1e-8))

        # Optimal transport plan: P = diag(u) * K * diag(v)
        # Cost: sum_{i,j} P_{ij} * C_{ij}
        # Vectorized: sum_b sum_i u_b,i * sum_j K_ij * C_ij * v_b,j
        KC = K * cost  # [T, T]
        transport_cost = torch.sum(u * torch.matmul(v, KC.t()), dim=-1)  # [B]
        return transport_cost

    def _generate_target_measure(
        self,
        gt_segments: list[torch.Tensor],
        valid_masks: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Generates continuous target measure from GT action boundaries and valid masks."""
        B = len(gt_segments)
        device = valid_masks.device
        dtype = torch.float32
        target_measure = torch.zeros((B, T), device=device, dtype=dtype)
        grid = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)  # [1, T]

        for b in range(B):
            mask_b = valid_masks[b].float()
            valid_len = mask_b.sum().clamp_min(1.0)
            segs = gt_segments[b]
            if segs is not None and len(segs) > 0:
                # Place Gaussian densities on start and end boundaries
                starts = segs[:, 0:1]  # [Num_gt, 1]
                ends = segs[:, 1:2]  # [Num_gt, 1]
                # Gaussian on starts and ends
                dist_start = (grid - starts) ** 2
                dist_end = (grid - ends) ** 2
                gauss_start = torch.exp(-dist_start / (2.0 * (self.boundary_bandwidth ** 2)))
                gauss_end = torch.exp(-dist_end / (2.0 * (self.boundary_bandwidth ** 2)))
                boundary_density = (gauss_start.sum(dim=0) + gauss_end.sum(dim=0)) * mask_b
                # Add uniform scaffold baseline to avoid empty background
                scaffold_density = 0.1 * mask_b
                density = boundary_density + scaffold_density
            else:
                density = mask_b

            # Normalize to valid probability measure (sums to 1)
            target_measure[b] = density / density.sum().clamp_min(1e-6)

        return target_measure

    def forward(
        self,
        pred_logits: torch.Tensor,  # [B, T]
        gt_segments: list[torch.Tensor],  # list of [Num_gt, 2]
        valid_masks: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        B, T = pred_logits.shape
        device = pred_logits.device
        dtype = pred_logits.dtype

        # Masked softmax to construct predicted empirical measure
        masked_logits = pred_logits.masked_fill(~valid_masks, -1e4)
        pred_probs = F.softmax(masked_logits / self.temperature, dim=-1)  # [B, T]

        # Target measure
        target_probs = self._generate_target_measure(gt_segments, valid_masks, T).to(dtype=dtype)

        # Cost matrix
        cost = self._build_cost_matrix(T, device=device, dtype=dtype)

        # Sinkhorn distance
        ot_loss = self._sinkhorn_1d(target_probs, pred_probs, cost)
        return self.loss_weight * ot_loss.mean()
