"""SingleClock relative physical-time attention bias."""
import torch


def exact_uniform_positions(valid_len: int, selected_count: int, device=None):
    valid_len, selected_count = int(valid_len), int(selected_count)
    if valid_len <= 0 or selected_count <= 0 or selected_count > valid_len:
        raise ValueError("valid_len and selected_count must satisfy 0 < selected_count <= valid_len")
    if selected_count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    # Python round is the H65 round-half-to-even rule.
    return torch.tensor([round(i * (valid_len - 1) / (selected_count - 1)) for i in range(selected_count)], dtype=torch.long, device=device)

def build_canonical_time_residual_bias(physical_time, canonical_time, spatial_tokens,
                                       num_heads=1, eps=1e-6, dtype=None):
    if int(num_heads) != 1:
        raise ValueError("SingleClock uses one shared attention bias; per-head expansion is forbidden")
    t = torch.as_tensor(physical_time)
    u = torch.as_tensor(canonical_time, device=t.device, dtype=t.dtype)
    if t.ndim == 1: t = t.unsqueeze(0)
    if u.ndim == 1: u = u.unsqueeze(0).expand(t.shape[0], -1)
    if t.shape != u.shape or t.ndim != 2: raise ValueError("time tensors must be [B,T]")
    r = torch.log((torch.abs(t[:, :, None] - t[:, None, :]) + eps) /
                  (torch.abs(u[:, :, None] - u[:, None, :]) + eps))
    bias = r[:, :, None, :, None, :].expand(-1, -1, spatial_tokens, -1, spatial_tokens, -1)
    bias = bias.reshape(t.shape[0], t.shape[1] * spatial_tokens, t.shape[1] * spatial_tokens)
    return bias.unsqueeze(1).to(dtype=dtype or t.dtype)
