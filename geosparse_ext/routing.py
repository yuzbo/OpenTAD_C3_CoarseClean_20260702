"""Hard routing, exact ordered policy likelihood, and signed task cost losses."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import RoutePlan


def ordered_log_prob(logits, order):
    if logits.ndim != 1 or order.ndim != 1 or order.dtype != torch.long:
        raise ValueError("PL expects logits[N] and int64 sampling order[K]")
    if order.numel() and (int(order.min()) < 0 or int(order.max()) >= len(logits) or order.unique().numel() != len(order)):
        raise ValueError("invalid or repeated policy action")
    if not len(order):
        return logits.sum() * 0
    x = logits if logits.dtype == torch.float64 else logits.float()
    remaining = torch.ones_like(x, dtype=torch.bool)
    remaining[order] = False
    denom = torch.logaddexp(torch.logcumsumexp(x[order].flip(0), 0).flip(0), torch.logsumexp(x[remaining], 0))
    return (x[order] - denom).sum()


def sample_subset(logits, k, generator=None):
    if not 0 <= k <= len(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("invalid subset size or nonfinite logits")
    if k in {0, len(logits)}:
        return torch.arange(k, device=logits.device), logits.sum() * 0
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32, generator=generator)
    u = u.clamp(torch.finfo(u.dtype).eps, 1 - torch.finfo(u.dtype).eps)
    order = (logits.float() - (-u.log()).log()).topk(k, sorted=True).indices
    return order, ordered_log_prob(logits, order)


def exploration_probability(epoch, warmup=6, mode="default"):
    if epoch < warmup:
        return 1.0
    if mode == "none":
        return 0.0
    if mode.startswith("constant_"):
        p = float(mode.split("_", 1)[1])
        if p not in {0.1, 0.5}:
            raise ValueError("unregistered exploration probability")
        return p
    if mode != "default":
        raise ValueError("unregistered exploration schedule")
    return max(0.1, 0.5 - 0.4 * max(0, epoch - 6) / 14)


class Scout(nn.Module):
    def __init__(self, width=128, resolution=112, temporal_stride=1):
        super().__init__()
        self.resolution, self.temporal_stride = resolution, temporal_stride
        layers, inc = [], 3
        for out in [width // 4, width // 2, width, width]:
            layers.extend([nn.Conv2d(inc, inc, 3, stride=2, padding=1, groups=inc),
                           nn.Conv2d(inc, out, 1), nn.GroupNorm(1, out), nn.GELU()])
            inc = out
        self.spatial = nn.Sequential(*layers)
        self.temporal = nn.Sequential(*[layer for _ in range(2) for layer in (
            nn.Conv1d(width, width, 3, padding=1, groups=width), nn.Conv1d(width, width, 1), nn.GELU())])
        self.gain = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.actionness = nn.Linear(width, 1)
        self.budget = nn.Linear(width, 5)
        self.critic = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, frames, valid):
        b, c, t, _, _ = frames.shape
        if t % 2:
            raise ValueError("Scout receives intact native pairs")
        # Subsample whole pairs, never re-pair isolated frames.
        frames = frames * valid[:, None, :, None, None].to(frames.dtype)
        pair = frames.reshape(b, c, t // 2, 2, *frames.shape[-2:])[:, :, ::self.temporal_stride]
        nt = pair.shape[2]
        images = pair.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, *frames.shape[-2:])
        images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        feats = F.interpolate(self.spatial(images), (7, 7), mode="bilinear", align_corners=False)
        feats = feats.reshape(b, nt, 2, -1, 7, 7).mean(2)
        feats = feats.permute(0, 4, 3, 2, 1).reshape(b * 49, -1, nt)
        selected_valid = valid.reshape(b, t // 2, 2).any(-1)[:, ::self.temporal_stride]
        temporal_mask = selected_valid[:, None, None].expand(-1, 49, 1, -1).reshape(b * 49, 1, nt)
        for layer in self.temporal:
            feats = layer(feats * temporal_mask.to(feats.dtype))
        feats = F.interpolate(feats * temporal_mask.to(feats.dtype), size=t // 2, mode="linear", align_corners=False)
        feats = feats.reshape(b, 7, 7, -1, t // 2).permute(0, 4, 2, 1, 3)
        pair_valid = valid.reshape(b, t // 2, 2).any(-1)
        feats = feats * pair_valid[:, :, None, None, None].to(feats.dtype)
        pooled = feats.sum((1, 2, 3)) / (pair_valid.sum(1).clamp_min(1) * 49)[:, None]
        return feats, self.budget(pooled), self.critic(pooled).squeeze(-1)


def atom_features(features, atoms, native_hw):
    b, t, h, w, d = features.shape
    x = F.interpolate(features.permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w),
                      native_hw, mode="bilinear", align_corners=False)
    x = x.reshape(b, t, d, *native_hw).permute(0, 1, 3, 4, 2).reshape(b, -1, d)
    return x[:, atoms].mean(2)


def _uniform(n, k, device):
    return ((torch.arange(k, device=device) + 0.5) * n / max(k, 1)).long()


def _cdf(logits, k):
    density = F.softplus(logits.float()) + 1e-4
    cdf = density.cumsum(0) / density.sum()
    chosen = torch.searchsorted(cdf, (torch.arange(k, device=logits.device) + 0.5) / k).unique(sorted=True)
    while len(chosen) < k:
        available = torch.ones_like(logits, dtype=torch.bool)
        available[chosen] = False
        ids = torch.where(available)[0]
        mass = density[ids].cumsum(0) / density[ids].sum()
        missing = k - len(chosen)
        fill = ids[torch.searchsorted(mass, (torch.arange(missing, device=ids.device) + 0.5) / missing)]
        chosen = torch.cat((chosen, fill)).unique(sorted=True)
    return chosen


def choose_order(logits, scores, k, selector, random_branch, is_learned, generator):
    if random_branch or selector == "random":
        return torch.randperm(len(logits), device=logits.device, generator=generator)[:k], logits.sum() * 0
    if selector == "uniform":
        return _uniform(len(logits), k, logits.device), logits.sum() * 0
    if selector == "cdf_native" and k:
        return _cdf(scores, k), logits.sum() * 0
    if is_learned:
        return sample_subset(logits, k, generator)
    return torch.argsort(scores, descending=True, stable=True)[:k], logits.sum() * 0


def quota_order(logits, scores, parent_ids, fraction, quota, selector, random_branch, is_learned, generator):
    """Exact likelihood for the actual constrained sampling procedure.

    Equal quota samples independent subsets in parent order. Global nonzero
    first samples one atom per valid parent, then samples the remaining global
    budget without replacement. Its latent two-stage order is part of the law.
    """
    k = min(len(logits), int(math.ceil(len(logits) * fraction)))
    if quota == "global_zero_allowed":
        return choose_order(logits, scores, k, selector, random_branch, is_learned, generator)
    if quota not in {"per_clip_equal_quota", "global_nonzero_per_clip"}:
        raise NotImplementedError(f"quota {quota} has no registered sampling law")
    groups = [torch.where(parent_ids == p)[0] for p in parent_ids.unique(sorted=True)]
    if quota == "global_nonzero_per_clip" and k < len(groups):
        raise ValueError("INFEASIBLE: global budget cannot provide one atom per valid parent")
    if k == len(logits):
        return torch.arange(k, device=logits.device), logits.sum() * 0
    chosen, probability = [], logits.sum() * 0
    for ids in groups:
        local_k = int(math.ceil(len(ids) * fraction)) if quota == "per_clip_equal_quota" else 1
        order, lp = choose_order(logits[ids], scores[ids], local_k, selector, random_branch, is_learned, generator)
        chosen.append(ids[order])
        probability = probability + lp
    order = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=logits.device)
    if quota == "global_nonzero_per_clip" and k > len(order):
        remaining = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
        remaining[order] = False
        ids = torch.where(remaining)[0]
        extra, lp = choose_order(logits[ids], scores[ids], k - len(order), selector, random_branch, is_learned, generator)
        order = torch.cat((order, ids[extra]))
        probability = probability + lp
    return order, probability


def make_plan(gain, atoms, native_valid, parent_tubelets, hw, config, budget_logits=None,
              actionness=None, motion=None, epoch=0, training=False, generators=None):
    """Hard actions; the joint likelihood includes only actually sampled actions."""
    b, a = gain.shape
    flat_valid = native_valid.flatten(1)
    valid_atoms = flat_valid[:, atoms].any(-1)
    p = native_valid.shape[1] // parent_tubelets
    n = parent_tubelets * hw[0] * hw[1]
    selector = config["selector"]
    learned_policy = selector in {"hybrid", "pg_only"}
    if selector not in {"none", "uniform", "random", "motion", "actionness", "uncertainty", "cdf_native", "hybrid", "pg_only"}:
        raise NotImplementedError(f"selector {selector} needs its registered estimator/controller")
    eps = exploration_probability(epoch, config["warmup_epochs"], config["exploration"]) if training and learned_policy else 0
    all_atoms, orders, executed, probs, eligibility, ratios, budget_ids, masks = [], [], [], [], [], [], [], []
    for row in range(b):
        gen = generators[row] if generators is not None else None
        eligible = torch.where(valid_atoms[row])[0]
        random_branch = training and learned_policy and bool(torch.rand((), device=gain.device, generator=gen) < eps)
        is_learned = learned_policy and training and not random_branch
        logp = gain[row].sum() * 0
        ratio, budget_id = float(config["budget"]), -1
        if config["budget_mode"] == "dynamic":
            menu = [0.25, 0.5, 0.75, 1.0] if config["route"] == "C" else [0., 0.25, 0.5, 0.75, 1.0]
            if not config["allow_global_zero"]:
                menu = [v for v in menu if v > 0]
            menu_indices = torch.tensor([round(v * 4) for v in menu], device=gain.device)
            logits = budget_logits[row, menu_indices]
            if training:
                probabilities = torch.ones_like(logits) if random_branch else logits.softmax(0)
                budget_id = int(torch.multinomial(probabilities, 1, generator=gen))
                if is_learned:
                    logp = logp + logits.log_softmax(0)[budget_id]
            else:
                budget_id = int(logits.argmax())
            ratio = menu[budget_id]
        fraction = (4 * ratio - 1) / 3 if config["route"] == "C" else ratio
        if not 0 <= fraction <= 1:
            raise ValueError("infeasible native budget")
        logits = gain[row, eligible]
        scores = logits
        if selector in {"actionness", "cdf_native", "uncertainty"}:
            scores = actionness[row, eligible]
            if selector == "uncertainty":
                q = scores.sigmoid().clamp(1e-6, 1 - 1e-6)
                scores = -(q * q.log() + (1 - q) * (1 - q).log())
        elif selector == "motion":
            scores = motion[row, eligible]
        order, lp = quota_order(logits, scores, atoms[eligible, 0] // n, fraction,
                                config["quota"], selector, random_branch, is_learned, gen)
        logp = logp + lp
        actual = eligible[order]
        mask = torch.zeros(a, dtype=torch.bool, device=gain.device)
        mask[actual] = True
        native = torch.zeros_like(flat_valid[row])
        native[atoms[actual].flatten()] = True
        native &= flat_valid[row]
        all_atoms.append(mask)
        orders.append(actual)
        executed.append(torch.where(native)[0])
        masks.append(native.reshape(p, n))
        probs.append(logp)
        eligibility.append(is_learned)
        ratios.append(ratio)
        budget_ids.append(budget_id)
    return RoutePlan(torch.stack(masks), torch.stack(all_atoms), atoms, orders, executed,
                     torch.stack(probs), torch.tensor(eligibility, device=gain.device),
                     torch.tensor(ratios, device=gain.device), torch.tensor(budget_ids, device=gain.device), gain)


def policy_losses(task_cost, baseline, plan, normalized_heavy_cost, target, dual):
    reward = task_cost.detach() + dual * (normalized_heavy_cost.detach() - target)
    active = plan.learned_sample.to(plan.log_prob.dtype)
    return {
        "actor_loss": ((reward - baseline).detach() * plan.log_prob * active).mean(),
        "critic_loss": (0.5 * (baseline - reward.detach()).square() * active).mean(),
    }


def acquisition_loss(predicted_gain, base_loss, acquired_loss):
    # Signed values are retained: extra evidence can increase task loss.
    target = (base_loss - acquired_loss).detach()
    return 0.1 * F.smooth_l1_loss(predicted_gain, target), target
