from __future__ import annotations


DUCA_LOSS_WEIGHT_DEFAULTS = {
    "detector": 1.0,
    "actionness": 0.0,
    "budget": 0.05,
    "boundary": 0.25,
    "hole": 0.25,
    "max_gap_hole": 0.0,
    "redundancy": 0.05,
    "radius": 0.02,
    "entropy": 0.01,
    "teacher": 0.50,
    "detector_utility": 0.0,
    "start": 0.0,
    "end": 0.0,
    "context": 0.0,
    "lagrangian_budget": 1.0,
    "marginal_monotonic": 0.01,
    "hard_budget_cap": 1.0,
    "transition": 0.0,
    "transition_boundary": 0.0,
}

DUCA_LOSS_TO_WEIGHT_KEY = {
    "detector_loss": "detector",
    "budget_loss": "budget",
    "lagrangian_budget_loss": "lagrangian_budget",
    "marginal_monotonic_loss": "marginal_monotonic",
    "hard_budget_cap_loss": "hard_budget_cap",
    "teacher_utility_loss": "teacher",
    "boundary_utility_proxy_distribution_loss": "detector_utility",
    "start_endpoint_distribution_loss": "start",
    "end_endpoint_distribution_loss": "end",
    "boundary_context_distribution_loss": "context",
    "transition_distribution_loss": "transition",
    "transition_boundary_coverage_loss": "transition_boundary",
    "boundary_coverage_loss": "boundary",
    "actionness_bce_loss": "actionness",
    "action_local_hole_loss": "hole",
    "temporal_max_gap_hole_loss": "max_gap_hole",
    "redundancy_loss": "redundancy",
    "radius_cost_loss": "radius",
    "entropy_anti_collapse_loss": "entropy",
}


if set(DUCA_LOSS_TO_WEIGHT_KEY.values()) != set(DUCA_LOSS_WEIGHT_DEFAULTS):
    raise RuntimeError("DUCA loss inventory and weight inventory disagree")


__all__ = ["DUCA_LOSS_TO_WEIGHT_KEY", "DUCA_LOSS_WEIGHT_DEFAULTS"]
