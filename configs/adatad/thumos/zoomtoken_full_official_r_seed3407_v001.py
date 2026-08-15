"""Seeded random token-selection control for the complete official THUMOS14 experiment."""

import os

_base_ = ["./zoomtoken_full_official_q_seed3407_v001.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_route_mode="random",
            georoute_policy_estimator="none",
            georoute_tokens_per_tubelet=64,
            georoute_random_seed=3407,
        )
    )
)
georoute_protocol = dict(
    arm="random_same_budget",
    dynamic_per_tubelet_budget=False,
)
work_dir = f"{os.environ['ZOOMTOKEN_FULL_RUN_ROOT']}/R"
