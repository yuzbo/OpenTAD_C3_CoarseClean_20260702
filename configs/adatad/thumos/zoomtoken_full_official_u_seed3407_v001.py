"""Uniform token-selection control for the complete official THUMOS14 experiment."""

import os

_base_ = ["./zoomtoken_full_official_q_seed3407_v001.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_route_mode="uniform",
            georoute_policy_estimator="none",
            georoute_tokens_per_tubelet=64,
        )
    )
)
georoute_protocol = dict(
    arm="uniform_same_budget",
    dynamic_per_tubelet_budget=False,
)
work_dir = f"{os.environ['ZOOMTOKEN_FULL_RUN_ROOT']}/U"
