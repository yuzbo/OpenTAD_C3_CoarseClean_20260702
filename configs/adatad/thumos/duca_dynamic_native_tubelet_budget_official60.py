"""Window-level real-variable-compute DUCA successor frozen by Pro."""

_base_ = ["./duca_native_tubelet_uniform_reconstruct_fixed384_official60.py"]

import os


def _required_path(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the dynamic native-tubelet experiment")
    return value


train_budget_table = _required_path("DUCA_DYNAMIC_TRAIN_BUDGET_TABLE")
validation_budget_table = _required_path("DUCA_DYNAMIC_VALIDATION_BUDGET_TABLE")

native_tubelet_contract = dict(
    attribution="window_level_real_variable_compute",
    dynamic_budget_role="main_candidate",
    fixed_k_role="immutable_job_1260184_control",
    selection_policy="native_tubelet_dynamic_uniform",
    clip_budgets=[16, 20, 24],
    mean_clips_per_video=20,
    realized_heavy_compute_required=True,
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        acquisition_policy="native_tubelet_dynamic_uniform",
        native_tubelet_selected_count=192,
    ),
    backbone=dict(
        custom=dict(
            variable_clip_len=16,
            post_processing_pipeline=[],
        ),
    ),
)

dataset = dict(
    train=dict(duca_native_tubelet_budget_table=train_budget_table),
    val=dict(duca_native_tubelet_budget_table=validation_budget_table),
    test=dict(duca_native_tubelet_budget_table=validation_budget_table),
)

# Structured terminal evidence requires the official prediction dictionary.
post_processing = dict(save_dict=True)

workflow = dict(training_profile="duca_dynamic_native_tubelet_budget_official60")
work_dir = "exps/thumos/adatad/duca_dynamic_native_tubelet_budget_official60"
