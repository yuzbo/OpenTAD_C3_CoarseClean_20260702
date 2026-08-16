"""Official-validation replay of the trained Q arm with route telemetry enabled."""

import os

_base_ = ["./zoomtoken_full_official_q_seed3407_v001.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_random_seed=3408,
            georoute_diagnostic_telemetry_enabled=True,
            georoute_role_calibration_telemetry_enabled=False,
        )
    )
)
solver = dict(test=dict(batch_size=1))
georoute_diagnostic_telemetry = dict(enabled=True)
georoute_development_profile = dict(enabled=False)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=True)
post_processing = dict(save_dict=True)
georoute_protocol = dict(
    status="official_validation_diagnostic_replay",
    diagnostic_only=True,
    changes_checkpoint=False,
    changes_predictions=False,
    changes_evaluator=False,
)
work_dir = os.environ["ZOOMTOKEN_Q_DIAGNOSTIC_RUN_ROOT"]

