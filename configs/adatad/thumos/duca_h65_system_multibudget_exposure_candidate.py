_base_ = ["./duca_h65_system_multibudget_exposure_control.py"]

import json
import os


def _required_float(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required after PRE_RUN actual-cost calibration")
    return float(value)


p256 = _required_float("DUCA_MB_P256")
p384 = 0.5
p512 = _required_float("DUCA_MB_P512")
seed = int(os.environ.get("DUCA_EXPERIMENT_SEED", ""))
if seed not in {3407, 3408, 3409}:
    raise ValueError("DUCA_EXPERIMENT_SEED must be 3407, 3408, or 3409")
if abs(p256 + p384 + p512 - 1.0) > 1.0e-9:
    raise ValueError("PRE_RUN-calibrated K256/K384/K512 probabilities must sum to one")

evaluation_budget = int(os.environ.get("DUCA_MB_EVAL_BUDGET", "384"))
evaluation_manifest_path = os.environ.get("DUCA_MB_EVAL_MANIFEST", "")
evaluation_manifest = None
if evaluation_manifest_path:
    with open(evaluation_manifest_path, "r", encoding="utf-8") as handle:
        evaluation_manifest = json.load(handle)
    if not isinstance(evaluation_manifest, dict):
        raise ValueError("DUCA_MB_EVAL_MANIFEST must contain one JSON object")


duca_multibudget_exposure_contract = dict(
    arm="candidate_k256_k384_k512_exposure",
    nested_sets="S256_subset_S384_subset_S512",
    producer="existing_h65_sampling_rate_order",
    p256=p256,
    p384=p384,
    p512=p512,
    one_homogeneous_budget_per_successful_update=True,
    budget_clock="successful_optimizer_update",
    data_rng_unchanged=True,
    actual_videomae_observations_counted=True,
    detector_length=384,
    packet_size=16,
)


model = dict(
    frame_selector=dict(
        multi_budget_exposure=dict(
            enabled=True,
            budgets=(256, 384, 512),
            probabilities={256: p256, 384: p384, 512: p512},
            total_updates=6000,
            seed=seed,
            detector_length=384,
            packet_size=16,
            evaluation_budget=evaluation_budget,
            evaluation_manifest=evaluation_manifest,
        ),
    ),
)


workflow = dict(
    training_profile="duca_h65_system_multibudget_exposure_candidate",
)


work_dir = f"exps/thumos/adatad/duca_h65_system_multibudget_exposure_candidate_seed{seed}"
