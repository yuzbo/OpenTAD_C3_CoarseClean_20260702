import copy

import pytest

from tools.bata.continuous_roi_s2_v3_full200_compute_train import (
    RECOVERY_SCHEMA,
    REQUIRED_IDENTITY_HASHES,
    validate_epoch_sampler_state,
    validate_recovery_payload,
)
from tools.bata.continuous_roi_s2_v3_full200_compute import PROTOCOL_ID


def _valid_payload():
    digest = "a" * 64
    return {
        "schema_version": RECOVERY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "model_state_dict": {},
        "ema_state_dict": {},
        "ema_update_counter": 500,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "grad_scaler_state_dict": {},
        "next_epoch_index": 5,
        "accepted_successful_updates": 500,
        "successful_updates_per_completed_epoch": [100] * 5,
        "next_epoch_identity_order": [f"train_{index:03d}" for index in range(200)],
        "distributed_sampler_state": {
            "epoch": 5,
            "cursor": 0,
            "world_size": 2,
            "rank_shards": [list(range(0, 200, 2)), list(range(1, 200, 2))],
        },
        "rank_rng_states": [
            {
                "rank": rank,
                "dataloader_generator_state": f"generator{rank}".encode(),
                "python_rng_state": (3, (), None),
                "numpy_rng_state": ("MT19937", [], 0, 0, 0.0),
                "torch_cpu_rng_state": f"cpu{rank}".encode(),
                "torch_cuda_rng_state": f"cuda{rank}".encode(),
                "augmentation_rng_streams": {"view": f"view{rank}".encode()},
            }
            for rank in range(2)
        ],
        "world_size": 2,
        "local_batch_size": 1,
        "global_batch_size": 2,
        "identity_hashes": {key: digest for key in REQUIRED_IDENTITY_HASHES},
        "completed_sample_order_trace_sha256": digest,
        "discarded_preemption_steps": 0,
        "zero_pending_gradient": True,
    }


def test_recovery_payload_requires_complete_epoch_boundary_state():
    validate_recovery_payload(_valid_payload())
    payload = _valid_payload()
    payload.pop("grad_scaler_state_dict")
    with pytest.raises(ValueError, match="fields changed"):
        validate_recovery_payload(payload)


def test_recovery_payload_rejects_noncertified_or_mismatched_progress():
    payload = _valid_payload()
    payload["accepted_successful_updates"] = 501
    with pytest.raises(ValueError, match="certified"):
        validate_recovery_payload(payload)

    payload = copy.deepcopy(_valid_payload())
    payload["distributed_sampler_state"]["cursor"] = 1
    with pytest.raises(ValueError, match="sampler"):
        validate_recovery_payload(payload)

    payload = _valid_payload()
    payload["ema_update_counter"] = 499
    with pytest.raises(ValueError, match="EMA"):
        validate_recovery_payload(payload)


def test_epoch_sampler_requires_exactly_once_full200_population():
    identities = [f"train_{index:03d}" for index in range(200)]
    state = {
        "epoch": 0,
        "cursor": 0,
        "world_size": 2,
        "rank_shards": [list(range(0, 200, 2)), list(range(1, 200, 2))],
    }
    assert validate_epoch_sampler_state(
        state, expected_identities=identities
    ) == tuple(identities)
    state = copy.deepcopy(state)
    state["rank_shards"][1][-1] = 0
    with pytest.raises(ValueError, match="every training identity"):
        validate_epoch_sampler_state(state, expected_identities=identities)
