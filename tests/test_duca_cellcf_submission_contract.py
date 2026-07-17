from __future__ import annotations

import pytest

from tools.bata.duca_cellcf_submission_contract import (
    JOB_ORDER,
    TRAINING_PROFILES,
    expected_job_name,
)


COMMIT = "abcdef0123456789abcdef0123456789abcdef01"


@pytest.mark.parametrize("profile", TRAINING_PROFILES)
@pytest.mark.parametrize("job_key", JOB_ORDER)
def test_all_profile_job_names_are_deterministic_and_unambiguous(
    profile: str,
    job_key: str,
) -> None:
    name = expected_job_name(profile, job_key, 3, COMMIT)

    assert name.startswith(f"cellcf-{profile}-")
    assert name.endswith("-s3-abcdef0")
    assert len(name) < 128


def test_transition_key_uses_stable_short_suffix() -> None:
    assert (
        expected_job_name("official60", "transition_beta0", 0, COMMIT)
        == "cellcf-official60-transition-s0-abcdef0"
    )


@pytest.mark.parametrize(
    "profile,job_key,seed,commit",
    (
        ("unknown", "uniform", 0, COMMIT),
        ("official60", "unknown", 0, COMMIT),
        ("official60", "uniform", -1, COMMIT),
        ("official60", "uniform", 0, "short"),
    ),
)
def test_invalid_job_name_inputs_fail_closed(
    profile: str,
    job_key: str,
    seed: int,
    commit: str,
) -> None:
    with pytest.raises(ValueError):
        expected_job_name(profile, job_key, seed, commit)
