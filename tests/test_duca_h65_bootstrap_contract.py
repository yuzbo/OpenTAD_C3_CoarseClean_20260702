import numpy as np
import pytest

from tools.bata.bootstrap_duca_h65_official_map import exact_interval, seed_from_nonce


def test_pcg64_seed_is_namespaced_deterministic_and_case_sensitive():
    first, first_digest = seed_from_nonce("nonce", "final-ema-on-vs-gate-zero")
    second, second_digest = seed_from_nonce("nonce", "final-ema-on-vs-gate-zero")
    changed, changed_digest = seed_from_nonce("nonce", "final-on-vs-gate-zero")
    assert first == second
    assert first_digest == second_digest
    assert first != changed
    assert first_digest != changed_digest
    assert first == int.from_bytes(
        __import__("hashlib").sha256(b"nonce\nfinal-ema-on-vs-gate-zero").digest()[:8],
        byteorder="big",
        signed=False,
    )
    assert np.random.Generator(np.random.PCG64(first)).integers(0, 100, 8).tolist() == np.random.Generator(
        np.random.PCG64(second)
    ).integers(0, 100, 8).tolist()


def test_exact_interval_uses_frozen_one_based_order_statistics():
    values = list(range(10000, 0, -1))
    lower, upper = exact_interval(values, lower_rank=250, upper_rank=9750)
    assert lower == 250.0
    assert upper == 9750.0
    with pytest.raises(ValueError, match="outside"):
        exact_interval(values, lower_rank=0, upper_rank=9750)
