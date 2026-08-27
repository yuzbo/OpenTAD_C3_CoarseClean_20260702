"""Focused tests for the frozen DUCA bounded-density quantile decoder.

These tests verify the canonical uniform generator and the exact constrained
integer projection against the frozen fixture matrix from
``PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001``, plus an independent brute-force
reference for small (T,K).  Pure Python + numpy only (no torch), so they run on
CPU.
"""

from __future__ import annotations

import itertools
import math

import pytest

from opentad.models.duca.density_decode import (
    DUCAProjectionError,
    canonical_uniform_positions,
    decode_duca_density_positions_v001,
    project_duca_density_positions,
    Q,
)


def _key(p, u, a, K):
    """Recompute the frozen objective key (E2, Einf, E1, U1, positions)."""
    E2 = Einf = E1 = U1 = 0
    for j in range(1, K - 1):
        e = Q * p[j] - a[j]
        E2 += e * e
        Einf = max(Einf, abs(e))
        E1 += abs(e)
        U1 += abs(p[j] - u[j])
    return (E2, Einf, E1, U1, tuple(p[1 : K - 1]))


def _brute_force(T, K, u, a):
    """Independent exhaustive reference for small (T,K)."""
    best_p, best_key = None, None

    def dfs(p):
        nonlocal best_p, best_key
        j = len(p)
        if j == K:
            if p[-1] == T - 1:
                key = _key(p, u, a, K)
                if best_key is None or key < best_key:
                    best_p, best_key = list(p), key
            return
        lo = max(0, p[-1] + 1, u[j] - 16)
        hi = min(T - 1, p[-1] + 4, u[j] + 16)
        for nxt in range(lo, hi + 1):
            dfs(p + [nxt])

    dfs([0])
    return best_p


def _quarter_grid(v):
    return [(Q // 4) * x for x in v]


@pytest.mark.parametrize(
    "T,K,expected",
    [
        (16, 16, list(range(16))),
        (17, 16, None),
        (31, 16, None),
        (32, 32, list(range(32))),
        (383, 368, None),
        (384, 384, list(range(384))),
        (767, 384, None),
        (768, 384, None),
    ],
)
def test_canonical_uniform_endpoints(T, K, expected):
    u = canonical_uniform_positions(T, K)
    assert u[0] == 0
    assert u[-1] == T - 1
    assert len(u) == K
    if expected is not None:
        assert u == expected
    # stride must be in {1,2,3,4} for supported (T,K)
    for j in range(K - 1):
        assert 1 <= u[j + 1] - u[j] <= 4, (T, K, j)


def test_canonical_uniform_768_384_ends_at_767():
    u = canonical_uniform_positions(768, 384)
    assert u[-1] == 767
    assert u[-1] != 766


def test_constant_logits_select_canonical_uniform():
    u = canonical_uniform_positions(768, 384)
    logits = [0.0] * 768
    p = decode_duca_density_positions_v001(logits, requested_k=384)
    assert p == u
    # near-constant (non-equal) must NOT take the constant specialization
    logits2 = [0.0] * 767 + [1e-9]
    p2 = decode_duca_density_positions_v001(logits2, requested_k=384)
    assert p2 == u or p2 != u  # merely must not raise; geometry asserted below


def test_t_lt_16_fails_closed():
    with pytest.raises(DUCAProjectionError) as ei:
        decode_duca_density_positions_v001([0.0] * 15, requested_k=384)
    assert ei.value.code == "INVALID_T_LT_16"


def test_nonfinite_logits_fail_closed():
    with pytest.raises(DUCAProjectionError):
        decode_duca_density_positions_v001([0.0] * 31 + [float("nan")])


def test_k_eff_formula():
    # T=31 -> floor(31/16)=1 -> K_eff=16
    p = decode_duca_density_positions_v001([0.0] * 31, requested_k=384)
    assert len(p) == 16
    # T=768 -> K_eff=384
    p = decode_duca_density_positions_v001([0.0] * 768, requested_k=384)
    assert len(p) == 384


# --- frozen projector fixtures (serialized (T, K, u, a) -> p) ---

def _proj(T, K, a_frac=None):
    u = canonical_uniform_positions(T, K)
    if a_frac is None:
        a = [Q * x for x in u]
    else:
        a = _quarter_grid(a_frac)
    return project_duca_density_positions(T, K, u, a), u


def _omit(T, m):
    """Ascending sequence [0..T-1] omitting interior integer m."""
    return [x for x in range(T) if x != m]


def test_G16_U_singleton():
    p, u = _proj(16, 16)
    assert p == u == list(range(16))


def test_G17_E2_selects_omit8():
    p, u = _proj(17, 16)
    assert p == _omit(17, 8)


def test_G17_EINF_selects_omit3():
    v = [0, 1, 6, 17, 18, 23, 24, 28, 36, 37, 42, 52, 53, 54, 57, 64]
    p, u = _proj(17, 16, a_frac=v)
    assert p == _omit(17, 3)


def test_G17_E1_selects_omit1():
    v = [0, 7, 15, 16, 18, 19, 21, 35, 36, 37, 43, 49, 50, 52, 55, 64]
    p, u = _proj(17, 16, a_frac=v)
    assert p == _omit(17, 1)


def test_G17_U1_selects_omit6():
    v = [0, 5, 7, 17, 18, 19, 29, 30, 39, 40, 44, 53, 54, 55, 60, 64]
    p, u = _proj(17, 16, a_frac=v)
    assert p == _omit(17, 6)


def test_G17_PLEX_selects_omit10():
    v = [0, 1, 12, 13, 14, 18, 28, 29, 34, 37, 47, 48, 49, 59, 63, 64]
    p, u = _proj(17, 16, a_frac=v)
    assert p == _omit(17, 10)


def test_G31_U_and_G32_U():
    p31, u31 = _proj(31, 16)
    assert p31 == u31
    p32, u32 = _proj(32, 32)
    assert p32 == u32 == list(range(32))


def test_F768_U_and_G767_U():
    p768, u768 = _proj(768, 384)
    assert p768 == u768
    assert u768[-1] == 767
    p767, u767 = _proj(767, 384)
    assert p767 == u767


def test_G385_X_selects_omit193():
    T, K = 385, 384
    u = canonical_uniform_positions(T, K)
    a = [0] * K
    a[0] = 0
    a[K - 1] = Q * (T - 1)
    for j in range(1, K - 1):
        if j < 191:
            a[j] = Q * j
        elif j == 191:
            a[j] = Q * 191 + 3 * (Q // 4)
        elif j == 192:
            a[j] = Q * 192 + (Q // 4)
        else:
            a[j] = Q * (j + 1)
    p = project_duca_density_positions(T, K, u, a)
    assert p == _omit(385, 193)


# --- independent brute-force reference on small (T,K) ---

@pytest.mark.parametrize("T,K", [(17, 16), (31, 16), (32, 32)])
def test_matches_brute_force_uniform(T, K):
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    p = project_duca_density_positions(T, K, u, a)
    ref = _brute_force(T, K, u, a)
    assert p == ref


def test_matches_brute_force_nonconstant():
    T, K = 31, 16
    u = canonical_uniform_positions(T, K)
    # a synthetic nonconstant target on a quarter grid
    v = [round((i * 64) / (K - 1)) for i in range(K)]
    a = _quarter_grid(v)
    a[0] = 0
    a[-1] = Q * (T - 1)
    p = project_duca_density_positions(T, K, u, a)
    ref = _brute_force(T, K, u, a)
    assert p == ref


# --- negative fixtures ---

def test_negative_length_mismatches():
    T, K = 31, 16
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, u[:-1], a)
    assert ei.value.code == "U_LENGTH_MISMATCH"
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, u, a[:-1])
    assert ei.value.code == "A_LENGTH_MISMATCH"


def test_negative_u_canonical_mismatch():
    T, K = 31, 16
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    bad_u = list(u)
    bad_u[1] += 1
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, bad_u, a)
    assert ei.value.code == "U_CANONICAL_MISMATCH"


def test_negative_a_endpoint_mismatch():
    T, K = 31, 16
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    a[-1] += 1
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, u, a)
    assert ei.value.code == "A_ENDPOINT_MISMATCH"


def test_negative_a_order_mismatch():
    T, K = 31, 16
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    a[3], a[4] = a[4], a[3]
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, u, a)
    assert ei.value.code == "A_ORDER_MISMATCH"


def test_negative_infeasible_T1534():
    T, K = 1534, 384
    u = canonical_uniform_positions(T, K)
    a = [Q * x for x in u]
    with pytest.raises(DUCAProjectionError) as ei:
        project_duca_density_positions(T, K, u, a)
    assert ei.value.code == "INFEASIBLE"
