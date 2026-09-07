"""Deterministic bounded-density quantile acquisition decoder.

This module implements the frozen
``DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`` hard decoder and the
exact constrained integer projection policy frozen by
``PRO_P0_PROJECTION_POLICY-v001`` / ``PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001``.

It is deliberately pure Python integer arithmetic (plus ``fractions`` for the
exact binary64 half-up fixed-point conversion).  No torch, no GPU, no detector,
no loss, no NMS, no evaluator and no data access is required: the decoder maps a
finite per-time logit vector to an exact-K integer position sequence.

Contract summary (authoritative definitions live in the Pro decisions):

* ``T`` is the valid-prefix length; ``K = K_eff = min(384, 16*floor(T/16))``;
  ``T < 16`` fails closed as ``INVALID_T_LT_16``.
* The canonical uniform reference is integer-half-up endpoint arithmetic::

      u_j = floor((2*j*(T-1) + (K-1)) / (2*(K-1))),  j = 0..K-1

  For ``T=768, K=384`` this ends at ``767`` (never ``766``).
* Exactly-equal constant logits select the canonical uniform vector verbatim.
* Otherwise: ``rho_t = 1e-6 + softplus(logit_t)``, trapezoidal masses
  ``m_t = (rho_t + rho_{t+1})/2``, cumulative ``A``, endpoint-inclusive
  inverse-CDF targets ``h_j = j*M/(K-1)`` -> ``x_j``, fixed-point
  ``a_j = floor(Q*x_j + 1/2)`` with ``Q = 2**20``, ``a_0 = 0``,
  ``a_{K-1} = Q*(T-1)``.
* The selected sequence ``p`` is the unique feasible sequence minimizing the
  exact lexicographic key ``(E2, E_inf, E1, U1, p_1, ..., p_{K-2})`` over the
  feasible set ``F(T,K)``::

      p_0 = 0, p_{K-1} = T-1,
      p_{j+1} - p_j in {1,2,3,4},
      |p_j - u_j| <= 16.

  with ``e_j = Q*p_j - a_j``,
  ``E2 = sum e_j^2``, ``E_inf = max |e_j|``, ``E1 = sum |e_j|``,
  ``U1 = sum |p_j - u_j|`` over internal ``j = 1..K-2``.

This module is the *authoritative production implementation* of the decoder and
projector; a test-only independent reference lives in the focused test suite.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "DUCAProjectionError",
    "canonical_uniform_positions",
    "decode_duca_density_positions_v001",
    "project_duca_density_positions",
]

Q = 1 << 20  # 2**20 fixed-point scale
SOFTPLUS_FLOOR = 1.0e-6


class DUCAProjectionError(RuntimeError):
    """Typed fail-closed projection error.

    ``code`` carries one of the frozen failure codes so callers can react
    without parsing the message.  A raised error invalidates the sample/run
    identity; it must never be repaired into uniform or a legacy selector.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _softplus(x: float) -> float:
    # Numerically stable softplus; all inputs must already be finite.
    if not math.isfinite(x):
        raise DUCAProjectionError("NON_FINITE_INPUT", "density logits must be finite")
    if x > 20.0:
        return x
    return math.log1p(math.exp(x))


def canonical_uniform_positions(T: int, K: int) -> List[int]:
    """Canonical integer-half-up endpoint-inclusive uniform positions.

    Returns ``[u_0, ..., u_{K-1}]`` with ``u_0 = 0`` and ``u_{K-1} = T-1``.
    """
    if T < 16:
        raise DUCAProjectionError("INVALID_T_LT_16", f"T={T} < 16")
    if K <= 1:
        raise DUCAProjectionError("K_EFF_MISMATCH", f"K={K} must be >= 2")
    denom = 2 * (K - 1)
    num_base = 2 * (T - 1)
    c = K - 1
    u = [(num_base * j + c) // denom for j in range(K)]
    if u[0] != 0 or u[-1] != T - 1:
        raise DUCAProjectionError("U_CANONICAL_MISMATCH", "canonical uniform endpoints wrong")
    return u


def _fixed_point_half_up(x: float) -> int:
    """Exact nonnegative half-up fixed-point conversion of a binary64 value.

    ``a = floor(Q*x + 1/2)`` computed exactly from the binary64 bit pattern via
    ``Fraction``, not host-language ``round`` (which is banker's rounding).
    """
    if not math.isfinite(x) or x < 0.0:
        raise DUCAProjectionError("NON_FINITE_INPUT", "inverse-CDF target invalid")
    return math.floor(Fraction.from_float(x) * Q + Fraction(1, 2))


def _inverse_cdf_fixed_point(logits: Sequence[float], K: int) -> List[int]:
    """Compute endpoint-inclusive inverse-CDF targets and return fixed-point a."""
    T = len(logits)
    if T < 16:
        raise DUCAProjectionError("INVALID_T_LT_16", f"T={T} < 16")

    rho = [SOFTPLUS_FLOOR + _softplus(float(v)) for v in logits]
    # trapezoidal interval masses m_t = (rho_t + rho_{t+1}) / 2
    m = [(rho[t] + rho[t + 1]) / 2.0 for t in range(T - 1)]
    A = [0.0]
    for mass in m:
        A.append(A[-1] + mass)
    M = A[-1]
    if not (M > 0.0 and math.isfinite(M)):
        raise DUCAProjectionError("NON_FINITE_INPUT", "density mass must be positive and finite")

    a = [0] * K
    a[0] = 0
    a[K - 1] = Q * (T - 1)
    for j in range(1, K - 1):
        h = j * M / (K - 1)
        # find r with A[r] < h < A[r+1], or h == A[r] exactly
        r = _cumulative_index(A, h)
        if A[r] == h:
            x = float(r)
        else:
            x = r + (h - A[r]) / m[r]
        a[j] = _fixed_point_half_up(x)
    return a


def _cumulative_index(A: Sequence[float], h: float) -> int:
    """Return the index r such that A[r] <= h <= A[r+1] (with A strictly inc)."""
    lo, hi = 0, len(A) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if A[mid] <= h:
            lo = mid
        else:
            hi = mid - 1
    return lo


def project_duca_density_positions(T: int, K: int, u: Sequence[int], a: Sequence[int]) -> List[int]:
    """Exact constrained integer projection: ``(T, K, u, a) -> p``.

    ``u`` is the canonical uniform vector, ``a`` the fixed-point inverse-CDF
    target vector.  Returns the unique feasible sequence minimizing the frozen
    lexicographic key ``(E2, E_inf, E1, U1, p_1, ..., p_{K-2})``.
    """
    if T < 16:
        raise DUCAProjectionError("INVALID_T_LT_16", f"T={T} < 16")
    if K <= 1:
        raise DUCAProjectionError("K_EFF_MISMATCH", f"K={K} must be >= 2")
    if len(u) != K:
        raise DUCAProjectionError("U_LENGTH_MISMATCH", f"len(u)={len(u)} != K={K}")
    if len(a) != K:
        raise DUCAProjectionError("A_LENGTH_MISMATCH", f"len(a)={len(a)} != K={K}")
    expected_u = canonical_uniform_positions(T, K)
    if list(u) != expected_u:
        raise DUCAProjectionError("U_CANONICAL_MISMATCH", "u is not the canonical uniform")
    if a[0] != 0 or a[-1] != Q * (T - 1):
        raise DUCAProjectionError("A_ENDPOINT_MISMATCH", "a endpoints must be 0 and Q*(T-1)")
    for j in range(1, K):
        if a[j] < a[j - 1]:
            raise DUCAProjectionError("A_ORDER_MISMATCH", "a must be non-decreasing")

    # Canonical uniform feasibility witness (protocol precondition).
    _assert_uniform_feasible(T, K, u)

    # e_j = Q * p_j - a_j for internal j = 1..K-2.
    def err(pj: int, j: int) -> int:
        return Q * pj - a[j]

    # DP over layers.  State key: (E2, Einf, E1, U1, prefix_tuple).
    # layer[j] maps position p_j -> best (E2, Einf, E1, U1, prefix).
    start = (0, 0, 0, 0, ())
    layer_prev = {0: start}
    for j in range(1, K - 1):
        layer_cur = {}
        for r in sorted(layer_prev):
            E2p, Einfp, E1p, U1p, prefix = layer_prev[r]
            low = max(0, r + 1, u[j] - 16, T - 1 - 4 * (K - 1 - j))
            high = min(T - 1, r + 4, u[j] + 16, T - 1 - (K - 1 - j))
            for p in range(low, high + 1):
                e = err(p, j)
                e2 = E2p + e * e
                einf = Einfp if Einfp >= abs(e) else abs(e)
                e1 = E1p + abs(e)
                u1 = U1p + abs(p - u[j])
                key = (e2, einf, e1, u1, prefix + (p,))
                cur = layer_cur.get(p)
                if cur is None or key < cur:
                    layer_cur[p] = key
        if not layer_cur:
            raise DUCAProjectionError("INFEASIBLE", f"feasible set empty at layer {j}")
        layer_prev = layer_cur

    # Terminal layer: p_{K-1} = T-1 with stride in {1,2,3,4} from p_{K-2}.
    best = None
    for r in sorted(layer_prev):
        E2p, Einfp, E1p, U1p, prefix = layer_prev[r]
        if 1 <= (T - 1 - r) <= 4:
            key = (E2p, Einfp, E1p, U1p, prefix)
            if best is None or key < best[0]:
                best = (key, prefix)
    if best is None:
        raise DUCAProjectionError("INFEASIBLE", "no feasible sequence reaches endpoint")

    _, prefix = best
    p = [0] + list(prefix) + [T - 1]
    _assert_feasible(T, K, u, p)
    return p


def _assert_uniform_feasible(T: int, K: int, u: Sequence[int]) -> None:
    # The canonical uniform vector must itself witness non-emptiness of F(T,K).
    # If its stride already exceeds 4, the feasible set is empty (the stride-4
    # bound cannot be met with this (T,K)); this is INFEASIBLE, not a wrong-u
    # error (u is still the correctly computed canonical vector).
    for j in range(K - 1):
        if not (1 <= u[j + 1] - u[j] <= 4):
            raise DUCAProjectionError("INFEASIBLE", "canonical uniform cannot witness a feasible set")
    if u[0] != 0 or u[-1] != T - 1:
        raise DUCAProjectionError("U_CANONICAL_MISMATCH", "uniform endpoints wrong")


def _assert_feasible(T: int, K: int, u: Sequence[int], p: Sequence[int]) -> None:
    if len(p) != K:
        raise DUCAProjectionError("CERTIFICATE_REJECTED", "p length mismatch")
    if p[0] != 0 or p[-1] != T - 1:
        raise DUCAProjectionError("CERTIFICATE_REJECTED", "p endpoints wrong")
    for j in range(K):
        if not (0 <= p[j] <= T - 1):
            raise DUCAProjectionError("CERTIFICATE_REJECTED", "p out of range")
        if abs(p[j] - u[j]) > 16:
            raise DUCAProjectionError("CERTIFICATE_REJECTED", "displacement exceeds 16")
        if j > 0 and not (1 <= p[j] - p[j - 1] <= 4):
            raise DUCAProjectionError("CERTIFICATE_REJECTED", "stride outside {1..4}")


def decode_duca_density_positions_v001(
    density_logits_valid: Sequence[float],
    requested_k: int = 384,
) -> List[int]:
    """Decode a valid-prefix density-logit vector into an exact-K position list.

    ``density_logits_valid`` is the finite per-time logit over the valid prefix
    (length ``T``).  Returns the selected integer positions ``p`` of length
    ``K_eff``.  Fails closed with a typed :class:`DUCAProjectionError`.
    """
    T = len(density_logits_valid)
    if T < 16:
        raise DUCAProjectionError("INVALID_T_LT_16", f"T={T} < 16")

    K = min(requested_k, 16 * (T // 16))
    if K != _declared_k(requested_k, T):
        raise DUCAProjectionError("K_EFF_MISMATCH", "effective K mismatch")

    logits = [float(v) for v in density_logits_valid]
    for v in logits:
        if not math.isfinite(v):
            raise DUCAProjectionError("NON_FINITE_INPUT", "density logits must be finite")

    u = canonical_uniform_positions(T, K)

    # Exactly-equal constant logits -> canonical uniform specialization.
    if all(v == logits[0] for v in logits):
        return list(u)

    a = _inverse_cdf_fixed_point(logits, K)
    return project_duca_density_positions(T, K, u, a)


def _declared_k(requested_k: int, T: int) -> int:
    return min(requested_k, 16 * (T // 16))
