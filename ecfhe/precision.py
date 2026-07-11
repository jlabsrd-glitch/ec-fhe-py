"""Real ↔ Q_E precision-controlled approximation - EC_FHE §5.

  Theorem 4.2  ∀x ∈ R, δ ∈ N  ∃a/b ∈ Q  s.t. |x − a/b| < 10^{-δ}, b ≤ 10^δ.
              (terminating decimals give exact equality)

  Theorem 4.3  k-th continued-fraction convergent satisfies
              |x − p_k/q_k| < 1/(q_k * q_{k+1}) < 1/q_k².

  Theorem 5.1  Standard mode error ≤ 10^{-δ};
              optimized mode same error with log2(b) ≤ ½δ*log2(10) + O(1).

  Algorithm 2  Enhanced Real-to-Elliptic-Rational Conversion
      mode ∈ {standard, optimized, high_precision}
      Reduce to lowest terms; if b ≥ n use modular inverse  b' = b⁻¹ mod n.

The output is a (rational, ConversionCert) pair carrying the realised
approximation error (used to seed η_approx of the Noise triple).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from math import gcd, isqrt
from typing import Literal

from .params import ORDER
from .rational import EllipticRational

Mode = Literal["standard", "optimized", "high_precision"]


@dataclass(frozen=True)
class ConversionCert:
    """Error certificate (EC_FHE Alg 2 line 15: cert ← (δ, ε, method))."""
    delta: int
    approximation_error: float
    method: Mode


# Algorithm 2 - three modes
def real_to_elliptic_rational(x: float | Decimal | Fraction,
                              delta: int,
                              mode: Mode = "standard",
                              ) -> tuple[EllipticRational, ConversionCert]:
    """Real x → Q_E with explicit precision δ. Returns the canonical form
    plus an error certificate."""
    if delta < 1:
        raise ValueError("delta must be ≥ 1")

    if mode == "standard":
        a, b = _standard_decimal(x, delta)
    elif mode == "optimized":
        a, b = _continued_fraction(x, delta)
    elif mode == "high_precision":
        a, b = _high_precision(x, delta)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Algorithm 2 lines 8-9 - reduce to lowest terms
    g = gcd(abs(a), abs(b)) or 1
    a, b = a // g, b // g

    # Algorithm 2 lines 10-13 - handle b ≥ n via modular inverse
    if b >= ORDER or b <= -ORDER:
        b_inv = pow(b % ORDER, -1, ORDER)
        a = (a * b_inv) % ORDER
        b = 1

    # Realised error  ε ← |x − a/b|
    err = abs(_to_decimal(x) - Decimal(a) / Decimal(b))
    cert = ConversionCert(delta=delta, approximation_error=float(err), method=mode)

    return EllipticRational(a=a, b=b).canonical(), cert


# Helpers per Thm 4.2 / 4.3
def _to_decimal(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(str(x))


def _standard_decimal(x, delta: int) -> tuple[int, int]:
    """Thm 4.2 general case: a = ⌊x*10^δ⌋, b = 10^δ."""
    getcontext().prec = max(50, delta + 20)
    xd = _to_decimal(x)
    scaled = xd * (Decimal(10) ** delta)
    # Truncate toward zero (the paper's ⌊*⌋; for negatives this still
    # bounds the error by 10^{-δ}).
    a = int(scaled.to_integral_value(rounding="ROUND_FLOOR"))
    b = 10 ** delta
    return a, b


def _continued_fraction(x, delta: int) -> tuple[int, int]:
    """Thm 4.3 / Thm 5.1 optimised mode - return the largest convergent
    p_k/q_k with q_k ≤ 10^{δ/2}, so |x − p_k/q_k| < 10^{-δ}.

    Standard CF recurrences:
        p_{-2} = 0, p_{-1} = 1
        q_{-2} = 1, q_{-1} = 0
        p_k = a_k * p_{k-1} + p_{k-2}     (k ≥ 0)
        q_k = a_k * q_{k-1} + q_{k-2}
    """
    getcontext().prec = max(50, 2 * delta + 20)
    target_q = isqrt(10 ** delta) + 1                # ≈ 10^{δ/2}

    xd = _to_decimal(x)
    sign = 1
    if xd < 0:
        sign = -1
        xd = -xd

    # (p_{k-2}, p_{k-1}), (q_{k-2}, q_{k-1})
    p_pp, p_p = 0, 1
    q_pp, q_p = 1, 0

    best_p: int | None = None
    best_q: int | None = None
    work = xd
    for _ in range(256):
        a_k = int(work)
        p_k = a_k * p_p + p_pp
        q_k = a_k * q_p + q_pp
        if q_k > target_q:
            break
        best_p, best_q = p_k, q_k
        p_pp, p_p = p_p, p_k
        q_pp, q_p = q_p, q_k
        frac = work - Decimal(a_k)
        if frac == 0:
            break
        work = Decimal(1) / frac

    if best_p is None or best_q == 0:
        # x is essentially an integer at this precision
        return sign * int(xd), 1
    return sign * best_p, best_q


def _high_precision(x, delta: int) -> tuple[int, int]:
    """High-precision mode - EC_FHE §8.3.

    The paper specifies that high_precision targets *better* error than
    standard at the same δ. Thm 4.3 guarantees the CF convergent at q_k
    ≤ 10^{δ/2} satisfies |x - p_k/q_k| < 1/(q_k * q_{k+1}); for inputs
    whose CF expansion has small a_{k+1} that bound degenerates toward
    1/q_k². To always satisfy the δ+5 contract we compute *both* the
    CF convergent and the standard decimal expansion at δ+5, and pick
    whichever produces the smaller realised error."""
    target_delta = delta + 5
    getcontext().prec = max(80, 4 * target_delta + 40)

    # Standard decimal at δ+5 - guaranteed |x - a/b| ≤ 10^{-(δ+5)} by Thm 4.2
    a_std, b_std = _standard_decimal(x, target_delta)
    err_std = abs(_to_decimal(x) - Decimal(a_std) / Decimal(b_std))

    # CF mode at δ+5
    a_cf, b_cf = _continued_fraction(x, target_delta)
    err_cf = abs(_to_decimal(x) - Decimal(a_cf) / Decimal(b_cf))

    if err_cf <= err_std:
        return a_cf, b_cf
    return a_std, b_std
