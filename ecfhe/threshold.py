"""Shamir t-of-n threshold decryption.

Mirrors the standard distributed-decryption setup referenced in EC_FHE
§2 ("Key rotation and threshold decryption protocols were implemented to
align with central bank governance requirements"). The construction
is the textbook Shamir split + Lagrange recombination of partial
decryptions inside G1, so the secret scalar s never reassembles in any
single location.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Sequence

from .ciphertext import Ciphertext
from .decrypt import BSGSTable
from .params import ORDER, add, multiply, neg


@dataclass(frozen=True)
class Share:
    index: int                       # 1-based; index 0 = the secret itself
    value: int                       # f(index) mod n


@dataclass(frozen=True)
class PartialDec:
    """One party's contribution:  share_i * C1  ∈ G1."""
    index: int
    point: object


# Polynomial split
def split(secret: int, n_parties: int, threshold: int) -> list[Share]:
    if not 1 <= threshold <= n_parties:
        raise ValueError("require 1 ≤ t ≤ n")
    if not 0 < secret < ORDER:
        raise ValueError("secret must be in (0, n)")
    coeffs = [secret] + [secrets.randbelow(ORDER) for _ in range(threshold - 1)]
    shares: list[Share] = []
    for i in range(1, n_parties + 1):
        v = 0
        for c in reversed(coeffs):
            v = (v * i + c) % ORDER
        shares.append(Share(index=i, value=v))
    return shares


def _lagrange_at_zero(indices: Sequence[int]) -> list[int]:
    out: list[int] = []
    for i in indices:
        num, den = 1, 1
        for j in indices:
            if j == i:
                continue
            num = (num * (-j)) % ORDER
            den = (den * (i - j)) % ORDER
        out.append((num * pow(den, -1, ORDER)) % ORDER)
    return out


# Threshold decrypt
def partial_decrypt(share: Share, ct: Ciphertext) -> PartialDec:
    return PartialDec(index=share.index, point=multiply(ct.C1, share.value))


def combine_partials(ct: Ciphertext, partials: Sequence[PartialDec],
                     bsgs: BSGSTable):
    """Combine ≥ t partials → recover (a, b) without ever assembling s.

    Returns Fraction(a, b)."""
    indices = [p.index for p in partials]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate share indices")
    lambdas = _lagrange_at_zero(indices)

    # Σ λ_i * (share_i * C1) = s * C1
    s_C1 = None
    for lam, pd in zip(lambdas, partials):
        term = multiply(pd.point, lam % ORDER)
        s_C1 = term if s_C1 is None else add(s_C1, term)

    # Now C2 − s*C1 → a*P  and  C3 − s*C1 → b*P
    aP = add(ct.C2, neg(s_C1))
    bP = add(ct.C3, neg(s_C1))
    a = bsgs.solve(aP)
    b = bsgs.solve(bP)
    if a is None or b is None:
        raise ValueError("threshold-decrypted plaintext outside BSGS range")
    if b == 0:
        raise ZeroDivisionError("denominator decrypted to 0")
    from fractions import Fraction
    return Fraction(a, b)
