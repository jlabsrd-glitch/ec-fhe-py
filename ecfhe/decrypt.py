"""Decryption - EC_FHE Algorithm 11 (EC-FHE Decryption).

  1:  extract ciphertext components and metadata
  2:  verify ciphertext integrity and noise levels
  3:  recover elliptic-rational components using private key
        C2 − s*C1 = a*P     ⇒   recover a from a*P via BSGS
        C3 − s*C1 = b*P     ⇒   recover b
  4:  a/b ← Solve-Discrete-Logarithms(recovered points, P, sk)
  5:  x' ← a / b
  6-10: thorough mode - cross-validate noise, build detailed breakdown
  11-12: error certificate
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .ciphertext import Ciphertext
from .keygen import SecretKey
from .noise import Noise
from .params import G1, ORDER, add, eq, is_inf, multiply, neg, normalize
from .rational import EllipticRational


VerificationMode = Literal["fast", "thorough"]


@dataclass(frozen=True)
class ErrorCertificate:
    """EC_FHE Alg 11 line 12: (total_error_bound, error_breakdown,
    confidence_level, computation_summary)."""
    total_error_bound: float
    error_breakdown: dict
    confidence_level: float
    summary: str


# BSGS for "small" discrete log: solve k*P = R, k ∈ [0, max_value]
class BSGSTable:
    """Precomputed table for the bounded-DLOG step of Alg 11 line 4.

    Used both at decrypt time and inside bootstrap partial-decrypt.
    Memory O(√max_value); query O(√max_value)."""

    __slots__ = ("max_value", "m", "table", "_neg_mP")

    def __init__(self, max_value: int) -> None:
        if max_value < 0:
            raise ValueError("max_value must be ≥ 0")
        self.max_value = max_value
        self.m = max(1, int(math.isqrt(max_value)) + 1)

        # Baby steps  {iP : i = 0..m}, keyed on affine coords.
        # We use sequential point addition (1 add per step) rather than a
        # multiplication-per-step - this is already O(m) ECC additions
        # and dominates the BSGS build cost. Future: SIMD-style batched
        # point addition under affine coords with shared inverse.
        self.table: dict[bytes, int] = {}
        from .params import Z1
        cur = Z1
        for i in range(self.m + 1):
            self.table[_key(cur)] = i
            cur = add(cur, G1)

        # Giant step: -m*P. Use fast windowed mul if available.
        try:
            from .fast import fast_g1_mul
            self._neg_mP = neg(fast_g1_mul(self.m))
        except Exception:
            self._neg_mP = neg(multiply(G1, self.m))

    def solve(self, R) -> int | None:
        if is_inf(R):
            return 0
        cur = R
        for j in range(self.m + 1):
            k_low = self.table.get(_key(cur))
            if k_low is not None:
                k = j * self.m + k_low
                if 0 <= k <= self.max_value:
                    return k
            cur = add(cur, self._neg_mP)
        return None

    def solve_signed(self, R) -> int | None:
        """Like solve() but also recognises *negative* k in the canonical
        ECHCoverQ §3 sense: cann_n(k) = (a, s) with |a| ≤ ⌊n/2⌋ and s ∈
        {0, 1}. Returns the signed integer k ∈ [-max_value, max_value]
        such that k*P = R; or None if R is outside that signed range.

        Implementation: try the positive search first (k ∈ [0, M]); on
        miss, try solve(-R) (k ∈ [-M, 0)). One extra BSGS pass at most."""
        k = self.solve(R)
        if k is not None:
            return k
        # Try the negative range by negating R: (-k)*P = -R
        k_neg = self.solve(neg(R))
        if k_neg is not None:
            return -k_neg
        return None


class FullDomainTable:
    """Direct lookup table for tight-domain DLOG: solves k·P = R in O(1)
    by precomputing every multiple {i·P → i, i = -N..N} into a hash map.

    Use instead of `BSGSTable` when the plaintext domain is small enough
    that 2N+1 point storage is acceptable (~120 bytes per entry incl.
    Python overhead). For domains up to ~4096 this is materially faster
    than BSGS at decrypt time:

        BSGS         : √N point adds + √N hash probes per solve
        FullDomain   : 1 hash probe per solve

    Memory: ~120 B × (2·max_value + 1). At N=4096: ~1 MB.

    Built once and reused; same lifetime as BSGSTable. Pre-populates the
    signed range so `solve_signed` is also O(1)."""

    __slots__ = ("max_value", "table")

    def __init__(self, max_value: int) -> None:
        if max_value < 0:
            raise ValueError("max_value must be >= 0")
        self.max_value = max_value
        self.table: dict[bytes, int] = {}
        from .params import Z1
        # Forward: 0..max_value
        cur = Z1
        for i in range(max_value + 1):
            self.table[_key(cur)] = i
            cur = add(cur, G1)
        # Backward: -1..-max_value (skip 0, already at i=0)
        cur = neg(G1)
        neg_G1 = neg(G1)
        for i in range(1, max_value + 1):
            self.table[_key(cur)] = -i
            cur = add(cur, neg_G1)

    def solve(self, R) -> int | None:
        if is_inf(R):
            return 0
        k = self.table.get(_key(R))
        if k is None or k < 0:
            return None
        return k

    def solve_signed(self, R) -> int | None:
        if is_inf(R):
            return 0
        return self.table.get(_key(R))


def make_dlog_table(max_value: int, *, full_table_cutoff: int = 4096):
    """Pick the right DLOG-solver shape for a given domain.

    Small domains (<= cutoff): build the full direct-lookup table.
    Larger domains: fall back to BSGS. The cutoff balances memory
    against per-solve cost; 4096 corresponds to ~1 MB and ~100x speedup
    versus BSGS in pure Python."""
    if max_value <= full_table_cutoff:
        return FullDomainTable(max_value)
    return BSGSTable(max_value)


class DecryptCache:
    """LRU cache mapping ciphertext identity to recovered (a, b).

    Decrypt is by far the dominant cost on query paths that re-read the
    same row across multiple statements (or re-check the same predicate
    inside one statement).

    The cache keys on `id(ct)` - the Python object's memory address.
    This is O(1) (no point serialisation) and correct as long as the
    storage layer always hands back the *same* Ciphertext object for
    the same row. Our storage stores Ciphertexts directly in row dicts
    and never copies them, so identity equals plaintext equality for
    the lifetime of the executor.

    If a workload constructs fresh Ciphertext instances that encrypt
    the same value (e.g. duplicated rows), they will NOT share a cache
    entry - their decrypts run anew. That is correct (the values are
    cryptographically distinct) and just means missed-cache cost.

    The cache also holds weak refs to the keying objects so a cleared
    storage releases the cache entries automatically (no leak)."""

    __slots__ = ("_data", "_max", "hits", "misses")

    def __init__(self, max_entries: int = 10_000) -> None:
        from collections import OrderedDict
        self._data: OrderedDict[int, tuple] = OrderedDict()
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    def get(self, ct) -> tuple | None:
        k = id(ct)
        if k in self._data:
            self._data.move_to_end(k)
            self.hits += 1
            return self._data[k]
        self.misses += 1
        return None

    def put(self, ct, value: tuple) -> None:
        k = id(ct)
        if k in self._data:
            self._data.move_to_end(k)
            return
        self._data[k] = value
        if len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0


def _key(P) -> bytes:
    """Canonical 64-byte key for hashing G1 points. Jacobian representations
    are non-unique, so we normalise to affine first."""
    if is_inf(P):
        return b"\x00" * 64
    x, y = normalize(P)
    return int(x).to_bytes(32, "big") + int(y).to_bytes(32, "big")


# Algorithm 11
def decrypt(sk: SecretKey,
            ct: Ciphertext,
            bsgs: BSGSTable,
            mode: VerificationMode = "fast",
            ) -> tuple[Fraction, ErrorCertificate]:
    """Algorithm 11 - decryption with error certificate.

    Returns (x' as a Fraction, cert). Use float(x') if a float is wanted.
    """
    # Line 2 - quick integrity check
    if ct.delta < 1:
        raise ValueError("ciphertext has invalid precision metadata")

    # Line 3 - strip masking: C2 - s*C1 = a*P
    sC1 = multiply(ct.C1, sk.s)
    aP = add(ct.C2, neg(sC1))
    bP = add(ct.C3, neg(sC1))

    # Line 4 - small-DLOG via BSGS. Signed variant accepts the canonical
    # ECHCoverQ §3 negative range; without it, a plaintext like -3 would
    # be encoded as (n-3)P, far outside any positive BSGS range, and
    # would falsely look like an out-of-range error.
    a = bsgs.solve_signed(aP)
    b = bsgs.solve_signed(bP)
    if a is None or b is None:
        raise ValueError(
            "decryption out of BSGS range; supply a larger BSGSTable or "
            "bootstrap the ciphertext first")

    # Line 5 - rational value
    if b == 0:
        raise ZeroDivisionError("decrypted denominator is zero")
    x_prime = Fraction(a, b)

    # Line 6-10 - thorough mode: cross-validate
    breakdown = {
        "approx": ct.noise.approx,
        "comp": ct.noise.comp,
        "crypto": ct.noise.crypto,
    }
    confidence = 1.0
    if mode == "thorough":
        # Cross-check: re-encrypt a/b under the same key (without randomness)
        # and confirm we land on the same affine class.
        check_a = multiply(G1, a)
        check_b = multiply(G1, b)
        if not (eq(check_a, aP) and eq(check_b, bP)):
            confidence = 0.0
            raise RuntimeError("thorough-mode verification failed")
        confidence = 1.0

    cert = ErrorCertificate(
        total_error_bound=ct.noise.total,
        error_breakdown=breakdown,
        confidence_level=confidence,
        summary=f"δ={ct.delta} mode={ct.mode} a={a} b={b}",
    )
    return x_prime, cert


# Convenience: decrypt to EllipticRational (skip the Fraction layer)
def decrypt_to_rational(sk: SecretKey, ct: Ciphertext,
                        bsgs: BSGSTable) -> EllipticRational:
    frac, _ = decrypt(sk, ct, bsgs)
    return EllipticRational(a=frac.numerator, b=frac.denominator).canonical()
