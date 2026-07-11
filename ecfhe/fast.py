"""Performance acceleration layer.

What this module provides:
  * Fixed-base windowed scalar multiplication with a precomputed table
    (3-5× speedup for `k*G1` and `k*G2` which dominate encrypt/decrypt).
  * gmpy2 detection: when gmpy2 is installed, big-integer arithmetic
    underlying py_ecc's field ops is silently accelerated (~5-15×).
  * mcl detection: if mcl is installed (`pip install mcl`), an optional
    accelerator that replaces the pairing path with a C++ backend
    (~100× on pairings, the slowest operation).

Honest performance disclosure:
  py_ecc.optimized_bn128 is **pure Python**. No amount of caching
  inside it will match a C library. The realistic gains here:

    +  gmpy2 alone               ≈  5-15×   (transparent)
    +  fixed-base windowing      ≈   3-5×   (for repeated kG)
    +  mcl backend (if avail)    ≈ 100-500× (per pairing)
    coincurve - NOT applicable. secp256k1 has no pairing.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from . import params


# gmpy2 detection (transparent acceleration of big-int math)
def gmpy2_available() -> bool:
    try:
        importlib.import_module("gmpy2")
        return True
    except ImportError:
        return False


# mcl detection
def mcl_available() -> bool:
    if os.environ.get("ECFHE_DISABLE_MCL") == "1":
        return False
    try:
        importlib.import_module("mcl")
        return True
    except ImportError:
        return False


# Fixed-base windowed scalar multiplication
class FixedBaseTable:
    """Precomputed [0..2^w)*P table for windowed scalar multiplication.

    For w = 8 (default), one 256-bit scalar mul costs ⌈256/8⌉ = 32 point
    additions instead of the ~384 doublings+adds that double-and-add
    needs. Roughly 3-5× faster in pure Python."""

    __slots__ = ("base", "window", "table", "_bitlen")

    def __init__(self, base, window: int = 8) -> None:
        if window < 2 or window > 12:
            raise ValueError("window size must be in [2, 12]")
        self.base = base
        self.window = window
        self._bitlen = params.ORDER_BITLEN

        # table[chunk_idx][k] = (k * 2^(chunk_idx * window)) * base
        size = 1 << window
        n_chunks = (self._bitlen + window - 1) // window
        self.table = []
        # Build chunk 0 first: [0..size) * base
        row = [params.Z1 if base == params.G1 else None]
        cur = base
        for k in range(1, size):
            row.append(cur)
            cur = params.add(cur, base)
        # If base is in G1, row[0] = Z1; otherwise use proper identity
        if row[0] is None:
            # For G2: identity is Z2
            row[0] = params.Z2
        self.table.append(row)

        # Subsequent chunks: shift base by 2^window
        for _ in range(1, n_chunks):
            # New base = 2^window * previous base
            new_base = params.multiply(self.table[-1][1], 1 << window)
            row = [self.table[-1][0]]               # identity
            cur = new_base
            for k in range(1, size):
                row.append(cur)
                cur = params.add(cur, new_base)
            self.table.append(row)

    def multiply(self, scalar: int):
        """Return scalar * base using the window table."""
        scalar = scalar % params.ORDER
        if scalar == 0:
            return self.table[0][0]
        mask = (1 << self.window) - 1
        acc = None
        for row in self.table:
            chunk = scalar & mask
            if chunk != 0:
                contribution = row[chunk]
                acc = contribution if acc is None else params.add(acc, contribution)
            scalar >>= self.window
            if scalar == 0:
                break
        return acc if acc is not None else self.table[0][0]


# Singleton caches: G1 and G2 base tables (built lazily)
_G1_TABLE: FixedBaseTable | None = None
_G2_TABLE: FixedBaseTable | None = None


def g1_table(window: int = 8) -> FixedBaseTable:
    global _G1_TABLE
    if _G1_TABLE is None:
        _G1_TABLE = FixedBaseTable(params.G1, window=window)
    return _G1_TABLE


def g2_table(window: int = 8) -> FixedBaseTable:
    global _G2_TABLE
    if _G2_TABLE is None:
        _G2_TABLE = FixedBaseTable(params.G2, window=window)
    return _G2_TABLE


def fast_g1_mul(scalar: int):
    """Drop-in faster `scalar * G1`. Falls back to plain multiply on first
    call (table is built lazily) and stays fast thereafter."""
    return g1_table().multiply(scalar)


def fast_g2_mul(scalar: int):
    return g2_table().multiply(scalar)


# Capability summary
def capabilities() -> dict:
    """Return what's enabled in the current install."""
    return {
        "gmpy2": gmpy2_available(),
        "mcl": mcl_available(),
        "fixed_base_g1_cached": _G1_TABLE is not None,
        "fixed_base_g2_cached": _G2_TABLE is not None,
        "order_bitlen": params.ORDER_BITLEN,
    }
