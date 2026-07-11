"""BSGS over GT - needed to decrypt products produced by ★_ct.

GT is the multiplicative target group of the pairing. The discrete-log
problem there reads:  given M ∈ GT, find k ∈ [0, max_value] with g^k = M
where g = e(G1, G2). Same √max_value memory/time profile as G1 BSGS;
the only difference is the per-step cost (FQ12 multiplication vs G1
addition), which is the dominant factor for performance on pure-Python
py_ecc and the main reason an `mcl`/`RELIC` backend gives such large
speedups.
"""

from __future__ import annotations

import math

from . import params


def _gt_key(g) -> bytes:
    """Canonical hashable key for an FQ12 element. py_ecc returns elements
    that always have .coeffs as a tuple of FQ; we serialise those."""
    out = bytearray()
    for c in g.coeffs:
        out += int(c).to_bytes(32, "big")
    return bytes(out)


class GTTable:
    """Baby-step giant-step table for solving g^k = M, k ∈ [0, max_value]."""

    __slots__ = ("max_value", "m", "table", "_neg_gm", "_g")

    def __init__(self, max_value: int) -> None:
        if max_value < 0:
            raise ValueError("max_value must be ≥ 0")
        self.max_value = max_value
        self.m = max(1, int(math.isqrt(max_value)) + 1)
        self._g = params.pairing(params.G2, params.G1)

        # Baby steps: g^i for i = 0..m
        self.table: dict[bytes, int] = {}
        cur = self._g ** 0
        for i in range(self.m + 1):
            self.table[_gt_key(cur)] = i
            cur = cur * self._g

        # Giant step: g^(-m) via Fermat
        self._neg_gm = self._g ** ((params.ORDER - self.m) % params.ORDER)

    def solve(self, M) -> int | None:
        cur = M
        for j in range(self.m + 1):
            low = self.table.get(_gt_key(cur))
            if low is not None:
                k = j * self.m + low
                if 0 <= k <= self.max_value:
                    return k
            cur = cur * self._neg_gm
        return None
