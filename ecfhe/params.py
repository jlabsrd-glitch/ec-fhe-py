"""Curve / pairing parameter selection - thin re-export over a pluggable
backend.

Reference: EC_FHE §10.1, ECHCoverQ §8.1.

Backend selection at import time via ECFHE_BACKEND env var:

    ECFHE_BACKEND=pyecc        (default; pure-Python py_ecc.optimized_bn128)
    ECFHE_BACKEND=mcl          (herumi mcl C++, ~100× faster pairings)

All algorithm modules (encrypt, decrypt, ★_ct, bootstrap, ...) reference
this module's symbols. Swapping a backend is therefore a one-knob
operation that needs zero changes elsewhere.
"""

from __future__ import annotations

from . import backends as _backends

_be = _backends.select()

# Re-exported group constants
G1 = _be.G1
G2 = _be.G2
Z1 = _be.Z1
Z2 = _be.Z2

ORDER: int = _be.ORDER                              # n in the papers
FIELD_MOD: int = _be.FIELD_MOD

FQ = _be.FQ
FQ2 = _be.FQ2
FQ12 = _be.FQ12

# Re-exported group operations
add = _be.add
multiply = _be.multiply
neg = _be.neg
pairing = _be.pairing
normalize = _be.normalize
is_inf = _be.is_inf

# Security-parameter bookkeeping (curve-independent)
ORDER_BITLEN: int = ORDER.bit_length()


def eq(P, Q) -> bool:
    """Group equality on Jacobian/projective points. Two representations
    of the same affine point may have different internal coords; the
    well-defined test is identity of the difference."""
    return is_inf(add(P, neg(Q)))


# GT identity (lazy, since computing one pairing is expensive)
_GT_ONE = None


def gt_one():
    """The identity in GT - computed once lazily and cached."""
    global _GT_ONE
    if _GT_ONE is None:
        _GT_ONE = pairing(G2, G1) ** 0
    return _GT_ONE


def max_precision_for_security(security_param_lambda: int) -> int:
    """Largest precision δ_max satisfying log10(n) > δ_max + λ."""
    import math
    return max(1, int(math.log10(ORDER)) - security_param_lambda)


def backend_name() -> str:
    """Name of the active backend (pyecc / mcl). Useful for diagnostics
    and benchmark labels."""
    return _backends.name()
