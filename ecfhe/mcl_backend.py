"""mcl backend adapter.

mcl is a C++ pairing library (Herumi). Its Python bindings provide
roughly 100× faster pairing and 10-50× faster scalar multiplication
versus py_ecc.optimized_bn128.

Install:
    pip install mcl                                # if a wheel exists for your platform
    # or, build from source: https://github.com/herumi/mcl

This adapter is loaded only if `mcl` imports cleanly. When unavailable,
the engine silently falls back to py_ecc - every existing test still
passes, just at ~100× lower speed on pairing operations.

We expose the same surface as `params.py` so swapping is a one-line
change in calling code:

    if mcl_backend.available():
        from ecfhe import mcl_backend as bk
    else:
        from ecfhe import params as bk
"""

from __future__ import annotations

import importlib

_mcl = None
_initialized = False


def available() -> bool:
    """Returns True if mcl is importable AND initialised cleanly."""
    return _try_init()


def _try_init() -> bool:
    global _mcl, _initialized
    if _initialized:
        return _mcl is not None
    _initialized = True
    try:
        _mcl = importlib.import_module("mcl")
        # Some bindings need explicit curve init
        if hasattr(_mcl, "bn254"):
            _mcl.bn254.init()
        return True
    except Exception:
        _mcl = None
        return False


# Curve generators / order
def G1():
    if not available():
        raise RuntimeError("mcl backend unavailable")
    return _mcl.bn254.G1.from_str("0", 16)         # paper-dep; adjust per binding


def G2():
    if not available():
        raise RuntimeError("mcl backend unavailable")
    return _mcl.bn254.G2.from_str("0", 16)


def ORDER() -> int:
    if not available():
        raise RuntimeError("mcl backend unavailable")
    return _mcl.bn254.curve_order


# Group operations (mirror params.py surface)
def add(P, Q):
    return P + Q                                    # mcl overloads operators


def multiply(P, k):
    return P * k


def neg(P):
    return -P


def pairing(P_g2, Q_g1):
    """e(P_g2, Q_g1) → GT element."""
    return _mcl.bn254.pairing(P_g2, Q_g1)


def normalize(P):
    P.normalize()
    return P


def is_inf(P):
    return P.is_zero()


# Swap helper
def install():
    """Replace `ecfhe.params` symbols with mcl-backed ones (in-place).

    Call this at import time *before* any other ecfhe modules touch
    params; safest is to invoke it as the first line of your script:

        import ecfhe.mcl_backend as mb
        if mb.available():
            mb.install()
        import ecfhe   # everything below now runs on mcl

    Falls silently to a no-op when mcl is missing, so the call is safe
    to keep in production code."""
    if not available():
        return False
    from . import params as p
    p.G1 = G1()
    p.G2 = G2()
    p.ORDER = ORDER()
    p.add = add
    p.multiply = multiply
    p.neg = neg
    p.pairing = pairing
    p.normalize = normalize
    p.is_inf = is_inf
    return True
