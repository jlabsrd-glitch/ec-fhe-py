"""Pluggable curve / pairing backends.

A backend is a module exposing the following symbols (mirroring
py_ecc.optimized_bn128 surface):

    G1, G2                # Jacobian generators
    Z1, Z2                # group identities
    ORDER                 # prime curve order  (int)
    FIELD_MOD             # base field characteristic
    FQ, FQ2, FQ12         # field element types

    add(P, Q)             # group law
    multiply(P, k)        # scalar mul
    neg(P)
    pairing(P_g2, Q_g1)   # → GT element
    normalize(P)          # Jacobian → affine
    is_inf(P)

Selection happens at *import time* of `ecfhe.params`:

    ECFHE_BACKEND=pyecc   (default, pure-Python via py_ecc)
    ECFHE_BACKEND=mcl     (herumi mcl C++ bindings, Linux/macOS)

This protocol-level abstraction means swapping a backend never touches
the algorithm code (encrypt/decrypt/★_ct/bootstrap)."""

from __future__ import annotations

import importlib
import os
from typing import Any


# Selection
_DEFAULT = "pyecc"


def _resolve(name: str) -> Any:
    """Resolve a backend name to its module. On import failure for a
    non-default name we fall back to pyecc and emit one warning."""
    if name == "pyecc":
        return importlib.import_module(".pyecc", package=__package__)
    if name == "mcl":
        try:
            return importlib.import_module(".mcl", package=__package__)
        except Exception as e:                              # pragma: no cover
            import warnings
            warnings.warn(
                f"ECFHE_BACKEND=mcl requested but mcl is unavailable ({e}); "
                "falling back to pyecc. Install mcl with `pip install mcl` "
                "(Linux/macOS only) or build from "
                "https://github.com/herumi/mcl.",
                RuntimeWarning,
            )
            return importlib.import_module(".pyecc", package=__package__)
    raise ValueError(f"unknown backend {name!r}; expected 'pyecc' or 'mcl'")


_active_name = _DEFAULT


def select() -> Any:
    """Return the configured backend module. Also records the *actual*
    backend that loaded - which may differ from the requested name if
    mcl was requested but its C library is missing."""
    global _active_name
    requested = os.environ.get("ECFHE_BACKEND", _DEFAULT).strip().lower()
    module = _resolve(requested)
    # If we hit the fallback path, _resolve returned the pyecc module
    # even though `requested` was mcl. Detect this via module name.
    actual = module.__name__.rsplit(".", 1)[-1]
    _active_name = actual
    return module


def name() -> str:
    """Return the name of the backend that *actually* loaded (post-
    fallback if any). For diagnostic / benchmark labels."""
    return _active_name
