"""mcl backend - herumi's C++ pairing library.

Install (Linux / macOS):

    pip install mcl

    # If pip can't find the library at /usr/local/lib/libmcl, build mcl
    # from source first:
    #   git clone https://github.com/herumi/mcl
    #   cd mcl && mkdir build && cd build && cmake .. && make && sudo make install
    # then re-run `pip install mcl`.

Expected speedup vs. pyecc on the same workload:
    - scalar multiplication on G1     :  ~20-50×
    - pairing  (the slow operation)   :  ~100-500×
    - encrypt / decrypt / SUM         :  3-5× (bottlenecked by Python)
    - DOT (N pairings + GT decode)    :  ~100× for large N

This backend is curve BN254 (a.k.a. BN128 in py_ecc) so it interoperates
with the pyecc backend at the integer (scalar) layer. Switching backends
mid-session does NOT migrate stored ciphertexts; choose one at process
start (set ECFHE_BACKEND=mcl before importing ecfhe).

If `import mcl` fails on this host, ecfhe/backends/__init__.py issues a
RuntimeWarning and falls back to pyecc.
"""

from __future__ import annotations

import mcl as _mcl_pkg                          # raises ImportError → caller handles

# herumi/mcl's Python wrapper exposes BN254 via the `bn254` namespace.
# Some forks use lowercase `bn`, others uppercase; we pick the namespace
# at import time so the rest of this module is binding-agnostic.
_bn254 = getattr(_mcl_pkg, "bn254", None) or getattr(_mcl_pkg, "BN254", None)
if _bn254 is None:                              # pragma: no cover
    raise ImportError("mcl python binding lacks a bn254 namespace")

if hasattr(_bn254, "init"):
    _bn254.init()


# Curve / field constants
# BN254 order and field modulus (universal across mcl forks):
ORDER: int = (21888242871839275222246405745257275088548364400416034343698204186575808495617)
FIELD_MOD: int = (21888242871839275222246405745257275088696311157297823662689037894645226208583)


# Generators
def _g1_generator():
    """Standard BN254 G1 base point (1, 2)."""
    if hasattr(_bn254, "G1"):
        # Common shape: G1 is a class; constructing default may return
        # the identity. Use from_str / from_x_y when available; else fall
        # back to hashing a constant.
        g1 = _bn254.G1()
        if hasattr(g1, "set_str"):
            g1.set_str("1 1 2")                    # "1" = "non-identity"; 1, 2 = x, y
        return g1
    raise RuntimeError("mcl binding does not expose G1")


def _g2_generator():
    """Standard BN254 G2 base point (matching py_ecc's optimized_bn128)."""
    if hasattr(_bn254, "G2"):
        g2 = _bn254.G2()
        if hasattr(g2, "set_str"):
            # py_ecc-compatible G2 generator (affine, FQ2 coords).
            g2.set_str(
                "1 "
                "10857046999023057135944570762232829481370756359578518086990519993285655852781 "
                "11559732032986387107991004021392285783925812861821192530917403151452391805634 "
                "8495653923123431417604973247489272438418190587263600148770280649306958101930 "
                "4082367875863433681332203403145435568316851327593401208105741076214120093531"
            )
        return g2
    raise RuntimeError("mcl binding does not expose G2")


G1 = _g1_generator()
G2 = _g2_generator()


def _g1_identity():
    z = _bn254.G1()
    if hasattr(z, "clear"):
        z.clear()
    return z


def _g2_identity():
    z = _bn254.G2()
    if hasattr(z, "clear"):
        z.clear()
    return z


Z1 = _g1_identity()
Z2 = _g2_identity()


# Group operations
def add(P, Q):
    return P + Q


def multiply(P, k):
    if isinstance(k, int):
        # mcl scalars must be Fr; build one
        s = _bn254.Fr()
        s.set_int(k)
        return P * s
    return P * k


def neg(P):
    return -P


def pairing(P_g2, Q_g1):
    """e(P_g2, Q_g1) → GT element."""
    return _bn254.pairing(P_g2, Q_g1)


def normalize(P):
    if hasattr(P, "normalize"):
        P.normalize()
    # Return (x, y) tuple for compatibility with py_ecc consumers.
    if hasattr(P, "x") and hasattr(P, "y"):
        return (P.x, P.y)
    return P


def is_inf(P):
    if hasattr(P, "is_zero"):
        return P.is_zero()
    if hasattr(P, "isZero"):
        return P.isZero()
    return False


# Field element types (compat shims)
FQ = getattr(_bn254, "Fp", object)
FQ2 = getattr(_bn254, "Fp2", object)
FQ12 = getattr(_bn254, "GT", object)
