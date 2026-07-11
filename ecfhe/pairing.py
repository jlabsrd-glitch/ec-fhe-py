"""Weil pairing - LSL24 §2.3.

Definition 8 (Weil Pairing):
    e_m : E[m] × E[m] → μ_m,    bilinear, alternating, non-degenerate.

Theorem 1 (Properties):
    e_m(P₁+P₂, Q)   = e_m(P₁, Q) * e_m(P₂, Q)
    e_m(P, Q₁+Q₂)   = e_m(P, Q₁) * e_m(P, Q₂)
    e_m(P, Q)       = e_m(Q, P)⁻¹
    e_m(P, P)       = 1   (alternating)

In our BN254 setting we use the optimal-ate pairing as a concrete
realisation (Galbraith-Paterson-Smart 2008 prove it shares all the
properties needed by Theorem 8). The py_ecc.optimized_bn128 module
provides this as `pairing(G2_point, G1_point) → GT_element`.

This module is the *only* place that touches the pairing primitive,
so swapping in a fast backend (mcl, RELIC) means changing one file.
"""

from __future__ import annotations

from . import params

# Re-export under paper-friendly name
e_n = params.pairing            # e_n : G2 × G1 → GT


def weil(P_in_G2, Q_in_G1):
    """Compute e_n(P, Q). P must be in G2, Q in G1 (Type-3 pairing).

    This is the curve-side primitive used by ★_ct (echc.star_ct). Callers
    that hold a single elliptic-curve point P ∈ E(F_q) should map it into
    G1 vs G2 using the "twist" choice fixed at curve setup; for ECHC
    Theorem 8 we use the canonical embedding P ↦ (P_G1, ψ(P)_G2) where
    ψ is the GLS untwist isomorphism."""
    return e_n(P_in_G2, Q_in_G1)


def gt_one():
    """Identity in GT - useful as initial accumulator for products of
    pairing values."""
    return params.gt_one() if hasattr(params, "gt_one") else _gt_one_compute()


_GT_ONE_CACHE = None


def _gt_one_compute():
    global _GT_ONE_CACHE
    if _GT_ONE_CACHE is None:
        _GT_ONE_CACHE = e_n(params.G2, params.G1) ** 0
    return _GT_ONE_CACHE


def gt_inv(g):
    """Multiplicative inverse in GT via Fermat: g^(n-1) ≡ g⁻¹ (mod r).
    Used inside ★_ct when subtracting pairing contributions."""
    return g ** ((params.ORDER - 1) % params.ORDER)
