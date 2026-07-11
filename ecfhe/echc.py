"""★ - Formal multiplication, ECHCoverQ §4.3.

  Def 4.5  For a cyclic subgroup ⟨R⟩ ⊆ E[n] equipped with the formal
  multiplication ★ transported from Z/nZ via the ECHC isomorphism,

      [a/b]_P * [c/d]_P  :=  [(a ★ c) / (b ★ d)]_P .

  Theorem 4.2 The formal multiplication preserves the equivalence
  relation on Q_E. The construction of ★ on the *elliptic-curve* side
  is the topic of the foundation paper

      [LSL24]  Y. Lee, M. Shim, J. Lee, "Homomorphic-based Encryption
               using Weil Pairing: A Foundation for Fully Homomorphic
               Encryption on Elliptic Curves", 2024.

  At the *integer* level, the isomorphism Z_n → ⟨R⟩ , k ↦ kR  carries
  multiplication mod n to ★. So when we hold the scalars `a`, `c` as
  elements of Z_n (which is exactly what `EllipticRational(a, b)` does),
  the value a ★ c reduces to plain (a * c) mod n:

      Φ⁻¹(Φ(a) ★ Φ(c))  =  (a * c)  mod n.

  We expose ★ as a top-level function so the day [LSL24]'s exact
  curve-side construction (pairing computation + transport map) is plugged
  in, only this file changes; rational.py, encrypt.py, evaluate.py all
  remain untouched.

Two ★ forms are needed by the rest of the package:

  ★ on cleartext integers (this file, `star`)
      Used by EllipticRational.__mul__ when we have access to a, c as
      ordinary Z_n scalars. This is the same value as the curve-side ★
      under the isomorphism, by construction.

  ★ on ciphertexts (file: evaluate.py, `mul_ct`)
      Used when a, c are *encrypted* and the server must compute the
      product without learning them. That is where the Weil-pairing
      machinery of [LSL24] is required; mul_ct delegates to a pluggable
      adapter (`CtStarBackend`) so the exact pairing recipe can be
      swapped in cleanly.
"""

from __future__ import annotations

from .params import ORDER


# ★ on Z_n (cleartext scalars)
def star(a: int, c: int) -> int:
    """Formal multiplication on cleartext integers.

    The ECHC isomorphism Z_n ≅ ⟨R⟩ (ECHCoverQ §4.3, Thm 4.2) transports
    * on Z_n to ★ on ⟨R⟩. At the *plaintext* level - which is where Def
    6.1's canonical representation lives - a and b are *integers*
    (possibly negative, possibly larger than n in absolute value), and
    the mod-n reduction is applied only when those integers are encoded
    as elliptic-curve points (multiply(G1, a % n)).

    So plaintext ★ is exact integer multiplication. This keeps Q_E's
    field-like properties (distributivity, signed inverses) intact;
    the mod-n behaviour belongs to the curve-side ★_ct, not here.
    """
    return a * c
