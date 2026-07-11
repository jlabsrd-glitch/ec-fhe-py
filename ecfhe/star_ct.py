"""★_ct - Weil-pairing-based multiplication on encrypted scalars.

  LSL24 Theorem 8 (Multiplicative Homomorphism via Tensor Products).
  For ciphertexts (C₁⁽¹⁾, C₂⁽¹⁾), (C₁⁽²⁾, C₂⁽²⁾) encrypting m₁, m₂:

      (C₁⁽¹⁾, C₂⁽¹⁾) ⊗ (C₁⁽²⁾, C₂⁽²⁾)  :=  (C₁⁽¹⁾ ⋆ C₁⁽²⁾, C₂⁽¹⁾ ⋆ C₂⁽²⁾)

  where    U ⋆ V := σ⁻¹( e_n( σ(U), σ(V) ) ).

  Decrypt of the tensor product yields m₁*m₂.

Concrete realisation (BN254 Type-3 pairing):
  * Encrypt produces a *paired* ciphertext: the same plaintext rational
    (a, b) is encoded both in G1 (C1_g1, C2_g1, C3_g1) and in G2
    (C1_g2, C2_g2, C3_g2) using the SAME randomness r and the SAME
    secret s. This is the standard BGN-style sequencing that makes the
    pairing-mult well-defined.

  * ★_ct on the G1×G2 pair lands in GT (the target group of the
    pairing). Pure-addition operations stay on the G1 layer; products
    push into GT and stay there (one ★_ct level deep). LSL24's full
    multiplicative chain uses bootstrap (EC_FHE Alg 5) to bring values
    back from GT to G1 between products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import params
from .ciphertext import Ciphertext
from .keygen import PublicKey
from .noise import Noise
from .pairing import e_n, gt_inv


# Paired ciphertext: same plaintext encoded in both G1 and G2
@dataclass(frozen=True)
class PairedCiphertext:
    """Carries (C1_g1, C2_g1, C3_g1) AND (C1_g2, C2_g2, C3_g2).

    Same rational a/b, same randomness r, same secret s. Either component
    decrypts to (a, b); pairing them gives a product ciphertext in GT.
    """
    g1: Ciphertext                       # standard G1 ciphertext (encrypt.py)
    C1_g2: Any                           # r*G2
    C2_g2: Any                           # a*G2 + r*s*G2 = a*G2 + r*Q2
    C3_g2: Any                           # b*G2 + r*s*G2

    @property
    def delta(self) -> int:
        return self.g1.delta

    @property
    def noise(self) -> Noise:
        return self.g1.noise

    def replace_noise(self, n: Noise) -> "PairedCiphertext":
        return PairedCiphertext(g1=self.g1.replace(noise=n),
                                C1_g2=self.C1_g2, C2_g2=self.C2_g2,
                                C3_g2=self.C3_g2)


def encrypt_paired(pk: PublicKey, x, delta: int | None = None,
                   mode: str = "standard") -> PairedCiphertext:
    """Encrypt a plaintext into the paired (G1, G2) form needed for ★_ct.

    Critical invariant: G1 and G2 sides MUST share randomness r and the
    SAME secret s. The G2 public key Q2 = s*G2 is derived from the same
    s that produced Q (G1 public key) - this is set up at KeyGen time.
    """
    import secrets
    from decimal import Decimal
    from fractions import Fraction

    from .encrypt import _rand_r
    from .precision import real_to_elliptic_rational
    from .rational import EllipticRational

    if delta is None:
        delta = pk.delta_max

    # 1) extract rational a/b
    if isinstance(x, EllipticRational):
        rat = x.canonical()
        approx_err = 0.0
    elif isinstance(x, int):
        rat = EllipticRational(a=x, b=1).canonical()
        approx_err = 0.0
    else:
        rat, cert = real_to_elliptic_rational(x, delta, mode=mode)
        approx_err = cert.approximation_error

    a = rat.a % params.ORDER
    b = rat.b % params.ORDER
    r = _rand_r()

    # 2) G1 side - standard ECHC ciphertext
    C1_g1 = params.multiply(params.G1, r)
    rQ = params.multiply(pk.Q, r)
    C2_g1 = params.add(params.multiply(params.G1, a), rQ)
    C3_g1 = params.add(params.multiply(params.G1, b), rQ)
    g1_ct = Ciphertext(C1=C1_g1, C2=C2_g1, C3=C3_g1, delta=delta,
                       noise=Noise.fresh_after_encrypt(approx_err),
                       mode=mode)

    # 3) G2 side - same a, b, r, s, but encoded in G2
    #    Q2 = s*G2  is computed once from the secret s in pk.
    #    (We store sk separately; here we use a fresh r on G2 with the
    #    publicly-known Q2 = s*G2, kept in pk._Q2 - see keygen.)
    Q2 = _public_Q2(pk)
    C1_g2 = params.multiply(params.G2, r)
    rQ2 = params.multiply(Q2, r)
    C2_g2 = params.add(params.multiply(params.G2, a), rQ2)
    C3_g2 = params.add(params.multiply(params.G2, b), rQ2)

    return PairedCiphertext(g1=g1_ct, C1_g2=C1_g2, C2_g2=C2_g2, C3_g2=C3_g2)


# ★_ct on paired ciphertexts
@dataclass(frozen=True)
class GTCiphertext:
    """Product ciphertext in GT. The plaintext m₁*m₂ is recovered by
    BSGS over GT (Alg 11 line 4, transported to GT). Stays in GT until
    bootstrapped back to G1 (Alg 5)."""
    c00: Any                             # e_n(C1_g2, C1_g1)
    c01: Any                             # e_n(C1_g2, C2_g1) component
    c10: Any                             # e_n(C2_g2, C1_g1) component
    c11: Any                             # e_n(C2_g2, C2_g1); encrypts num * num
    # ditto for denominator slot (C3)
    d00: Any
    d01: Any
    d10: Any
    d11: Any
    delta: int
    noise: Noise


def star_ct(A: PairedCiphertext, B: PairedCiphertext) -> GTCiphertext:
    """★_ct per Theorem 8.

    For each of the (numerator, denominator) ciphertext components we
    compute four cross-pairings - the BGN-style four-term layout - so
    decryption can solve for the product scalar via Fermat-style
    cancellation against the secret s:

        M = c00 * c01^(-s) * c10^(-s) * c11^(s²)  →  e(G1,G2)^(a*c)

    (Same shape on the C3/denominator side; same s.)"""
    # Numerator slot - uses C2 components (which encrypt the numerator).
    c00 = e_n(A.C1_g2, B.g1.C1)               # → g^(r₁*r₂)
    c01 = e_n(A.C1_g2, B.g1.C2)               # → g^(r₂*a₁ + r₁*r₂*s)
    c10 = e_n(A.C2_g2, B.g1.C1)               # → g^(r₁*a₂ + r₁*r₂*s)
    c11 = e_n(A.C2_g2, B.g1.C2)               # → g^(a₁*a₂ + a₂r₁s + a₁r₂s + r₁r₂s²)
    # Denominator slot - same shape with C3 (encrypts denominator).
    d00 = e_n(A.C1_g2, B.g1.C1)               # identical to c00 (shared randomness)
    d01 = e_n(A.C1_g2, B.g1.C3)
    d10 = e_n(A.C3_g2, B.g1.C1)
    d11 = e_n(A.C3_g2, B.g1.C3)

    # Noise: multiplicative propagation per Thm 6.4
    new_noise = A.noise.mul(
        B.noise,
        x_abs=1.0, y_abs=1.0,            # values bounded; conservative
        c_const=1.0,
    )
    return GTCiphertext(c00=c00, c01=c01, c10=c10, c11=c11,
                        d00=d00, d01=d01, d10=d10, d11=d11,
                        delta=max(A.delta, B.delta), noise=new_noise)


# GT-deep aggregation (encrypted dot product)
def aggregate_gt(gt_cts: list[GTCiphertext]) -> GTCiphertext:
    """Combine N GT ciphertexts into one that encrypts the SUM of their
    plaintexts (Σ aᵢ*bᵢ for an encrypted dot product).

    GT is multiplicative; multiplication in GT corresponds to ADDITION
    in the exponent. So:
        prod_i  g^(aᵢbᵢ)  =  g^(Σ aᵢbᵢ).

    Each GT ciphertext has four numerator components (c00..c11) and four
    denominator components (d00..d11). We multiply them component-wise.
    The Fermat-cancellation at decrypt time then yields:
        Σ aᵢ*bᵢ   (numerator)
        Σ aᵢ_den * bᵢ_den   (denominator slot, if used)
    in a single GT-decrypt-and-DLOG step. **One** BSGS solve over GT
    extracts the entire dot product - N times faster than N decrypts
    plus N-1 plaintext additions."""
    if not gt_cts:
        raise ValueError("aggregate_gt needs at least one ciphertext")
    if len(gt_cts) == 1:
        return gt_cts[0]
    head = gt_cts[0]
    c00, c01, c10, c11 = head.c00, head.c01, head.c10, head.c11
    d00, d01, d10, d11 = head.d00, head.d01, head.d10, head.d11
    noise = head.noise
    delta = head.delta
    for ct in gt_cts[1:]:
        c00 = c00 * ct.c00
        c01 = c01 * ct.c01
        c10 = c10 * ct.c10
        c11 = c11 * ct.c11
        d00 = d00 * ct.d00
        d01 = d01 * ct.d01
        d10 = d10 * ct.d10
        d11 = d11 * ct.d11
        noise = noise.add(ct.noise)
        delta = max(delta, ct.delta)
    return GTCiphertext(c00=c00, c01=c01, c10=c10, c11=c11,
                        d00=d00, d01=d01, d10=d10, d11=d11,
                        delta=delta, noise=noise)


def encrypted_dot_product(paired_a: list[PairedCiphertext],
                          paired_b: list[PairedCiphertext]
                          ) -> GTCiphertext:
    """Σ aᵢ * bᵢ - one pairing-per-pair, single GT BSGS to decrypt.

    Use case: encrypted DOT in SQL, encrypted inner product in ML
    inference, encrypted convolution. The whole sum stays inside GT
    until the key-holder decrypts; the server never sees the partial
    products."""
    if len(paired_a) != len(paired_b):
        raise ValueError("vector length mismatch")
    products = [star_ct(a, b) for a, b in zip(paired_a, paired_b)]
    return aggregate_gt(products)


# KeyGen helper: derive Q2 = s*G2 from existing pk/sk
_PUBLIC_Q2_CACHE: dict[int, Any] = {}


def _public_Q2(pk: PublicKey):
    """Returns Q2 = s*G2. Cached per (pk identity)."""
    key = id(pk)
    if key in _PUBLIC_Q2_CACHE:
        return _PUBLIC_Q2_CACHE[key]
    # We can't recover s from pk alone - but the caller (encrypt_paired)
    # is only invoked with sk available through the session. We resolve
    # this by reading sk from pk's companion in a session, or by having
    # a higher-level wrapper precompute Q2 at keygen time. For now we
    # rely on the session having attached Q2.
    Q2 = getattr(pk, "_Q2", None)
    if Q2 is None:
        raise RuntimeError(
            "PublicKey is missing _Q2 = s*G2; call keygen_paired() instead "
            "of keygen() for ★_ct support")
    _PUBLIC_Q2_CACHE[key] = Q2
    return Q2


# Decrypt GT ciphertext via BSGS over GT
def decrypt_gt_to_scalar(gt_ct: GTCiphertext, secret_s: int,
                         gt_table: "GTTable") -> tuple[int, int]:
    """Recover (numerator*numerator, denom*denom) cleartext integers
    via the Fermat-cancellation trick on the four-pairing layout."""
    s = secret_s
    s2 = (s * s) % params.ORDER
    neg_s = (-s) % params.ORDER

    # With ciphertext layout
    #   C1_g{1,2} = r*G{1,2}
    #   C2_g{1,2} = a*G{1,2} + r*s*G{1,2}
    # the four cross-pairings give (in additive log over g = e(G2,G1)):
    #   c00 = log e(C1_g2, C1_g1)  = r₁*r₂
    #   c01 = log e(C1_g2, C2_g1)  = r₂*a₁  +  r₁*r₂*s
    #   c10 = log e(C2_g2, C1_g1)  = r₁*a₂  +  r₁*r₂*s
    #   c11 = log e(C2_g2, C2_g1)  = a₁*a₂  +  a₂*r₁*s + a₁*r₂*s + r₁*r₂*s²
    # Cancellation:  c11 − s*c01 − s*c10 + s²*c00  =  a₁*a₂.
    #
    # Multiplicatively in GT:
    #   M_num = c11 * c01^(−s) * c10^(−s) * c00^(s²)  =  g^(a₁*a₂)
    M_num = gt_ct.c11
    M_num = M_num * (gt_ct.c01 ** neg_s)
    M_num = M_num * (gt_ct.c10 ** neg_s)
    M_num = M_num * (gt_ct.c00 ** s2)
    num_prod = gt_table.solve(M_num)
    if num_prod is None:
        raise ValueError("numerator product outside GT BSGS range")

    M_den = gt_ct.d11
    M_den = M_den * (gt_ct.d01 ** neg_s)
    M_den = M_den * (gt_ct.d10 ** neg_s)
    M_den = M_den * (gt_ct.d00 ** s2)
    den_prod = gt_table.solve(M_den)
    if den_prod is None:
        raise ValueError("denominator product outside GT BSGS range")

    return num_prod, den_prod
