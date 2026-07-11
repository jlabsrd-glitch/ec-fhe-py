"""Bootstrap operations - EC_FHE §7.

  Def 7.1  Three bootstrap types:
    (1) Precision-Preserving Bootstrap (Algorithm 5)
 - reduces η_comp and η_crypto while preserving η_approx and δ.
    (2) Precision-Enhancing Bootstrap (Algorithm 6)
 - also improves δ_old → δ_new with finer rational approximation.
    (3) Selective Bootstrap Scheduler (Algorithm 7)
 - chooses *which* ciphertexts in a collection should be refreshed,
        prioritised by  criticality_i = η_total(ct_i) / B(next_usage(ct_i)).

  Thm 7.2  Algorithm 5 yields a refreshed ciphertext whose noise satisfies
        η_total(ct_fresh) ≤ η_approx(ct_noisy) + ε_bootstrap + η_fresh.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Sequence

from .ciphertext import Ciphertext
from .decrypt import BSGSTable, decrypt_to_rational
from .encrypt import encrypt
from .keygen import BootstrapKey, PublicKey
from .noise import Noise
from .params import ORDER
from .precision import real_to_elliptic_rational
from .rational import EllipticRational


# Algorithm 5 - Precision-Preserving Bootstrap
def bootstrap_precision_preserving(ct: Ciphertext,
                                   *, pk: PublicKey, bsk: BootstrapKey,
                                   bsgs: BSGSTable,
                                   maintain_delta: int | None = None,
                                   ) -> Ciphertext:
    """Algorithm 5 verbatim.

    Phases:
      1. Precision verification - abort if δ < δ_maintain (line 1-3).
      2. Decrypt-and-Extract rational coefficients (lines 4-9).
      3. Re-encode with fresh randomness r' (lines 10-14).
      4. Noise estimation and certification (lines 15-19).

    Returns a refreshed ciphertext encrypting the same rational with
        η_approx unchanged
        η_comp  reset to ε_extract
        η_crypto reset to the fresh-encryption floor
    matching Thm 7.2."""
    delta_maintain = maintain_delta if maintain_delta is not None else ct.delta

    # Phase 1
    if ct.delta < delta_maintain:
        raise ValueError("INSUFFICIENT_PRECISION_FOR_BOOTSTRAP")

    # Phase 2 - Partial-Decrypt(ct, bsk)
    # bsk carries the main secret s; we route through decrypt_to_rational
    # which performs C2 − s*C1 → a*P, BSGS → a, and the same for b.
    from .keygen import SecretKey
    rational = decrypt_to_rational(SecretKey(s=bsk.s), ct, bsgs)
    eps_extract = _extraction_floor(bsgs.max_value)

    # Phase 3 - re-encode with fresh randomness (using s_boot to differ
    # from any prior r used on this rational, mitigating ciphertext-linkage).
    fresh_r = (secrets.randbelow(ORDER - 1) + 1)
    fresh = encrypt(pk, rational, delta=ct.delta, mode=ct.mode, r=fresh_r)

    # Phase 4 - noise budget per Thm 7.2
    new_noise = Noise(
        approx=ct.noise.approx,                # unchanged
        comp=eps_extract,                      # reset
        crypto=fresh.noise.crypto,             # fresh
    )
    return fresh.replace(noise=new_noise)


# Algorithm 6 - Precision-Enhancing Bootstrap
def bootstrap_precision_enhancing(ct: Ciphertext,
                                  *, pk: PublicKey, bsk: BootstrapKey,
                                  bsgs: BSGSTable,
                                  new_delta: int,
                                  real_hint: float | Decimal | Fraction | None = None,
                                  ) -> Ciphertext:
    """Algorithm 6 verbatim.

    Extracts the current rational, then asks Real-to-EC-Rational to find a
    *better* approximation at the higher target precision δ_new > δ_old.
    If a hint of the true real value is available, it is preferred over
    the recovered cleartext (line 2-5)."""
    if new_delta <= ct.delta:
        raise ValueError("new_delta must be > ct.delta for precision-enhancing")

    from .keygen import SecretKey
    rational_old = decrypt_to_rational(SecretKey(s=bsk.s), ct, bsgs)
    estimate = (Decimal(real_hint) if real_hint is not None
                else Decimal(rational_old.a) / Decimal(rational_old.b))

    new_rat, cert = real_to_elliptic_rational(estimate, new_delta,
                                              mode="optimized")
    # Line 10-12: if the new conversion offers no measurable improvement
    # over the bound that the caller is willing to lose (here interpreted
    # as the existing approx noise, with a small slack to avoid floating-
    # point ties), return the input unchanged.
    if cert.approximation_error > ct.noise.approx * 0.999 and ct.noise.approx > 0:
        return ct

    fresh = encrypt(pk, new_rat, delta=new_delta, mode="optimized")
    new_noise = Noise(
        approx=cert.approximation_error,
        comp=_extraction_floor(bsgs.max_value),
        crypto=fresh.noise.crypto,
    )
    return fresh.replace(noise=new_noise, delta=new_delta)


# Algorithm 7 - Selective Bootstrap Scheduler
@dataclass(frozen=True)
class BootstrapCandidate:
    index: int
    criticality: float
    ratio: float


def selective_bootstrap_schedule(cts: Sequence[Ciphertext],
                                 *, budget_at_next_usage: Sequence[float],
                                 threshold: float = 1.0,
                                 ) -> list[BootstrapCandidate]:
    """Algorithm 7 - return an ordered schedule of ciphertexts to refresh.

    criticality_i = η_total(ct_i) / B(next_usage(ct_i, O))

    The (cost, benefit) ratio is approximated by η_total/maintenance_cost.
    Following the paper, a ciphertext is selected when
        ratio > threshold  AND  criticality > 0.7."""
    out: list[BootstrapCandidate] = []
    for i, (ct, b) in enumerate(zip(cts, budget_at_next_usage)):
        criticality = ct.noise.total / max(b, 1e-300)
        # Estimate: a precision-preserving bootstrap costs ~1 unit and
        # buys the user the entirety of η_comp + η_crypto reduction.
        benefit = ct.noise.comp + ct.noise.crypto
        ratio = benefit / 1.0
        if ratio > threshold and criticality > 0.7:
            out.append(BootstrapCandidate(index=i, criticality=criticality,
                                          ratio=ratio))
    out.sort(key=lambda c: -c.criticality)         # highest-criticality first
    return out


# Helper: extraction noise floor used in Phase 2 + Phase 4
def _extraction_floor(bsgs_max: int) -> float:
    """ε_extract bound. BSGS recovers an integer in [0, bsgs_max] exactly
    when its plaintext lies in range, so the *extraction* error is the
    quantisation floor 1/(2n) (cf. Thm 6.3's ε_add)."""
    return 1.0 / (2.0 * ORDER)
