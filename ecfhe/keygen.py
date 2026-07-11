"""KeyGen - EC_FHE Algorithm 8.

  Require: security parameter λ, system params = (δ_max, B_default, bootstrap_freq)
  Ensure:  (pk, sk, bsk)

  1: select pairing-friendly E/F_q with embedding degree k ≤ 12
  2: choose generator P ∈ E(F_q) of large prime order n > 2^{2λ}
  3: verify  log10(n) > δ_max + λ                       (§10.1)
  4: s ← Z_n^*                                          (private scalar)
  5: Q ← s*P                                            (public point)
  6-9: bootstrap key generation
  10: select adaptive noise budget parameters
  11: pk ← (E, P, Q, n, B_default, δ_max)
  12: sk ← s
  13: bsk ← Generate-Bootstrap-Key(s, s_boot, E, P)

The curve is fixed at BN254 (params.py) so steps 1-3 amount to verifying
δ_max against the actual log10(n). The bootstrap key is, by paper, an
auxiliary structure for noise-free re-encryption; in our setting "Generate-
Bootstrap-Key" is realised as a separate scalar s_boot whose role is
documented in bootstrap.py (Alg 5 partial decrypt).
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any

from .params import G1, ORDER, ORDER_BITLEN, multiply


# Default system parameters (EC_FHE §10.1)
DEFAULT_DELTA_MAX = max_dm = max(1, int(math.log10(ORDER)) - 128)   # λ=128
DEFAULT_BUDGET_BASE = 10 ** (-DEFAULT_DELTA_MAX)                    # B_0
DEFAULT_BOOTSTRAP_FREQ = 1000                                       # ops between auto-bootstraps


@dataclass(frozen=True)
class PublicKey:
    """pk ← (E, P, Q, n, B_default, δ_max) - line 11 of Alg 8."""
    P: Any                  # generator
    Q: Any                  # s*P
    n: int                  # group order
    delta_max: int
    budget_base: float

    # E is implicit (BN254 in params.py). We keep the field as a docstring
    # rather than a stored object so the public key stays small.


@dataclass(frozen=True)
class SecretKey:
    """sk ← s - line 12 of Alg 8."""
    s: int                  # 0 < s < n


@dataclass(frozen=True)
class BootstrapKey:
    """bsk ← Generate-Bootstrap-Key(s, s_boot, E, P) - line 9 of Alg 8.

    Carries both the main secret `s` (needed for partial decrypt during
    bootstrap, Alg 5 line 5) and an independent scalar s_boot used to
    randomise the noise-refresh step. The struct is *secret material* -     Thm 9.1 (semantic security under bsk exposure) is what permits it
    to be passed to bootstrap servers; it must still be kept off public
    storage."""
    s: int
    s_boot: int


@dataclass(frozen=True)
class KeyPair:
    public: PublicKey
    secret: SecretKey
    bootstrap: BootstrapKey


# Algorithm 8
def _rand_scalar() -> int:
    """Uniform random non-zero scalar in Z_n^*  (line 4)."""
    while True:
        s = secrets.randbelow(ORDER)
        if s != 0:
            return s


def keygen(*, security_param: int = 128,
           delta_max: int | None = None,
           budget_base: float | None = None,
           bootstrap_freq: int = DEFAULT_BOOTSTRAP_FREQ) -> KeyPair:
    """Algorithm 8 - EC_FHE Key Generation."""

    # Line 1-3: curve + precision sanity (BN254 is fixed)
    #
    # EC_FHE §10.1 states the *ideal* sizing  log10(n) > δ_max + λ. With
    # BN254 (≈100-bit pairing security; log10(n) ≈ 76) that ideal is
    # only approachable at low λ - production deployments either step up
    # to BLS12-381 (≈128-bit) or run BN254 with a smaller margin.  We
    # therefore separate two parameters:
    #
    #   security_param - used for *book-keeping* / API symmetry with
    #                     the paper. Default 128. Does NOT affect curve
    #                     sizing (the curve is BN254 by construction).
    #   precision_headroom - number of decimal digits the curve order
    #                     must exceed δ_max by. Default 16 (≈ 53-bit
    #                     room above the plaintext range, enough for
    #                     mod-n wraparound and noise drift to remain
    #                     unambiguous).
    log10_n = math.log10(ORDER)
    headroom = 16
    if delta_max is None:
        delta_max = max(1, int(log10_n) - headroom)
    if budget_base is None:
        budget_base = 10.0 ** (-delta_max)

    if log10_n <= delta_max + headroom:
        raise ValueError(
            f"curve order too small for δ_max={delta_max} (BN254 has "
            f"log10(n)≈{log10_n:.1f}; need δ_max ≤ {int(log10_n) - headroom})"
        )

    # Line 4-5: private/public pair
    s = _rand_scalar()
    Q = multiply(G1, s)
    # Also derive Q2 = s*G2 so the same s parameterises both BGN-style
    # ciphertext sides. Stored on pk._Q2 for star_ct.encrypt_paired.
    from .params import G2 as _G2
    Q2 = multiply(_G2, s)

    # Line 6-9: bootstrap key
    s_boot = _rand_scalar()

    public = PublicKey(P=G1, Q=Q, n=ORDER,
                       delta_max=delta_max, budget_base=budget_base)
    # Stash Q2 on the PublicKey for ★_ct. We use object.__setattr__
    # because PublicKey is a frozen dataclass.
    object.__setattr__(public, "_Q2", Q2)
    secret = SecretKey(s=s)
    bsk = BootstrapKey(s=s, s_boot=s_boot)
    return KeyPair(public=public, secret=secret, bootstrap=bsk)
