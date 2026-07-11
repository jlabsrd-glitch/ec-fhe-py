"""Ciphertext layout - EC_FHE §7.2 (Algorithm 5 phase 2).

The paper specifies the refreshed ciphertext components as:

      C'_1 ← r'*P
      C'_2 ← (a + r'*s)*P
      C'_3 ← (b + r'*s)*P

i.e. a triple of G1 points together with the precision δ and the noise
estimate η. We carry η as the multi-level noise tuple from noise.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .noise import Noise


@dataclass(frozen=True)
class Ciphertext:
    """An EC-FHE ciphertext: (C1, C2, C3, δ, η).

    Encrypts a rational a/b ∈ Q_E whose numerator and denominator share
    the same masking randomness r - this shared-r structure is what
    makes the ciphertext-level homomorphic addition work, because
    additions on (C2, C3) preserve the *same* r*s*P offset on both
    coordinates and the rational arithmetic (ad+bc)/(bd) is computable
    from the ciphertext alone."""

    C1: Any           # r*P
    C2: Any           # (a + r*s)*P  ; encrypts numerator a
    C3: Any           # (b + r*s)*P  ; encrypts denominator b
    delta: int        # precision parameter δ (EC_FHE §3.1 external layer)
    noise: Noise = field(default_factory=Noise)

    # The encryption "mode" used to produce this ciphertext is stored only
    # for diagnostics; once encrypted, mode doesn't affect HE ops.
    mode: str = "standard"

    def replace(self, **kw) -> "Ciphertext":
        from dataclasses import replace as _r
        return _r(self, **kw)
