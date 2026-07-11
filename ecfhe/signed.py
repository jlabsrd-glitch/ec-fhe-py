"""Canonical signed elliptic integers - ECHCoverQ §3.

Implements verbatim:

  Def 3.1  Raw signed elliptic integer   (a, s)_E := (aP, s)
  Def 3.2  cann_n(r̄) : Z_n → {0,...,⌊n/2⌋} × {0,1}
  Def 3.3  Canonical signed elliptic integer [r̄]_E
  Def 3.4  Addition (a₁,s₁)_E ⊕ (a₂,s₂)_E
  Def 3.5  Point map Π
  Def 3.6  Integer scalar multiplication k ⊙ (a,s)_E
  Def 3.7  Negation -(a,s)_E
  Prop 3.1 (canonical signed pairs, ⊕) ≅ Z_n

The point coordinate (aP) is *not* materialised inside SignedInt itself;
that lives in rational.py / ciphertext.py. SignedInt only carries the
canonical (magnitude, sign) pair - which is exactly Φ((a,s)) of Prop 3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import G1, ORDER, multiply, neg


# Def 3.2
def cann(r: int, n: int = ORDER) -> tuple[int, int]:
    """cann_n(r̄). Returns the canonical (a, s) ∈ {0,...,⌊n/2⌋} × {0,1}."""
    r = r % n
    half = n // 2
    if r <= half:
        return (r, 0)
    return (n - r, 1)


# Def 3.1 + 3.3
@dataclass(frozen=True)
class SignedInt:
    """Canonical signed elliptic integer [r̄]_E.

    Carries the unique pair (a, s) with 0 ≤ a ≤ ⌊n/2⌋, s ∈ {0, 1}.
    The corresponding curve point is recoverable via .point() - but is
    *not* cached, since the same SignedInt may live in different
    cryptographic contexts (plain Z_n point, blinded ciphertext, etc.).
    """
    a: int
    s: int

    @classmethod
    def from_int(cls, k: int) -> "SignedInt":
        """ψ(k) of ECHCoverQ §3: integer → canonical signed pair."""
        a, s = cann(k)
        return cls(a=a, s=s)

    # Def 3.5 - Point map Π((a,s)_E) = (-1)^s * aP
    def point(self):
        P = multiply(G1, self.a)
        return neg(P) if self.s == 1 else P

    # Φ of Prop 3.1 - canonical signed pair → residue in Z_n
    def to_zn(self) -> int:
        return ((-1) ** self.s * self.a) % ORDER

    # Def 3.4 - Addition ⊕
    def __add__(self, other: "SignedInt") -> "SignedInt":
        return SignedInt.from_int(self.to_zn() + other.to_zn())

    def __sub__(self, other: "SignedInt") -> "SignedInt":
        return self + (-other)

    # Def 3.7 - Negation -(a,s)_E
    def __neg__(self) -> "SignedInt":
        return SignedInt.from_int((-self.to_zn()) % ORDER)

    # Def 3.6 - Integer scalar multiplication k ⊙ (a,s)_E
    def __mul__(self, k: int) -> "SignedInt":
        if not isinstance(k, int):
            return NotImplemented
        return SignedInt.from_int((k * self.to_zn()) % ORDER)

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        return isinstance(other, SignedInt) and self.a == other.a and self.s == other.s

    def __hash__(self) -> int:
        return hash((self.a, self.s))

    def __repr__(self) -> str:
        sign = "+" if self.s == 0 else "-"
        return f"SignedInt({sign}{self.a})"


# Module-level identity (cann(0) = (0, 0)) - see Sanity checks in §3
ZERO = SignedInt(a=0, s=0)
