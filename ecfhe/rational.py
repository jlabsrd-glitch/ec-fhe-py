"""Elliptic rational numbers Q_E - ECHCoverQ §4.

Implements verbatim:

  Def 4.1  Elliptic rational pair  (a, b)_P := (aP, bP)
  Def 4.2  (a,b)_P ~ (c,d)_P  ⟺  adP = bcP    in E(F_q)
  Lemma 4.1 ~ is an equivalence relation
  Def 4.3  Q_E := { (aP, bP) } / ~
  Def 4.4  Addition: [a/b]_P + [c/d]_P = [(ad+bc)/(bd)]_P
  Def 4.5  Multiplication: [a/b]_P * [c/d]_P = [(a★c)/(b★d)]_P
  Thm 4.1  Addition is well-defined
  Thm 4.2  Formal multiplication preserves the equivalence relation
  Thm 5.1  (Q_E, +) is an abelian group
  Thm 5.2  Multiplicative structure (associativity, distributivity,
           identity [1/1]_P, inverse [b/a]_P)
  Def 6.1  Canonical representation: gcd(a,b) = 1, b > 0
  Alg 1    Canonical Form Computation
  Alg 2    Addition in Q_E
  Alg 3    Multiplication in Q_E

This module is the *plaintext layer* of Q_E. Ciphertext-level ops live in
ciphertext.py / evaluate.py; the integer arithmetic a*d + b*c happens
here on the cleartext integers (mod n) and the result is then encoded
back to (aP, bP).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Union

from .echc import star
from .params import G1, ORDER, eq, multiply


# Def 4.1, 4.3, 6.1
@dataclass(frozen=True)
class EllipticRational:
    """[a/b]_P ∈ Q_E in canonical form (gcd(a, b) = 1, b > 0)."""
    a: int
    b: int

    def __post_init__(self) -> None:
        if self.b == 0:
            raise ValueError("denominator b must be nonzero  (Def 4.1)")
        # Note: bP ≠ O is enforced at the point level; here b is an integer.
        # If b ≡ 0 (mod n) the caller has chosen a degenerate denominator.

    # Constructors
    @classmethod
    def from_pair(cls, a: int, b: int) -> "EllipticRational":
        return cls(a=a, b=b).canonical()

    @classmethod
    def from_int(cls, k: int) -> "EllipticRational":
        return cls(a=k, b=1).canonical()

    @classmethod
    def zero(cls) -> "EllipticRational":
        return cls(a=0, b=1)

    @classmethod
    def one(cls) -> "EllipticRational":
        return cls(a=1, b=1)

    # Algorithm 1 - Canonical Form Computation
    def canonical(self) -> "EllipticRational":
        """Algorithm 1 of ECHCoverQ §6.1 verbatim:
            g  ← gcd(a, b)
            a' ← a / g
            b' ← |b| / g
            if b < 0: a' ← -a'
        """
        a, b = self.a, self.b
        g = gcd(abs(a), abs(b)) or 1
        a2, b2 = a // g, abs(b) // g
        if b < 0:
            a2 = -a2
        return EllipticRational(a=a2, b=b2)

    # Point representation (aP, bP)
    def point_pair(self) -> tuple:
        """(aP, bP) - the two elliptic-curve points that this rational
        equivalence-class chose as representative. Used for equivalence
        testing (Def 4.2) and ciphertext construction."""
        return (multiply(G1, self.a % ORDER),
                multiply(G1, self.b % ORDER))

    # Def 4.2 - Equivalence on Q_E
    def equivalent_to(self, other: "EllipticRational") -> bool:
        """(a, b)_P ~ (c, d)_P  ⟺  adP = bcP    (Def 4.2, tested on the curve)."""
        adP = multiply(G1, (self.a * other.b) % ORDER)
        bcP = multiply(G1, (self.b * other.a) % ORDER)
        return eq(adP, bcP)

    def __eq__(self, other) -> bool:
        """Python equality compares canonical forms - *not* equivalence
        on the curve. For Def-4.2 equivalence use .equivalent_to()."""
        if not isinstance(other, EllipticRational):
            return False
        u, v = self.canonical(), other.canonical()
        return u.a == v.a and u.b == v.b

    def __hash__(self) -> int:
        c = self.canonical()
        return hash((c.a, c.b))

    # Def 4.4 / Algorithm 2 - Addition
    def __add__(self, other: "EllipticRational") -> "EllipticRational":
        a, b = self.a, self.b
        c, d = other.a, other.b
        num = a * d + b * c
        den = b * d
        return EllipticRational(a=num, b=den).canonical()

    def __neg__(self) -> "EllipticRational":
        return EllipticRational(a=-self.a, b=self.b).canonical()

    def __sub__(self, other: "EllipticRational") -> "EllipticRational":
        return self + (-other)

    # Def 4.5 / Algorithm 3 - Multiplication via ★
    def __mul__(self, other) -> "EllipticRational":
        if isinstance(other, int):
            # Integer scalar - corresponds to (k*a) / b
            return EllipticRational(a=self.a * other, b=self.b).canonical()
        if not isinstance(other, EllipticRational):
            return NotImplemented
        num = star(self.a, other.a)   # a ★ c
        den = star(self.b, other.b)   # b ★ d
        return EllipticRational(a=num, b=den).canonical()

    __rmul__ = __mul__

    # Thm 5.2 - Multiplicative inverse
    def inverse(self) -> "EllipticRational":
        if self.a == 0:
            raise ZeroDivisionError("zero has no multiplicative inverse in Q_E")
        return EllipticRational(a=self.b, b=self.a).canonical()

    def __truediv__(self, other: "EllipticRational") -> "EllipticRational":
        return self * other.inverse()

    def __repr__(self) -> str:
        return f"[{self.a}/{self.b}]_P"


# Convenience identities
ZERO = EllipticRational.zero()
ONE = EllipticRational.one()


# Lemma 4.1 sanity helpers
def is_equivalence(a: EllipticRational, b: EllipticRational) -> bool:
    """Convenience wrapper for (Def 4.2)."""
    return a.equivalent_to(b)
