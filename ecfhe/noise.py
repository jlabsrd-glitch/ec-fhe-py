"""Multi-level noise model - EC_FHE §6.

  Def 6.1  Multi-level noise:
      η_approx(α) = |x − a/b|                    (real→rational conversion)
      η_comp(α)   = accumulated rounding         (homomorphic ops)
      η_crypto(α) = encryption/decryption noise
      η_total(α)  = η_approx(α) + η_comp(α) + η_crypto(α)

  Def 6.2  Dynamic noise budget B(t).

  Thm 6.3  Additive propagation:
      η_approx(α+β) ≤ η_approx(α) + η_approx(β)
      η_comp  (α+β) ≤ η_comp(α)   + η_comp(β)   + ε_add
      η_crypto(α+β) ≤ η_crypto(α) + η_crypto(β)

  Thm 6.4  Multiplicative propagation:
      η_approx(α*β) ≤ |y|*η_approx(α) + |x|*η_approx(β) + η_approx(α)*η_approx(β)
      η_comp  (α*β) ≤ max(|x|,|y|)*(η_comp(α)+η_comp(β)) + ε_mult
      η_crypto(α*β) ≤ C*(η_crypto(α) + η_crypto(β))

  ε_add  ≤ 1/(2n)  per Thm 6.3 (mod-n encoding quantisation)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .params import ORDER


# 1/(2n) bound from Thm 6.3, taken in absolute value at the rational level
_EPS_ADD = 1.0 / (2.0 * ORDER)
_EPS_MULT_BASE = 1.0 / (2.0 * ORDER)


@dataclass(frozen=True)
class Noise:
    """The triple (η_approx, η_comp, η_crypto) from Def 6.1."""
    approx: float = 0.0
    comp: float = 0.0
    crypto: float = 0.0

    @property
    def total(self) -> float:                       # η_total
        return self.approx + self.comp + self.crypto

    # Fresh noises
    @classmethod
    def fresh_after_encrypt(cls, approx_err: float, *,
                            crypto_floor: float | None = None) -> "Noise":
        """Noise of a freshly produced ciphertext. η_comp = 0 (no ops yet),
        η_crypto is bounded by the encryption-randomness floor."""
        return cls(approx=approx_err, comp=0.0,
                   crypto=crypto_floor if crypto_floor is not None else _EPS_MULT_BASE)

    # Thm 6.3 - Additive propagation
    def add(self, other: "Noise", *, eps_add: float = _EPS_ADD) -> "Noise":
        return Noise(
            approx=self.approx + other.approx,
            comp=self.comp + other.comp + eps_add,
            crypto=self.crypto + other.crypto,
        )

    # Thm 6.4 - Multiplicative propagation
    def mul(self, other: "Noise", *, x_abs: float, y_abs: float,
            c_const: float = 1.0, eps_mult: float = _EPS_MULT_BASE) -> "Noise":
        return Noise(
            approx=y_abs * self.approx + x_abs * other.approx
                   + self.approx * other.approx,
            comp=max(x_abs, y_abs) * (self.comp + other.comp) + eps_mult,
            crypto=c_const * (self.crypto + other.crypto),
        )

    def scalar_mul(self, k_abs: float, *, eps_mult: float = _EPS_MULT_BASE) -> "Noise":
        """Plaintext-by-ciphertext multiplication - same as mul with the
        other operand carrying zero noise of its own."""
        return Noise(
            approx=k_abs * self.approx,
            comp=k_abs * self.comp + eps_mult,
            crypto=k_abs * self.crypto,
        )

    # Within-budget check
    def within(self, budget: float) -> bool:
        return self.total <= budget

    def __repr__(self) -> str:
        return (f"Noise(approx={self.approx:.2e}, "
                f"comp={self.comp:.2e}, crypto={self.crypto:.2e}, "
                f"total={self.total:.2e})")


# Def 6.2 - Dynamic noise budget
@dataclass
class Budget:
    """B(t) of Def 6.2. Three strategies are supported (linear, adaptive,
    threshold). Default is linear decay used by Alg 4."""
    base: float
    expected_ops: int = 1000
    strategy: str = "linear"      # "linear" | "adaptive" | "threshold"
    high: float | None = None
    low: float | None = None
    strict: float | None = None

    def at(self, t: int) -> float:
        if self.strategy == "linear":
            decay = (self.base / max(1, self.expected_ops)) * t
            return max(self.base * 0.01, self.base - decay)
        if self.strategy == "threshold":
            # B(t) = B_high  if  current < B_low  else  B_strict
            # The "current" comparison happens at call site; here we
            # return the high band as the default
            return self.high if self.high is not None else self.base
        # adaptive: leave reduction to caller-supplied scaling
        return self.base
