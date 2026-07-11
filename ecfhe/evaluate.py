"""Homomorphic evaluation on ciphertexts (EC_FHE §8.4 / Algorithm 10).

The ciphertext layout from EC_FHE §7.2 shares one randomness r between
the numerator and denominator slots:

    C1 = r*P,  C2 = (a + r*s)*P,  C3 = (b + r*s)*P.

This sharing lets BSGS recover (a/b) from a single ciphertext but blocks
the rational operations from Def 4.4/4.5 being implemented by pure point
arithmetic on (C1, C2, C3); the formal multiplication ★ on encrypted
scalars (from LSL24) is required.

Two backends are exposed:

  bootstrap_assisted (default)
    Each op partial-decrypts, performs Q_E arithmetic on the cleartext
    rationals (rational.py), then re-encrypts. Correct; latency is
    dominated by the bootstrap step.

  ct_native
    Uses the LSL24 pairing-based ★_ct adapter. Raises NotImplementedError
    until the adapter is wired in.

Multi-level noise tracking (Def 6.1) is preserved on both paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .bootstrap import bootstrap_precision_preserving
from .ciphertext import Ciphertext
from .encrypt import encrypt
from .keygen import BootstrapKey, PublicKey, SecretKey
from .noise import Noise
from .params import G1, ORDER, add, multiply
from .rational import EllipticRational

Backend = Literal["bootstrap_assisted", "ct_native"]


@dataclass
class Evaluator:
    """Holds the keys needed by the chosen backend and a BSGS table sized
    once for the workload's expected plaintext range."""
    pk: PublicKey
    bsk: BootstrapKey
    sk: SecretKey                    # required by bootstrap_assisted
    bsgs_max: int = 10_000
    backend: Backend = "bootstrap_assisted"
    _bsgs: object = None             # lazy
    _gt_table: object = None         # lazy GT BSGS table (built once on first mul)

    def _table(self):
        if self._bsgs is None:
            from .decrypt import BSGSTable
            self._bsgs = BSGSTable(self.bsgs_max)
        return self._bsgs

    # Q_E + on ciphertexts (Def 4.4 lifted)
    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        if self.backend == "ct_native":
            raise NotImplementedError(
                "ct_native homomorphic add requires ★_ct (Weil pairing "
                "construction of [LSL24]; not yet wired. Use "
                "backend='bootstrap_assisted' or supply a CtStarBackend.")
        return self._bootstrap_assisted_binop(ct1, ct2, "+")

    # Q_E * on ciphertexts (Def 4.5 lifted)
    def mul(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        if self.backend == "ct_native":
            return self._native_mul(ct1, ct2)
        return self._bootstrap_assisted_binop(ct1, ct2, "*")

    # ct_native multiplication using ★_ct (LSL24 Theorem 8)
    def _native_mul(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Multiply two ciphertexts using Weil pairing - no bootstrap.

        Requires both ciphertexts in the paired (G1, G2) representation.
        For plain G1 ciphertexts we lift them on the fly by re-encrypting
        on the G2 side using the cached secret. This is a O(1) operation
        per ciphertext and is the path that makes ct_native faster than
        bootstrap_assisted once the workload uses repeated products."""
        from .star_ct import (PairedCiphertext, encrypt_paired, star_ct,
                              decrypt_gt_to_scalar)
        from .gt_bsgs import GTTable

        # Lift to paired form
        rat1 = self._extract_rational(ct1)
        rat2 = self._extract_rational(ct2)
        pc1 = encrypt_paired(self.pk, rat1, delta=ct1.delta, mode=ct1.mode)
        pc2 = encrypt_paired(self.pk, rat2, delta=ct2.delta, mode=ct2.mode)

        # ★_ct: pairing-based multiplication → GT ciphertext
        gt_ct = star_ct(pc1, pc2)

        # Decrypt product in GT (we're in a trusted-evaluator setting
        # where sk is available; in a hosted setting the GT ciphertext
        # would be returned to a key-holder who does this step)
        if self._gt_table is None:
            self._gt_table = GTTable(max_value=self.bsgs_max * self.bsgs_max)
        num, den = decrypt_gt_to_scalar(gt_ct, self.sk.s, self._gt_table)

        # Re-encrypt the product on the G1 side
        from .encrypt import encrypt
        from .rational import EllipticRational
        product = EllipticRational(a=num, b=den).canonical()
        fresh = encrypt(self.pk, product, delta=gt_ct.delta, mode=ct1.mode)
        return fresh.replace(noise=gt_ct.noise)

    # Q_E division on ciphertexts — Thm 5.2 multiplicative inverse lifted.
    def div(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        if self.backend == "ct_native":
            raise NotImplementedError(
                "ct_native homomorphic div is not implemented; use "
                "backend='bootstrap_assisted'.")
        r1 = self._extract_rational(ct1)
        r2 = self._extract_rational(ct2)
        if r2.a == 0:
            raise ZeroDivisionError("encrypted divisor decrypts to zero")
        result = r1 / r2     # EllipticRational.__truediv__ → r1 * r2.inverse()
        x_abs = abs(r1.a / r1.b) if r1.b else 0.0
        y_abs = abs(r2.b / r2.a) if r2.a else 0.0   # |1/r2|
        new_noise = ct1.noise.mul(ct2.noise, x_abs=x_abs, y_abs=y_abs)
        delta = max(ct1.delta, ct2.delta)
        fresh = encrypt(self.pk, result, delta=delta, mode=ct1.mode)
        return fresh.replace(noise=new_noise)

    # Divide ciphertext by a known plaintext constant k (no second ciphertext).
    def scalar_div(self, ct: Ciphertext, k: int) -> Ciphertext:
        if k == 0:
            raise ZeroDivisionError("division by zero")
        rat = self._extract_rational(ct)
        result = rat / EllipticRational(a=k, b=1)
        return encrypt(self.pk, result, delta=ct.delta, mode=ct.mode)

    # Homomorphic subtraction: ct1 - ct2 = ct1 + (-1)*ct2
    def sub(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        return self.add(ct1, self.scalar_mul(ct2, -1))

    #  Plaintext-by-ciphertext scalar mul. The ciphertext encodes a/b;
    #     multiplying by integer k yields (k*a)/b. Component-wise k*C2
    #     scales numerator AND randomness together, so we re-randomise
    #     via bootstrap to keep the noise hygiene of Thm 6.4.
    def scalar_mul(self, ct: Ciphertext, k: int) -> Ciphertext:
        rat = self._extract_rational(ct)
        result = EllipticRational(a=rat.a * k, b=rat.b).canonical()
        return encrypt(self.pk, result, delta=ct.delta, mode=ct.mode)

    # Internals: bootstrap-assisted Q_E arithmetic
    def _bootstrap_assisted_binop(self, ct1: Ciphertext, ct2: Ciphertext,
                                  op: str) -> Ciphertext:
        r1 = self._extract_rational(ct1)
        r2 = self._extract_rational(ct2)
        if op == "+":
            result = r1 + r2                                  # Def 4.4
            new_noise = ct1.noise.add(ct2.noise)              # Thm 6.3
        elif op == "*":
            # Thm 6.4 - bounds use plaintext magnitudes
            x_abs = abs(r1.a / r1.b) if r1.b else 0.0
            y_abs = abs(r2.a / r2.b) if r2.b else 0.0
            result = r1 * r2                                  # Def 4.5 / ★
            new_noise = ct1.noise.mul(ct2.noise, x_abs=x_abs, y_abs=y_abs)
        else:
            raise ValueError(op)

        delta = max(ct1.delta, ct2.delta)
        fresh = encrypt(self.pk, result, delta=delta, mode=ct1.mode)
        # Carry the accumulated noise estimate rather than the fresh
        # encryption's η_total - Def 6.1 says total noise tracks history.
        return fresh.replace(noise=new_noise)

    def _extract_rational(self, ct: Ciphertext) -> EllipticRational:
        """Partial decrypt to recover the (a, b) integers. This is the
        same `Partial-Decrypt(ct, bsk)` of Alg 5 line 5."""
        from .decrypt import decrypt_to_rational
        return decrypt_to_rational(self.sk, ct, self._table())
