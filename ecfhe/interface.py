"""Two-Layer External Interface - EC_FHE §3.

   External Layer (Real Number Interface)
       Real Number Input/Output, Precision Management, Error Reporting
            | real numbers, δ            | real results, error bounds
            ▼                            ▲
   Interface Layer
       Real-to-Rational Conversion, Error Bounds, Precision Tracking
            | elliptic rationals         | computation results
            ▼                            ▲
   Internal Layer (Elliptic Rational Engine)
       EC-FHE Operations, Noise Management, Bootstrapping

Algorithm 1 (System Data Flow with Precision Management) is implemented
by `ECFHESession.evaluate(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable

from .bootstrap import bootstrap_precision_preserving
from .ciphertext import Ciphertext
from .decrypt import BSGSTable, decrypt
from .encrypt import encrypt
from .evaluate import Evaluator
from .keygen import KeyPair, keygen
from .noise import Budget
from .precision import Mode


@dataclass
class ECFHESession:
    """The session a user holds: keys + a BSGS table sized to the
    workload, plus the chosen evaluator backend.

    Two evaluation backends are exposed via the `backend` parameter:

      "bootstrap_assisted" (default) - every homomorphic op partial-
            decrypts, computes on Q_E rationals, re-encrypts. Strict
            paper semantics, latency dominated by bootstrap.

      "ct_native" - homomorphic add stays in G1 (no bootstrap); homo-
            morphic mul uses LSL24 Thm 8 ★_ct via Weil pairing, lands
            in GT, returns a fresh G1 ciphertext after GT-decrypt. Much
            faster for multiplication-heavy workloads. Requires a
            secret-key holder to be co-located with the evaluator
            (typical for single-tenant FHE-as-a-service)."""
    keypair: KeyPair
    bsgs: BSGSTable
    backend: str = "bootstrap_assisted"
    evaluator: Evaluator = field(init=False)

    def __post_init__(self) -> None:
        self.evaluator = Evaluator(
            pk=self.keypair.public,
            bsk=self.keypair.bootstrap,
            sk=self.keypair.secret,
            bsgs_max=self.bsgs.max_value,
            backend=self.backend,
        )
        self.evaluator._bsgs = self.bsgs

    @classmethod
    def create(cls, *, bsgs_max: int = 10_000,
               security_param: int = 128,
               delta_max: int | None = None,
               backend: str = "bootstrap_assisted") -> "ECFHESession":
        return cls(
            keypair=keygen(security_param=security_param, delta_max=delta_max),
            bsgs=BSGSTable(bsgs_max),
            backend=backend,
        )

    #  Encrypted-DOT via GT-deep aggregation (LSL24 Thm 8 + summation) -
    def encrypted_dot(self, xs: list, ys: list,
                      gt_max: int | None = None):
        """Σ xᵢ*yᵢ on encrypted-then-aggregated data.

        Each xᵢ, yᵢ is a Python int (the dot product is integer-valued).
        Returns the cleartext sum as an int. The sum is computed via N
        pairings + one GT BSGS, NOT N decrypts + N-1 plaintext additions;
        the server-side computation is over ciphertexts throughout."""
        from .gt_bsgs import GTTable
        from .star_ct import (decrypt_gt_to_scalar, encrypt_paired,
                              encrypted_dot_product)
        from .rational import EllipticRational

        pa = [encrypt_paired(self.keypair.public,
                             EllipticRational(x, 1)) for x in xs]
        pb = [encrypt_paired(self.keypair.public,
                             EllipticRational(y, 1)) for y in ys]
        gt_ct = encrypted_dot_product(pa, pb)

        if gt_max is None:
            # Σ xᵢ*yᵢ bound. Caller can pass tighter bound for speed.
            gt_max = sum(abs(x) * abs(y) for x, y in zip(xs, ys)) or 1
        gt_table = GTTable(max_value=gt_max)
        num, _den = decrypt_gt_to_scalar(gt_ct, self.keypair.secret.s, gt_table)
        return num

    # External Layer entry points (EC_FHE §3.1)
    def encrypt(self, x, delta: int | None = None,
                mode: Mode = "standard") -> Ciphertext:
        return encrypt(self.keypair.public, x, delta=delta, mode=mode)

    def decrypt(self, ct: Ciphertext, *, thorough: bool = False) -> Fraction:
        x_prime, _ = decrypt(self.keypair.secret, ct, self.bsgs,
                             mode="thorough" if thorough else "fast")
        return x_prime

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        return self.evaluator.add(ct1, ct2)

    def sub(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        return self.evaluator.sub(ct1, ct2)

    def mul(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        return self.evaluator.mul(ct1, ct2)

    def div(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        return self.evaluator.div(ct1, ct2)

    def scalar_mul(self, ct: Ciphertext, k: int) -> Ciphertext:
        return self.evaluator.scalar_mul(ct, k)

    def scalar_div(self, ct: Ciphertext, k: int) -> Ciphertext:
        return self.evaluator.scalar_div(ct, k)

    def bootstrap(self, ct: Ciphertext) -> Ciphertext:
        return bootstrap_precision_preserving(
            ct, pk=self.keypair.public, bsk=self.keypair.bootstrap,
            bsgs=self.bsgs)

    # Algorithm 1 - System Data Flow with Precision Management
    def evaluate(self, f: Callable[[Iterable[Ciphertext]], Ciphertext],
                 reals: list, *, deltas: list[int] | None = None,
                 budget: Budget | None = None) -> tuple[Fraction, dict]:
        """End-to-end evaluation:  x_1..x_k (real)  →  y (real) + error cert.

        Validates inputs, encrypts each x_i at its requested δ_i, runs the
        user-supplied function f over the ciphertexts, decrypts the result.
        Returns (y', certificate-dict)."""
        if deltas is None:
            deltas = [self.keypair.public.delta_max] * len(reals)
        if len(deltas) != len(reals):
            raise ValueError("deltas must match reals in length")

        cts = [self.encrypt(x, delta=d) for x, d in zip(reals, deltas)]
        result_ct = f(cts)
        y_prime = self.decrypt(result_ct, thorough=True)

        cert = {
            "delta_in": deltas,
            "delta_out": result_ct.delta,
            "noise": {
                "approx": result_ct.noise.approx,
                "comp": result_ct.noise.comp,
                "crypto": result_ct.noise.crypto,
                "total": result_ct.noise.total,
            },
            "budget_violation": (budget is not None
                                 and result_ct.noise.total > budget.base),
        }
        return y_prime, cert
