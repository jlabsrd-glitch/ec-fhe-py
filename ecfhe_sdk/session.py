"""FHESession — high-level entry point for EC-FHE operations."""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import List

# Make the engine importable when installed from inside the repo
_here = os.path.dirname(__file__)
_engine_root = os.path.abspath(os.path.join(_here, "..", "..", "..", "ec_fhe_engine"))
if os.path.isdir(_engine_root) and _engine_root not in sys.path:
    sys.path.insert(0, _engine_root)

from ecfhe.interface import ECFHESession as _ECFHESession
from .ciphertext import FHECiphertext


class FHESession:
    """A single-party FHE session: holds keypair + BSGS precomputation.

    Parameters
    ----------
    max_value : int
        Largest absolute plaintext value you will encrypt or expect as a
        result.  Larger values require more memory (O(√max_value) for
        the BSGS table) and a longer warm-up on first ``decrypt``.
        Default: 10 000.
    backend : str
        ``"bootstrap_assisted"`` (default, strict paper semantics) or
        ``"ct_native"`` (faster, uses Weil pairing for multiplication).
    """

    def __init__(self, max_value: int = 10_000,
                 backend: str = "bootstrap_assisted"):
        self._inner = _ECFHESession.create(
            bsgs_max=max_value, backend=backend
        )
        self._max_value = max_value

    # ── Encrypt / decrypt ──────────────────────────────────────────────────

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def delta_max(self) -> int:
        """Maximum precision (decimal digits) this session supports.

        Controlled at construction time by ``max_value``; typically 59
        on BN254 (log10(ORDER) - 16 headroom). Use as the upper bound
        for the ``delta`` argument of ``encrypt_real``.
        """
        return self._inner.keypair.public.delta_max

    # ── Encrypt / decrypt ──────────────────────────────────────────────────

    def encrypt(self, value: int) -> FHECiphertext:
        """Encrypt a plaintext integer."""
        if not isinstance(value, int):
            raise TypeError(f"expected int, got {type(value).__name__}")
        ct = self._inner.encrypt(value)
        return FHECiphertext(ct, self)

    def encrypt_real(self, value: float,
                     delta: int | None = None,
                     mode: str = "standard") -> FHECiphertext:
        """Encrypt a real (decimal) number with configurable precision.

        Parameters
        ----------
        value : float | Decimal | Fraction
            The value to encrypt. Integers are also accepted.
        delta : int, optional
            Precision in decimal digits. Approximation error ≤ 10^{-delta}.
            Defaults to ``self.delta_max`` (maximum for this session).
        mode : str
            Conversion algorithm:

            * ``"standard"``       — a = floor(x · 10^δ), b = 10^δ.
                                     Error ≤ 10^{-δ}. Fastest.
            * ``"optimized"``      — continued-fraction convergent with
                                     q ≤ 10^{δ/2}. Same error bound,
                                     smaller denominator (better noise).
            * ``"high_precision"`` — targets error ≤ 10^{-(δ+5)} by
                                     running both methods at δ+5 and
                                     picking the better one.

        Notes
        -----
        ``max_value`` (set at ``FHESession(max_value=...)`` construction)
        must cover the rational numerator AND denominator after encoding.
        For ``standard`` mode at precision δ, the denominator is 10^δ and
        the numerator is approximately ``|value| * 10^δ``.  A safe rule of
        thumb: ``max_value ≥ max(|value|, 1) * 10^delta * 10``.
        """
        if delta is None:
            delta = self._inner.keypair.public.delta_max
        ct = self._inner.encrypt(value, delta=delta, mode=mode)
        return FHECiphertext(ct, self)

    def decrypt(self, ct: FHECiphertext) -> int:
        """Decrypt a ciphertext, returning the recovered integer (rounded)."""
        result: Fraction = self._inner.decrypt(ct._ct)
        num, den = result.numerator, result.denominator
        if den == 0:
            raise ArithmeticError("decryption produced zero denominator")
        return round(num / den)

    def decrypt_real(self, ct: FHECiphertext) -> float:
        """Decrypt a ciphertext, returning the exact rational value as a float.

        Unlike ``decrypt``, this does NOT round to the nearest integer —
        use it when the encrypted value was produced by ``encrypt_real``.
        """
        result: Fraction = self._inner.decrypt(ct._ct)
        if result.denominator == 0:
            raise ArithmeticError("decryption produced zero denominator")
        return float(result)

    # ── Homomorphic arithmetic ─────────────────────────────────────────────

    def add(self, a: FHECiphertext, b: FHECiphertext) -> FHECiphertext:
        """Ciphertext addition: enc(a) + enc(b) = enc(a+b)."""
        ct = self._inner.add(a._ct, b._ct)
        return FHECiphertext(ct, self)

    def sub(self, a: FHECiphertext, b: FHECiphertext) -> FHECiphertext:
        """Ciphertext subtraction: enc(a) - enc(b) = enc(a-b)."""
        neg_b = self._inner.scalar_mul(b._ct, -1)
        ct = self._inner.add(a._ct, neg_b)
        return FHECiphertext(ct, self)

    def mul(self, a: FHECiphertext, b: FHECiphertext) -> FHECiphertext:
        """Ciphertext multiplication: enc(a) * enc(b) = enc(a*b).

        Uses Weil pairing (★_ct / LSL24 Thm 8). The result is a fresh
        G1 ciphertext — no pairing output is exposed to the caller.
        Requires ``backend="ct_native"`` for best performance.
        """
        ct = self._inner.mul(a._ct, b._ct)
        return FHECiphertext(ct, self)

    def mul_scalar(self, ct: FHECiphertext, k: int) -> FHECiphertext:
        """Scalar multiplication: enc(a) * k = enc(a*k).

        Faster than ``mul`` — implemented as repeated point doubling,
        no pairing needed.
        """
        result = self._inner.scalar_mul(ct._ct, k)
        return FHECiphertext(result, self)

    def div(self, a: FHECiphertext, b: FHECiphertext) -> FHECiphertext:
        """Ciphertext division: enc(a) / enc(b) = enc(a/b).

        Uses the multiplicative inverse in Q_E (Thm 5.2):
        enc(a) / enc(b) = enc(a) * enc(b).inverse().
        Division by an encrypted zero raises ZeroDivisionError.
        Non-exact quotients are rounded to the nearest integer on decrypt.
        """
        ct = self._inner.div(a._ct, b._ct)
        return FHECiphertext(ct, self)

    def div_scalar(self, ct: FHECiphertext, k: int) -> FHECiphertext:
        """Divide enc(a) by a known plaintext constant k → enc(a/k).

        Faster than div() — no second ciphertext needed.
        Non-exact quotients are rounded to the nearest integer on decrypt.
        """
        if k == 0:
            raise ZeroDivisionError
        result = self._inner.scalar_div(ct._ct, k)
        return FHECiphertext(result, self)

    # ── Batch helpers ──────────────────────────────────────────────────────

    def sum(self, cts: List[FHECiphertext]) -> FHECiphertext:
        """Homomorphic sum of a list of ciphertexts."""
        if not cts:
            raise ValueError("empty list")
        result = cts[0]
        for ct in cts[1:]:
            result = self.add(result, ct)
        return result

    def dot(self, xs: List[int], ys: List[int]) -> int:
        """Encrypted dot product Σ xᵢ*yᵢ.

        Uses GT-level aggregation (N pairings + one GT-BSGS) — the
        server never sees individual values.
        """
        return self._inner.encrypted_dot(xs, ys)

    # ── Serialisation helpers ──────────────────────────────────────────────

    def ciphertext_from_bytes(self, data: bytes) -> FHECiphertext:
        return FHECiphertext.from_bytes(data, self)

    def ciphertext_from_json(self, s: str) -> FHECiphertext:
        return FHECiphertext.from_json(s, self)

    # ── Key export (for multi-party scenarios) ─────────────────────────────

    @property
    def public_key(self) -> dict:
        """Export the public key as a dict (safe to share).

        The public key allows any holder to *encrypt* values for this
        session, but not to decrypt. Use ``FHESession.from_public_key``
        on the remote party to create an encrypt-only session.
        """
        pk = self._inner.keypair.public
        from ecfhe import params as _p
        ax, ay = _p.normalize(pk.Q)
        def _int(v):
            return v.n if hasattr(v, "n") else int(v)
        return {
            "curve": "BN254",
            "Q": {"x": _int(ax), "y": _int(ay)},
            "delta_max": pk.delta_max,
            "n": str(pk.n),
        }

    @classmethod
    def from_public_key(cls, pk_dict: dict,
                        max_value: int = 10_000) -> "FHESession":
        """Create an *encrypt-only* session from an exported public key.

        The returned session can call ``encrypt`` but not ``decrypt``
        or any method that requires the secret key.
        """
        raise NotImplementedError(
            "Encrypt-only sessions require reconstructing the PublicKey "
            "object from the dict. Use FHESession() for a full session "
            "or pass ciphertexts via JSON to a remote decrypt endpoint."
        )

    def __repr__(self) -> str:
        return (f"<FHESession max_value={self._max_value} "
                f"backend={self._inner.backend!r}>")
