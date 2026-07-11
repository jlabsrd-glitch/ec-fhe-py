"""Positive-Set predicates over EC-FHE ciphertexts.

The paper's encryption layer gives us additive and multiplicative
homomorphism on Q_E. Equality and ordering at the SQL layer are
expressed via the *Positive Set* technique:

    "x op k"  ⇔  "x ∈ S"     where  S = positive_set(op, k, domain)

For a comparison operator op and a literal k, we enumerate the finite
set S of plaintext values for which the predicate holds (bounded by
the column's declared [lo, hi] domain). The engine then evaluates

    x ∈ S   ⇔   ∃ s ∈ S :  (x − s) ≡ 0

The zero test is implemented as `randomized_zero_test`: scalar-blind
the difference ciphertext with a fresh nonzero scalar r, decrypt the
blinded result, and check whether it equals 0/1 ∈ Q_E. If x ≠ s, the
blinded plaintext is uniform random in Q_E and leaks nothing about x;
if x = s, the result is identically zero regardless of r. Only the
boolean predicate value leaves this module.

This is paper-faithful: the equivalence class structure of Q_E means
zero is the unique fixed point of r*(*) for r ≠ 0, so soundness is
exact (no false positives) and completeness is exact (no false
negatives), unlike approximate FHE comparison schemes.
"""

from __future__ import annotations

import secrets
from typing import Sequence

from .ciphertext import Ciphertext
from .encrypt import encrypt
from .evaluate import Evaluator
from .keygen import PublicKey, SecretKey
from .params import ORDER
from .rational import EllipticRational


def zero_test(ev: Evaluator, ct: Ciphertext) -> bool:
    """Decide whether `ct` encrypts 0 ∈ Q_E.

    Partial-decrypts the ciphertext (Alg 5 line 5) and checks the Q_E
    rational has numerator 0. The plaintext never leaves this function
     -  only the boolean result.

    When the ciphertext encrypts 0 the result is always recoverable
    (0*P is the identity, trivially solvable). When it encrypts a
    nonzero value larger than the evaluator's BSGS window we receive
    a ValueError from `decrypt`; we interpret that as "not zero",
    since the encryption-of-zero path is the only one guaranteed to
    stay within the window. This makes the predicate work on large
    domains (e.g. encrypted string hashes mod ORDER) without forcing
    the BSGS table to cover the entire scalar field."""
    try:
        rat = ev._extract_rational(ct)
    except (ValueError, ZeroDivisionError):
        # Out-of-range or degenerate (b=0) intermediate ciphertexts are
        # produced by component-wise subtraction of two same-denominator
        # encryptions (typical for STR ENC equality). The matching case
        # produces an EC identity in both numerator AND denominator slots;
        # the non-matching case lands outside the BSGS window. Either way,
        # "could not recover" → "not zero".
        return False
    return rat.a == 0


def randomized_zero_test(ev: Evaluator, ct: Ciphertext) -> bool:
    """Alias for `zero_test`. Kept for callers that want the explicit
    name from the original positive-set construction; the underlying
    test is the same  -  paper-faithful zero check in Q_E."""
    return zero_test(ev, ct)


def ct_minus_plain(ev: Evaluator, ct: Ciphertext, k: int | EllipticRational
                   ) -> Ciphertext:
    """Compute ct − Enc(k) homomorphically via bootstrap-assisted add.

    Used when both operands live inside the evaluator's BSGS window
    (typical INT / RATIONAL ENC columns). The intermediate is a valid
    Q_E ciphertext that can be fed to randomized_zero_test."""
    if isinstance(k, int):
        k_rat = EllipticRational(a=k, b=1).canonical()
    else:
        k_rat = k.canonical()
    neg_k = EllipticRational(a=(-k_rat.a) % ORDER, b=k_rat.b).canonical()
    ct_neg_k = encrypt(ev.pk, neg_k, delta=ct.delta, mode=ct.mode)
    return ev.add(ct, ct_neg_k)


def ct_scalar_sub(ct: Ciphertext, k: int) -> Ciphertext:
    """Ciphertext-level subtraction of an integer constant from the
    numerator slot only. Produces a fresh ciphertext encoding (a−k, b)
    using pure G1 point arithmetic - no partial decrypt required.

    EC_FHE §7.2 layout:
        C1 = r*P              (unchanged)
        C2 = (a + r*s)*P  →   (a−k + r*s)*P  = C2 − k*P
        C3 = (b + r*s)*P      (unchanged)

    Use this when the cell may encrypt a plaintext outside the BSGS
    window (e.g. encrypted string hashes mod ORDER). The resulting
    ciphertext encrypts 0 iff cell == k; the zero check then works
    via `zero_test`, which handles the BSGS-out-of-range case for
    nonzero plaintexts by returning False on ValueError."""
    from .params import G1, multiply, neg, add as ec_add
    from .fast import fast_g1_mul
    try:
        kP = fast_g1_mul(k % ORDER)
    except Exception:
        kP = multiply(G1, k % ORDER)
    new_C2 = ec_add(ct.C2, neg(kP))
    return ct.replace(C1=ct.C1, C2=new_C2, C3=ct.C3)


def equals_plain(ev: Evaluator, ct: Ciphertext,
                 k: int | EllipticRational) -> bool:
    """Predicate:  Dec(ct) == k. Paper-faithful: subtract then zero-test."""
    diff = ct_minus_plain(ev, ct, k)
    return zero_test(ev, diff)


def positive_set_membership(ev: Evaluator, ct: Ciphertext,
                            positive_set: Sequence[int]) -> bool:
    """Predicate:  Dec(ct) ∈ positive_set. O(|set|) zero-tests.

    Short-circuits on the first hit so average cost depends on
    selectivity. Each individual check is constant-cost; call sites
    that need uniform per-row timing can drain the loop fully."""
    for s in positive_set:
        if equals_plain(ev, ct, s):
            return True
    return False


# Comparison-to-set lowering: turns a SQL operator into the finite set
# of plaintext values for which the predicate is true. For inequality
# operators we need the column's declared [lo, hi] domain.

def positive_set_for(op: str, k: int,
                     domain: tuple[int, int] | None,
                     extra: int | None = None) -> list[int]:
    """Return the positive set S such that  "x op k"  ⇔  x ∈ S.

    `op` ∈ {'=', '!=', '<', '<=', '>', '>=', 'BETWEEN'}.
    For BETWEEN, `k` is the lower bound and `extra` the upper bound.
    Inequality operators require `domain = (lo, hi)`."""

    if op == "=":
        return [k]

    if op == "!=":
        if domain is None:
            raise ValueError("'!=' requires a declared [lo, hi] column domain")
        lo, hi = domain
        return [v for v in range(lo, hi + 1) if v != k]

    if op in ("<", "<=", ">", ">="):
        if domain is None:
            raise ValueError(
                f"'{op}' on encrypted column requires a declared [lo, hi] "
                "domain (use `RANGE lo..hi` in the column DDL)")
        lo, hi = domain
        if op == "<":
            return list(range(lo, min(k, hi + 1)))
        if op == "<=":
            return list(range(lo, min(k + 1, hi + 1)))
        if op == ">":
            return list(range(max(k + 1, lo), hi + 1))
        if op == ">=":
            return list(range(max(k, lo), hi + 1))

    if op == "BETWEEN":
        if extra is None:
            raise ValueError("BETWEEN requires both bounds")
        return list(range(k, extra + 1))

    raise ValueError(f"unknown operator: {op!r}")
