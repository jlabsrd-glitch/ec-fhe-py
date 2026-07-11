"""EC-FHE - Paper-faithful implementation.

This package implements the elliptic-curve fully homomorphic encryption scheme
defined across two papers:

    [P1] "Rational Number Representation over Elliptic Curves via Equivalence
         Relations: Extending ECHC to Fractional Scalar Multiplication"
         (Anonymous, 2025-08-12).  Cited below as ECHCoverQ.

    [P2] "Extension to Real Number Representations over Elliptic Curves:
         Precision-Controlled Approximation and Fully Homomorphic Encryption
         with Advanced Bootstrapping and Noise Management"
         (Anonymous, 2025-08-14).  Cited below as EC_FHE.

Module → paper mapping (every algorithm number and definition is the paper's;
this is intentional, so reviewers / auditors can grep both ways):

    params.py        EC_FHE §10.1 (curve / pairing parameter selection)
    signed.py        ECHCoverQ §3  (signed elliptic integers, cann)
    rational.py      ECHCoverQ §4  (Q_E equivalence, +, canonical form)
    echc.py          ECHCoverQ §4.3 + Lee/Shim/Lee 2024 (★ formal multiplication)
    ciphertext.py    EC_FHE §7.2  ((C1,C2,C3,δ,η) layout)
    keygen.py        EC_FHE §8.2 / Alg 8
    encrypt.py       EC_FHE §8.3 / Alg 9 (3 modes)
    decrypt.py       EC_FHE §8.5 / Alg 11
    noise.py         EC_FHE §6.1 / Def 6.1 + Thms 6.3, 6.4
    precision.py     EC_FHE §5 + Thm 4.2 / 4.3 / 5.1
    bootstrap.py     EC_FHE §7 / Alg 5, 6, 7  + Thm 7.2
    evaluate.py      EC_FHE §8.4 / Alg 10  (+ predictive Alg 4)
    threshold.py     EC_FHE / Shamir t-of-n
    interface.py     EC_FHE §3  (Two-Layer External Layer)

The ★ formal multiplication on encrypted scalars (echc.py) is the one
component whose construction lives in a separate referenced paper
("Homomorphic-based Encryption using Weil Pairing", Lee/Shim/Lee 2024).
We provide a pairing-based implementation isolated behind a stable
interface so it can be swapped for the exact paper construction without
rippling through the rest of the package.
"""

__version__ = "0.2.0"

from .signed import SignedInt, cann
from .rational import EllipticRational, is_equivalence
from .echc import star
from .keygen import KeyPair, PublicKey, SecretKey, BootstrapKey, keygen
from .ciphertext import Ciphertext
from .noise import Noise, Budget
from .precision import real_to_elliptic_rational, ConversionCert
from .encrypt import encrypt
from .decrypt import (
    decrypt, decrypt_to_rational, BSGSTable, ErrorCertificate,
    FullDomainTable, DecryptCache, make_dlog_table,
)
from .evaluate import Evaluator
from .bootstrap import (
    bootstrap_precision_preserving,
    bootstrap_precision_enhancing,
    selective_bootstrap_schedule,
)
from .threshold import Share, PartialDec, split, partial_decrypt, combine_partials
from .predicate import (
    zero_test, randomized_zero_test, ct_minus_plain, ct_scalar_sub,
    equals_plain, positive_set_membership, positive_set_for,
)
from .interface import ECFHESession

__all__ = [
    "__version__",
    # signed integers (ECHCoverQ §3)
    "SignedInt", "cann",
    # Q_E rationals (ECHCoverQ §4)
    "EllipticRational", "is_equivalence", "star",
    # keys + ciphertexts
    "KeyPair", "PublicKey", "SecretKey", "BootstrapKey", "keygen",
    "Ciphertext", "Noise", "Budget",
    # encrypt / decrypt
    "real_to_elliptic_rational", "ConversionCert",
    "encrypt", "decrypt", "decrypt_to_rational",
    "BSGSTable", "FullDomainTable", "DecryptCache", "make_dlog_table",
    "ErrorCertificate",
    # evaluate + bootstrap
    "Evaluator",
    "bootstrap_precision_preserving",
    "bootstrap_precision_enhancing",
    "selective_bootstrap_schedule",
    # threshold
    "Share", "PartialDec", "split", "partial_decrypt", "combine_partials",
    # positive-set predicates
    "zero_test", "randomized_zero_test", "ct_minus_plain", "ct_scalar_sub",
    "equals_plain", "positive_set_membership", "positive_set_for",
    # external layer
    "ECFHESession",
]
